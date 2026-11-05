import { Database } from "bun:sqlite"
import { mkdir, readFile, unlink, open } from "node:fs/promises"
import path from "node:path"
import { AppError, VERSION, type Config, type Message } from "./contracts"
import { id, now, writeJSON } from "./util"

export class Store {
  db: Database
  constructor(public root: string, readonlyMode = false) {
    this.db = new Database(path.join(root, "runtime/state.sqlite"), readonlyMode ? {readonly:true} : {create:true})
    if(readonlyMode) return
    this.db.run("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA busy_timeout=5000;")
    this.db.run(`CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS messages(seq INTEGER PRIMARY KEY AUTOINCREMENT,session TEXT NOT NULL,value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,at TEXT NOT NULL,type TEXT NOT NULL,value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS operations(id TEXT PRIMARY KEY,status TEXT NOT NULL,value TEXT,at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS calls(id TEXT PRIMARY KEY,status TEXT NOT NULL,reserved INTEGER NOT NULL,used INTEGER,at TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS messages_session ON messages(session,seq);`)
    const version = this.get<string>("version")
    if (version && version !== VERSION) { this.db.close(); throw new AppError("VERSION_MISMATCH", `Workspace requires version ${version}`, 3) }
    this.set("version", VERSION)
  }
  get<T = any>(key: string): T | undefined {
    const row = this.db.query("SELECT value FROM kv WHERE key=?").get(key) as { value: string } | null
    return row ? JSON.parse(row.value) : undefined
  }
  set(key: string, value: unknown) { this.db.query("INSERT OR REPLACE INTO kv VALUES(?,?)").run(key, JSON.stringify(value)) }
  transaction<T>(fn: () => T): T { return this.db.transaction(fn)() }
  event(type: string, value: unknown = {}) {
    this.db.query("INSERT INTO events(at,type,value) VALUES(?,?,?)").run(now(), type, JSON.stringify(value))
  }
  messages(session: string): Message[] {
    return (this.db.query("SELECT value FROM messages WHERE session=? ORDER BY seq").all(session) as {value:string}[]).map(r => JSON.parse(r.value))
  }
  append(session: string, message: Message) { this.db.query("INSERT INTO messages(session,value) VALUES(?,?)").run(session, JSON.stringify(message)) }
  operation(key: string): { status: string; value: unknown } | undefined {
    const row = this.db.query("SELECT status,value FROM operations WHERE id=?").get(key) as {status:string;value:string} | null
    return row ? { status: row.status, value: row.value ? JSON.parse(row.value) : undefined } : undefined
  }
  saveOperation(key: string, status: string, value: unknown = null) {
    this.db.query("INSERT OR REPLACE INTO operations VALUES(?,?,?,?)").run(key, status, JSON.stringify(value), now())
  }
  reserve(tokens: number, config: Config) {
    return this.transaction(() => {
      const row = this.db.query("SELECT COUNT(*) AS n,COALESCE(SUM(COALESCE(used,reserved)),0) AS tokens FROM calls").get() as {n:number;tokens:number}
      if (row.n >= config.execution.max_total_llm_calls || row.tokens + tokens > config.execution.max_total_tokens)
        throw new AppError("BUDGET_EXHAUSTED", "Global LLM budget exhausted", 5, true)
      const callId = id("llm")
      this.db.query("INSERT INTO calls VALUES(?,?,?,NULL,?)").run(callId, "started", tokens, now())
      return callId
    })
  }
  settle(callId: string, status: string, used?: number) {
    this.db.query("UPDATE calls SET status=?,used=? WHERE id=? AND status='started'").run(status, used ?? null, callId)
  }
  usage() { return this.db.query("SELECT COUNT(*) AS calls,COALESCE(SUM(COALESCE(used,reserved)),0) AS tokens FROM calls").get() }
  async project() {
    const rows = this.db.query("SELECT seq,at,type,value FROM events ORDER BY seq").all() as {seq:number;at:string;type:string;value:string}[]
    await Bun.write(path.join(this.root, "runtime/events.jsonl"), rows.map(r => JSON.stringify({...r,value:JSON.parse(r.value)})).join("\n") + "\n")
    await writeJSON(path.join(this.root, "runtime/status.json"), { ...this.get<Record<string, unknown>>("state"), usage: this.usage() })
  }
  close() { this.db.close() }
}

export async function lockWorkspace(workspace: string) {
  await mkdir(path.join(workspace, ".forecast/runtime"), { recursive: true })
  const db=new Database(path.join(workspace,".forecast/runtime/state.sqlite"),{create:true})
  db.run("PRAGMA busy_timeout=5000")
  db.run("CREATE TABLE IF NOT EXISTS executor_lock(id INTEGER PRIMARY KEY,pid INTEGER NOT NULL,token TEXT NOT NULL)")
  const token = id("lock")
  try {
    db.transaction(()=>{
      const owner=db.query("SELECT pid FROM executor_lock WHERE id=1").get() as {pid:number}|null
      if(owner) {
        try {process.kill(owner.pid,0);throw new AppError("WORKSPACE_LOCKED","Another executor is using this workspace",3)}
        catch(e:any) {if(e.code!=="ESRCH") throw e}
      }
      db.query("INSERT OR REPLACE INTO executor_lock VALUES(1,?,?)").run(process.pid,token)
    }).immediate()
  } catch(e) {db.close();throw e}
  return async()=>{try {db.query("DELETE FROM executor_lock WHERE id=1 AND token=?").run(token)} finally {db.close()}}
}
