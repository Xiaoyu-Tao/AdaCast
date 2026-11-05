import {test,expect} from "bun:test"
import path from "node:path"
import {mkdir} from "node:fs/promises"
import {fixture} from "./helpers"
import {Store} from "../src/storage"
import {LLMClient} from "../src/llm"

test("transport timeouts use the configured bounded HTTP retry budget",async()=>{
  const f=await fixture({execution:{mode:"agent"},llm:{base_url:"http://127.0.0.1:1/v1",model:"test",max_retries:1}})
  const root=path.join(f.workspace,".forecast");await mkdir(path.join(root,"runtime"),{recursive:true})
  const store=new Store(root),original=globalThis.fetch
  let attempts=0
  globalThis.fetch=(async()=>{
    if(++attempts===1)throw Object.assign(new Error("The operation timed out."),{name:"TimeoutError",code:"ETIMEDOUT"})
    return Response.json({choices:[{finish_reason:"stop",message:{role:"assistant",content:"OK"}}],usage:{total_tokens:2}})
  }) as unknown as typeof fetch
  try {
    const result=await new LLMClient(f.config,store).complete([{role:"user",content:"test"}],[],"transport-retry",new AbortController().signal)
    expect(result.content).toBe("OK");expect(attempts).toBe(2)
    expect(store.db.query("SELECT COUNT(*) AS count FROM calls WHERE status='failed'").get()).toEqual({count:1})
    expect(store.db.query("SELECT COUNT(*) AS count FROM calls WHERE status='received'").get()).toEqual({count:1})
  } finally {globalThis.fetch=original;store.close();await f.cleanup()}
})
