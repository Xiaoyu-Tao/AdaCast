import { test,expect } from "bun:test"
import { mkdir } from "node:fs/promises"
import path from "node:path"
import { z } from "zod"
import { fixture } from "./helpers"
import { ConfigSchema } from "../src/contracts"
import { Store } from "../src/storage"
import { buildContext,messageGroups } from "../src/context"
import { AgentRuntime } from "../src/runtime"
import { LLMClient } from "../src/llm"
import { ForecastService } from "../src/forecast"

test("configuration rejects ambiguous splits and implicit reflection in deterministic mode",async()=>{
  const f=await fixture()
  try {
    expect(()=>ConfigSchema.parse({...f.config,task:{...f.config.task,train_end_timestamp:"2025-01-02"}})).toThrow()
    expect(()=>ConfigSchema.parse({...f.config,task:{...f.config.task,reflection_enabled:true}})).toThrow()
  } finally {await f.cleanup()}
})

test("context compaction retains complete tool groups and preserves durable constraints",async()=>{
  const f=await fixture()
  const root=path.join(f.dir,"store");await mkdir(path.join(root,"runtime"),{recursive:true})
  const store=new Store(root)
  try {
    store.set("handoff",{constraint:"training scale is frozen"})
    for(let i=0;i<20;i++) {
      store.append("s",{role:"user",content:"Task round "+i})
      store.append("s",{role:"assistant",content:null,tool_calls:[{id:"c"+i,type:"function",function:{name:"tool",arguments:"{}"}}]})
      store.append("s",{role:"tool",tool_call_id:"c"+i,content:"result ".repeat(200)})
    }
    const context=await buildContext(store,"s","Protect the frozen task",[],4000,500,async()=>"training scale is frozen; preserve the pending task.")
    expect(context.length).toBeLessThan(store.messages("s").length)
    expect(JSON.stringify(context)).toContain("training scale is frozen")
    expect(()=>messageGroups(context)).not.toThrow()
    expect(()=>messageGroups([{role:"tool",content:"orphan",tool_call_id:"x"}])).toThrow("Orphan")
  } finally {store.close();await f.cleanup()}
})

test("global LLM reservations count unknown attempts conservatively",async()=>{
  const f=await fixture({execution:{max_total_llm_calls:2,max_total_tokens:1000}})
  const root=path.join(f.dir,"store");await mkdir(path.join(root,"runtime"),{recursive:true})
  const store=new Store(root)
  try {
    const first=store.reserve(400,f.config)
    store.settle(first,"received",100)
    store.reserve(400,f.config)
    expect(()=>store.reserve(10,f.config)).toThrow("budget exhausted")
    expect((store.usage() as any).tokens).toBe(500)
  } finally {store.close();await f.cleanup()}
})

test("tool recovery reuses committed results without repeating side effects",async()=>{
  const f=await fixture()
  const root=path.join(f.dir,"store");await mkdir(path.join(root,"runtime"),{recursive:true})
  const store=new Store(root)
  try {
    const runtime=new AgentRuntime(f.config,store,new LLMClient(f.config,store),()=>"",new AbortController().signal)
    let count=0
    runtime.register({name:"count",description:"count",schema:z.object({}),roles:["*"],replaySafe:true,async execute(){return {count:++count}}})
    store.append("s",{role:"assistant",content:null,tool_calls:[{id:"c",type:"function",function:{name:"count",arguments:"{}"}}]})
    await runtime.executePending("s","general",0)
    await runtime.executePending("s","general",0)
    expect(count).toBe(1)
    expect(store.messages("s").filter(m=>m.role==="tool")).toHaveLength(1)
  } finally {store.close();await f.cleanup()}
})

test("a resumed helper accepts a new prompt without discarding its isolated history",async()=>{
  const f=await fixture({execution:{mode:"agent"},llm:{model:"mock",base_url:"http://127.0.0.1:1"}})
  const root=path.join(f.dir,"store");await mkdir(path.join(root,"runtime"),{recursive:true})
  const store=new Store(root)
  try {
    let calls=0
    const client={async complete(messages:any[]){calls++;return {role:"assistant",content:"Response to "+messages.at(-1).content}}} as unknown as LLMClient
    const runtime=new AgentRuntime(f.config,store,client,()=>"Isolated helper",new AbortController().signal)
    const first=await runtime.run("general","sub-test","First task")
    const second=await runtime.run("general","sub-test","Follow-up task",undefined,0,true)
    expect(first).toContain("First task")
    expect(second).toContain("Follow-up task")
    expect(store.messages("sub-test").filter(m=>m.role==="user")).toHaveLength(2)
    await runtime.run("general","sub-test","Follow-up task",undefined,0,true)
    expect(calls).toBe(2)
  } finally {store.close();await f.cleanup()}
})

test("a fitted candidate cannot bypass the fatal consecutive-crash stop",async()=>{
  const f=await fixture({task:{model_families:["Naive","Holt"]},execution:{max_consecutive_crashes:1}})
  const root=path.join(f.dir,"store");await mkdir(path.join(root,"runtime"),{recursive:true})
  const store=new Store(root)
  try {
    store.set("state",{phase:"forecast"})
    const service=new ForecastService(f.config,store,new AbortController().signal)
    const valid=await service.experiment("Naive")
    await service.reflect(valid.run_id,"keep","A valid model exists before the intentionally invalid experiment.")
    await expect(service.experiment("Holt",{alpha:2})).rejects.toThrow("Consecutive crash threshold")
    await expect(service.lockSelection()).rejects.toThrow("Consecutive crash threshold")
  } finally {store.close();await f.cleanup()}
})


test("semantic compaction persists the original summary and reuses it after context reconstruction",async()=>{
  const f=await fixture()
  const root=path.join(f.dir,"semantic-store");await mkdir(path.join(root,"runtime"),{recursive:true})
  const store=new Store(root)
  try {
    store.set("handoff",{phase:"forecast",constraint:"Training scaler and validation sources are frozen"})
    for(let i=0;i<12;i++)store.append("semantic",{role:"user",content:"Observed experiment and next action "+i+" x".repeat(1200)})
    let calls=0
    const compact=async(messages:any[])=>{calls++;expect(messages[0].content).toContain("summarizing conversations");expect(messages.at(-1).content).toContain("validation sources");return "Frozen sources and scaler must be reused. Continue the pending experiment reflection."}
    const first=await buildContext(store,"semantic","Frozen task",[],4000,500,compact)
    const recovered=await buildContext(store,"semantic","Frozen task",[],4000,500,compact)
    expect(calls).toBeGreaterThan(0)
    const afterFirst=calls
    await buildContext(store,"semantic","Frozen task",[],4000,500,compact)
    expect(calls).toBe(afterFirst)
    expect(first).toEqual(recovered)
    expect(JSON.stringify(first)).toContain("Continue the pending experiment reflection")
  }finally{store.close();await f.cleanup()}
})
