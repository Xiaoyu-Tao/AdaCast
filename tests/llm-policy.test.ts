import {test,expect} from "bun:test"
import {mkdir,readFile} from "node:fs/promises"
import path from "node:path"
import {fixture} from "./helpers"
import {Store} from "../src/storage"
import {LLMClient} from "../src/llm"
import {AgentRuntime} from "../src/runtime"
import {referenceSampling,resolveLLMPolicy} from "../src/llm-policy"
import {buildContext,messageGroups} from "../src/context"
import {truncateToolOutput} from "../src/tool-output"
import reference from "./fixtures/reference-sampling.json"
import {withDomain} from "../src/domain/runtime"
import {reflectionExecutionSignature,windowReflectionPredictionOutputTokenLimit,windowReflectionLlmTimeoutMs} from "../src/domain/forecast-window-reflection"

function provider(respond:(body:any,index:number)=>Response){
  const requests:any[]=[]
  const server=Bun.serve({hostname:"127.0.0.1",port:0,async fetch(req){const body=await req.json();requests.push(body);return respond(body,requests.length)}})
  return {requests,url:`http://127.0.0.1:${server.port}/v1`,close:()=>server.stop(true)}
}
const success=(content="done")=>Response.json({choices:[{finish_reason:"stop",message:{role:"assistant",content}}],usage:{prompt_tokens:20,completion_tokens:10,total_tokens:30}})
async function storeFor(dir:string){const root=path.join(dir,"store");await mkdir(path.join(root,"runtime"),{recursive:true});return new Store(root)}

test("sampling defaults match the recorded reference model policy",()=>{
  for(const [model,expected] of Object.entries(reference)){
    const actual=referenceSampling(model)
    expect(JSON.parse(JSON.stringify({temperature:actual.temperature,topP:actual.top_p,topK:actual.top_k}))).toEqual(expected)
  }
})

test("declared model limits permit 64K window output and retain compatible 32K checkpoints",async()=>{
  const p=provider(()=>success())
  const f=await fixture({execution:{mode:"agent"},llm:{model:"large-model",base_url:p.url,context_window_tokens:1000000,max_output_tokens:32000}})
  const store=await storeFor(f.dir),client=new LLMClient(f.config,store),signal=new AbortController().signal
  try {
    await mkdir(path.join(f.dir,".forecast"),{recursive:true})
    await Bun.write(path.join(f.dir,".forecast/task.json"),JSON.stringify(f.config.task))
    const policyPath=path.join(f.dir,"policy.json"),predictions=path.join(f.dir,"predictions.csv")
    await Bun.write(policyPath,"{}");await Bun.write(predictions,"timestamp,predicted\n2025-01-01,1\n")
    const input:any={split:"test",runId:"test",candidateId:null,task:f.config.task,policyPath,
      baseline:{predictions,deterministicPredictions:null,candidate:null,baselineKind:"raw_anchor"},gapPredictions:null,
      llmEnabled:true,apiConcurrency:8,maxRounds:3,maxFormatRetries:2,llmIdentity:{providerID:"openai-compatible",modelID:"large-model"}}
    await withDomain({config:f.config,store,llm:client,signal},async()=>{
      expect(windowReflectionPredictionOutputTokenLimit()).toBe(32000)
      expect(windowReflectionLlmTimeoutMs()).toBe(300000)
      const original=await reflectionExecutionSignature(input)
      f.config.llm!.model_capabilities["large-model"]={max_output_tokens:65536}
      f.config.execution.llm_timeout_seconds=600
      expect(windowReflectionPredictionOutputTokenLimit()).toBe(65536)
      expect(windowReflectionLlmTimeoutMs()).toBe(600000)
      await client.complete([{role:"user",content:"test declared limit"}],[],"declared-cap",signal,undefined,{maxOutputTokens:65536})
      expect(p.requests[0].max_tokens).toBe(65536)
      expect(await reflectionExecutionSignature(input)).not.toBe(original)
      expect(await reflectionExecutionSignature({...input,predictionOutputTokenLimit:32000,llmTimeoutMs:300000})).toBe(original)
      f.config.llm!.model_capabilities["large-model"]={max_output_tokens:8192}
      expect(windowReflectionPredictionOutputTokenLimit()).toBe(8192)
    })
    expect(windowReflectionPredictionOutputTokenLimit()).toBe(32000)
    expect(windowReflectionLlmTimeoutMs()).toBe(300000)
  } finally {store.close();p.close();await f.cleanup()}
})

test("HTTP requests honor role parameters, model limits, token usage and tool-name repair",async()=>{
  const p=provider(()=>Response.json({choices:[{finish_reason:"tool_calls",message:{role:"assistant",content:null,reasoning_content:"Visible provider reasoning",tool_calls:[{id:"c1",type:"function",function:{name:"FORECAST_STATE",arguments:"{}"}}]}}],usage:{prompt_tokens:23,completion_tokens:17,total_tokens:40,prompt_tokens_details:{cached_tokens:17},completion_tokens_details:{reasoning_tokens:8}}}))
  const f=await fixture({execution:{mode:"agent"},llm:{model:"qwen3",base_url:p.url,temperature:.3,max_output_tokens:2000,model_capabilities:{"qwen3":{max_output_tokens:512,supports_temperature:false,supports_top_k:true}}},agents:{forecaster:{temperature:.8,top_p:.7,top_k:30,seed:7,reasoning_effort:"high"}}})
  const store=await storeFor(f.dir)
  try {
    const client=new LLMClient(f.config,store)
    const response=await client.complete([{role:"user",content:"read",audit:{usage:{total:999}}}],[{type:"function",function:{name:"forecast_state",parameters:{type:"object"}}}],"contract",new AbortController().signal,undefined,{agent:"forecaster",maxOutputTokens:10000})
    const sent=p.requests[0]
    expect(sent.max_tokens).toBe(512);expect(sent.temperature).toBeUndefined();expect(sent.top_p).toBe(.7);expect(sent.top_k).toBe(30);expect(sent.seed).toBe(7);expect(sent.reasoning_effort).toBe("high")
    expect(sent.messages[0].audit).toBeUndefined()
    expect(response.tool_calls![0].function.name).toBe("forecast_state")
    expect(response.reasoning_content).toBe("Visible provider reasoning")
    expect(response.audit!.usage).toEqual({input:6,output:17,cache_read:17,reasoning:8,total:40})
    expect((store.usage() as any).tokens).toBe(40)
  }finally{store.close();p.close();await f.cleanup()}
})

test("unknown models omit sampling defaults and incompatible tool capability fails before HTTP",async()=>{
  const p=provider(()=>success())
  const f=await fixture({execution:{mode:"agent"},llm:{model:"plain-model",base_url:p.url,model_capabilities:{"plain-model":{supports_tools:false,max_output_tokens:1024}}}})
  const store=await storeFor(f.dir)
  try {
    const client=new LLMClient(f.config,store)
    await client.complete([{role:"user",content:"hello"}],[],"plain",new AbortController().signal)
    expect(p.requests[0].temperature).toBeUndefined();expect(p.requests[0].top_p).toBeUndefined();expect(p.requests[0].top_k).toBeUndefined()
    await expect(client.complete([{role:"user",content:"hello"}],[{}],"tools",new AbortController().signal)).rejects.toThrow("does not support")
    expect(p.requests).toHaveLength(1)
  }finally{store.close();p.close();await f.cleanup()}
})

test("default provider retries are zero and truncated responses do not restart generation",async()=>{
  for(const truncated of [false,true]){
    const p=provider(()=>truncated?Response.json({choices:[{finish_reason:"length",message:{role:"assistant",content:"partial"}}]}):Response.json({error:{message:"unavailable"}},{status:503}))
    const f=await fixture({execution:{mode:"agent"},llm:{model:"plain",base_url:p.url,max_retries:truncated?2:0}})
    const store=await storeFor(f.dir)
    try {
      expect(f.config.llm!.max_retries).toBe(truncated?2:0)
      await expect(new LLMClient(f.config,store).complete([{role:"user",content:"hello"}],[],"failure",new AbortController().signal)).rejects.toThrow(truncated?"output limit":"HTTP 503")
      expect(p.requests).toHaveLength(1)
    }finally{store.close();p.close();await f.cleanup()}
  }
})

test("provider context overflow compacts once, replays the latest turn and records all calls",async()=>{
  const p=provider((body,index)=>{
    if(index===1)return Response.json({error:{code:"context_length_exceeded"}},{status:400})
    if(body.messages[0].content.includes("summarizing conversations"))return success("Frozen target and training scale; historical work completed. Continue the latest request.")
    expect(body.messages.some((m:any)=>m.content==="latest pending request")).toBe(true)
    expect(JSON.stringify(body)).toContain("Frozen target and training scale")
    return success("finished")
  })
  const f=await fixture({execution:{mode:"agent"},llm:{model:"plain",base_url:p.url,max_output_tokens:1024}})
  const store=await storeFor(f.dir)
  try {
    store.append("s",{role:"user",content:"earlier request"});store.append("s",{role:"assistant",content:"earlier public result"});store.append("s",{role:"user",content:"latest pending request"})
    const runtime=new AgentRuntime(f.config,store,new LLMClient(f.config,store),()=>"Frozen task",new AbortController().signal)
    expect(await runtime.run("general","s","latest pending request")).toBe("finished")
    expect(p.requests).toHaveLength(3);expect((store.usage() as any).calls).toBe(3)
    const checkpoint=store.get<any>("context-checkpoint:s");expect(checkpoint.through).toBe(2)
    expect(store.messages("s")[0].content).toBe("earlier request")
    const policy=resolveLLMPolicy(f.config,"general")
    const recovered=await buildContext(store,"s","Frozen task",[],policy.context,policy.output,async()=>{throw new Error("must reuse summary")})
    expect(()=>messageGroups(recovered)).not.toThrow()
  }finally{store.close();p.close();await f.cleanup()}
})

test("reference pruning protects two recent turns and skill outputs without deleting audit history",async()=>{
  const f=await fixture(),store=await storeFor(f.dir)
  try {
    for(let i=0;i<8;i++){
      store.append("s",{role:"user",content:"turn "+i});store.append("s",{role:"assistant",content:null,tool_calls:[{id:"c"+i,type:"function",function:{name:i===0?"skill":"read",arguments:"{}"}}]});store.append("s",{role:"tool",tool_call_id:"c"+i,content:("output-"+i+" ").repeat(12000)})
    }
    const before=JSON.stringify(store.messages("s"))
    const result=await buildContext(store,"s","Task",[],1000000,1000)
    expect(result.some(m=>m.content==="[Old tool result content cleared]")).toBe(true)
    for(const i of [0,6,7])expect(result.find(m=>m.tool_call_id==="c"+i)!.content).toStartWith("output-"+i)
    expect(JSON.stringify(store.messages("s"))).toBe(before)
    const second=await buildContext(store,"s","Task",[],1000000,1000)
    expect(second).toEqual(result)
    expect(()=>messageGroups(result)).not.toThrow()
  }finally{store.close();await f.cleanup()}
})

test("reported token counts trigger compaction even when the current estimate is small",async()=>{
  const f=await fixture(),store=await storeFor(f.dir)
  try {
    store.append("s",{role:"user",content:"first"});store.append("s",{role:"assistant",content:"past",audit:{usage:{total:70000}}});store.append("s",{role:"user",content:"pending"})
    let calls=0
    const result=await buildContext(store,"s","Task",[],64000,4096,async()=>{calls++;return "First task completed; continue pending."})
    expect(calls).toBe(1);expect(result.some(m=>m.content==="pending")).toBe(true)
    await buildContext(store,"s","Task",[],64000,4096,async()=>{throw new Error("Do not compact the same reported usage twice")})
  }finally{store.close();await f.cleanup()}
})

test("tool output follows the reference byte and line limits",()=>{
  expect(truncateToolOutput("x".repeat(20000)).truncated).toBe(false)
  expect(truncateToolOutput("x".repeat(51201)).truncated).toBe(true)
  const text=Array.from({length:2001},()=>"line").join("\n")
  expect(truncateToolOutput(text).preview.split("\n")).toHaveLength(2000)
})

test("fatal window-call budgets abort the owning domain scope",async()=>{
  const {withDomain,Session,SessionPrompt}=await import("../src/domain/runtime")
  const {AppError}=await import("../src/contracts")
  const f=await fixture(),store=await storeFor(f.dir),cancellation=new AbortController()
  const client={async complete(){throw new AppError("BUDGET_EXHAUSTED","Global budget exhausted",5)}} as unknown as LLMClient
  try {
    await withDomain({config:f.config,store,llm:client,signal:cancellation.signal,onFatal:error=>cancellation.abort(error)},async()=>{
      const child=await Session.create({parentID:"p",title:"isolated",permission:{}})
      await expect(SessionPrompt.prompt({sessionID:child.id,messageID:"m",model:{providerID:"openai-compatible",modelID:"mock"},agent:"window-reflection",system:"Blind window",tools:{},parts:[{type:"text",text:"{}"}]})).rejects.toThrow("Global budget exhausted")
    })
    expect(cancellation.signal.aborted).toBe(true)
    expect(cancellation.signal.reason.exitCode).toBe(5)
  }finally{store.close();await f.cleanup()}
})

test("disabled automatic compaction preserves a fitting projection despite old reported usage",async()=>{
  const f=await fixture(),store=await storeFor(f.dir)
  try {
    store.append("s",{role:"user",content:"task"});store.append("s",{role:"assistant",content:"past",audit:{usage:{total:100000}}});store.append("s",{role:"user",content:"continue"})
    const messages=await buildContext(store,"s","Task",[],64000,4096,async()=>{throw new Error("Auto compaction is disabled")},{auto:false})
    expect(messages.at(-1)!.content).toBe("continue")
    expect(store.get("context-checkpoint:s")).toBeUndefined()
  }finally{store.close();await f.cleanup()}
})
