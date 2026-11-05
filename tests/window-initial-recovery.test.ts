import {test,expect} from "bun:test"
import path from "node:path"
import {fixture,mockProvider} from "./helpers"
import {AppError} from "../src/contracts"
import {runTask} from "../src/workflow"
import {Store} from "../src/storage"
import {ForecastService} from "../src/forecast"
import {CriticController} from "../src/critic"
import {LLMClient} from "../src/llm"

test("recovery performs the failed initial assessment before any prediction generation",async()=>{
  let outage=false
  const order:Array<{window:number;kind:string}>=[]
  const provider=mockProvider({evidence:true})
  const proxy=Bun.serve({hostname:"127.0.0.1",port:0,async fetch(request){
    const body=await request.json() as any
    const system=String(body.messages[0].content)
    if(!system.includes("FROZEN TASK AND CURRENT STATE:")) {
      const raw=String(body.messages[1].content)
      const text=raw.includes("Blind window context:")?raw.split("Blind window context:").at(-1)!.trim():raw
      const input=JSON.parse(text.slice(text.indexOf("{")))
      const assessment=!!input.unused_evidence_candidate_summaries
      if(assessment && outage)return Response.json({error:{message:"initial assessment outage"}},{status:503})
      const state=input.state??input.critic_prediction_state
      if(state?.window_index!==undefined)order.push({window:state.window_index,kind:assessment?(state.evidences.length?"reassessment":"initial"):"generation"})
    }
    return fetch(provider.baseURL+"/chat/completions",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})
  }})
  process.env.ADACAST_INITIAL_RECOVERY_KEY="local-test"
  const f=await fixture({task:{reflection_enabled:true,model_families:["Naive"]},execution:{mode:"agent",window_llm_concurrency:2},
    llm:{base_url:`http://127.0.0.1:${proxy.port}/v1`,model:"mock-model",api_key_env:"ADACAST_INITIAL_RECOVERY_KEY",max_retries:0}})
  let store:Store|undefined
  try {
    await expect(runTask({configFile:f.configFile,workspace:f.workspace,signal:new AbortController().signal,
      emit(type,value:any){if(type==="phase.changed"&&value.phase==="post-forecast")throw new AppError("INTERRUPTED","prepared checkpoint",130)}})).rejects.toThrow("prepared checkpoint")
    store=new Store(path.join(f.workspace,".forecast"))
    const config=store.get<any>("config"),signal=new AbortController().signal
    const service=new ForecastService(config,store,signal)
    await service.testBest()
    const critic=new CriticController(config,store,new LLMClient(config,store),signal)
    outage=true
    await expect(critic.run("test")).rejects.toThrow("LLM window reasoning did not complete")
    expect(store.get("test_reflection")).toBeUndefined()
    outage=false;order.length=0
    await critic.run("test")
    expect(store.get<any>("test_reflection").completed_windows).toBe(6)
    for(let window=0;window<6;window++) {
      const calls=order.filter(call=>call.window===window)
      expect(calls[0]?.kind).toBe("initial")
      expect(calls.some(call=>call.kind==="generation")).toBe(true)
    }
  } finally {store?.close();proxy.stop(true);provider.close();await f.cleanup();delete process.env.ADACAST_INITIAL_RECOVERY_KEY}
},360000)
