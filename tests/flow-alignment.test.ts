import {test,expect} from "bun:test"
import path from "node:path"
import {mkdir,readFile} from "node:fs/promises"
import {fixture} from "./helpers"
import {Store} from "../src/storage"
import {ForecastService} from "../src/forecast"
import {AgentRuntime} from "../src/runtime"
import {LLMClient} from "../src/llm"
import {CriticController} from "../src/critic"
import {registerTools} from "../src/tools"
import {compute} from "../src/python"
import {FORECAST_MODEL_NAMES} from "../src/domain/model-family"

async function setup(f:Awaited<ReturnType<typeof fixture>>){const root=path.join(f.dir,"store");await mkdir(path.join(root,"runtime"),{recursive:true});const store=new Store(root);store.set("state",{phase:"forecast"});return store}

test("reference spec interface checks frozen task fields and records full experiment reflection",async()=>{
  const f=await fixture({task:{model_families:["Naive"]}}),store=await setup(f)
  const signal=new AbortController().signal,client=new LLMClient(f.config,store),service=new ForecastService(f.config,store,signal)
  const runtime=new AgentRuntime(f.config,store,client,()=>"",signal)
  registerTools(runtime,service,new CriticController(f.config,store,client,signal),f.config)
  const ctx={sessionId:"forecaster",agent:"forecaster",depth:0,callId:"call",signal}
  const call=async(name:string,args:any)=>{const tool=runtime.tools.get(name)!;return tool.execute(tool.schema.parse(args),ctx) as Promise<any>}
  try {
    await expect(call("generate_model",{spec:{model:"Naive",seq_len:99}})).rejects.toThrow("frozen task field seq_len")
    await expect(call("generate_model",{model:"Naive",params:{pred_len:99}})).rejects.toThrow("frozen task field pred_len")
    const run=await call("generate_model",{spec:{model:"Naive",seq_len:24,pred_len:12,hyperparams:{}}})
    const evaluated=await call("evaluate_experiment",{run_id:run.run_id})
    expect(evaluated.comparison.improved).toBe(true)
    expect(evaluated.metrics).toEqual(run.metrics)
    await call("forecast_reflect",{run_id:run.run_id,decision:"keep",effective:true,reason:"This is an actual evaluated baseline and its fit is reusable.",next_direction:"Compare another permitted family."})
    const reflection=store.get<any[]>("reflection_records")![0]
    expect(reflection.next_direction).toBe("Compare another permitted family.")
    const history=(await readFile(path.join(store.root,"history.jsonl"),"utf8")).trim().split("\n").map(x=>JSON.parse(x))
    expect(history).toHaveLength(2);expect(history[1].type).toBe("reflection")
    const originalTimestamp=history[0].timestamp
    await service.projectHistory()
    expect(JSON.parse((await readFile(path.join(store.root,"history.jsonl"),"utf8")).split("\n")[0]).timestamp).toBe(originalTimestamp)
  }finally{store.close();await f.cleanup()}
})

test("failed attempts can be reflected and retried without spending another unique-spec slot",async()=>{
  const f=await fixture({task:{model_families:["Holt"]},execution:{max_experiments:1,max_consecutive_crashes:3}}),store=await setup(f)
  const service=new ForecastService(f.config,store,new AbortController().signal)
  try {
    await expect(service.experiment("Holt",{alpha:2})).rejects.toThrow("fixed alpha/beta")
    const first=store.get<any[]>("attempts")![0]
    await expect(service.experiment("Holt",{alpha:2})).rejects.toThrow("Reflect on the previous")
    await service.reflect(first.run_id,"crash","The backend rejected unsupported parameters; record the failed attempt.",{effective:false,next_direction:"Resolve the parameter contract before continuing."})
    await expect(service.experiment("Holt",{alpha:2})).rejects.toThrow("fixed alpha/beta")
    const attempts=store.get<any[]>("attempts")!
    expect(attempts).toHaveLength(2)
    expect(new Set(attempts.map(a=>a.run_id)).size).toBe(1)
    expect(attempts[1].attempt_index).toBe(2)
    expect(store.get<number>("consecutive_crashes")).toBe(2)
    await service.reflect(first.run_id,"crash","The same declared failure was retried and recorded separately.")
    expect(store.get<any[]>("reflection_records")).toHaveLength(2)
    expect(store.get<any>("domain-state").stop_counter_last_attempt_index).toBe(2)
    expect(service.state().budgets.experiments_remaining).toBe(0)
  }finally{store.close();await f.cleanup()}
})

test("compiled runtime catalog and Python implementations declare the same model names",async()=>{
  const f=await fixture()
  try {
    const result=await compute(f.config,{operation:"models"},undefined,new AbortController().signal)
    expect([...result.models.map((m:any)=>m.name),"UserPrior"].sort()).toEqual([...FORECAST_MODEL_NAMES].sort())
    expect(result.availability_scope).toContain("dependency presence")
  }finally{await f.cleanup()}
})
