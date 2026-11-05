import {test,expect} from "bun:test"
import path from "node:path"
import os from "node:os"
import {mkdtemp,readFile,writeFile,rm} from "node:fs/promises"
import {fixture,mockProvider} from "./helpers"
import {AppError} from "../src/contracts"
import {runTask} from "../src/workflow"
import {Store} from "../src/storage"
import {ForecastService} from "../src/forecast"
import {CriticController} from "../src/critic"
import {LLMClient} from "../src/llm"
import {digest} from "../src/util"
import {isReusableEvidenceLoopCorrection,shouldReopenAbnormalEvidenceLoopCheckpoint} from "../src/domain/forecast-window-reflection"
import {ForecastWindowReflectionTool} from "../src/domain/forecast-window-reflection"

test("transport failures reopen even after candidate exhaustion, while genuine semantic exhaustion remains terminal",()=>{
  const checkpoint={version:1,window_index:0,state_sha256:"a".repeat(64),phase:"terminated",selected_candidate_id:null,
    attempted_candidate_ids:["context:current-window"],seeking_trace:[{exhausted:true}],
    rejected_transitions:[{candidate_id:"context:current-window",reason:"provider_failed"}],accepted_transitions:0,
    termination_reason:"provider_failed",predictability:"unpredictable",updated_at:new Date().toISOString()}
  expect(shouldReopenAbnormalEvidenceLoopCheckpoint(checkpoint)).toBe(true)
  expect(shouldReopenAbnormalEvidenceLoopCheckpoint({...checkpoint,termination_reason:"no_valid_transition"})).toBe(true)
  expect(shouldReopenAbnormalEvidenceLoopCheckpoint({...checkpoint,termination_reason:"no_valid_transition",attempted_candidate_ids:[]})).toBe(false)
  expect(shouldReopenAbnormalEvidenceLoopCheckpoint({...checkpoint,termination_reason:"evidence_exhausted",rejected_transitions:[]})).toBe(false)
})

test("accepted flags cannot make a provider-failed terminal correction reusable",async()=>{
  const dir=await mkdtemp(path.join(os.tmpdir(),"adacast-recovery-"))
  const points=[{horizon_index:0,timestamp:"2024-12-01 00:00:00",value:1}]
  const expected={windowIndex:0,horizonIndices:[0],timestamps:[points[0].timestamp]}
  const state:any={version:1,window_index:0,evidences:[],predictions:points,evidence_gap:1,needed_evidence:"Inspect context"}
  async function correction(reason:string,rejections:any[]=[]) {
    const bytes=JSON.stringify(state)+"\n",file=path.join(dir,"state.json")
    await writeFile(file,bytes)
    return {window_index:0,accepted:true,format_retry:false,format_errors:[],detector_report:{},points,rationale:"Committed state",evidence_ids:[],
      assessment_stage:"evidence_loop_terminal",critic_prediction_state:state,critic_prediction_state_artifact:{path:file,sha256:digest(bytes)},
      evidence_loop:{termination_reason:reason,predictability:"unpredictable",accepted_transitions:state.evidences.length,
        max_evidence_iterations:3,seeking_trace:[{exhausted:true}],rejected_transitions:rejections}}
  }
  try {
    expect(await isReusableEvidenceLoopCorrection(await correction("evidence_exhausted"),expected)).toBe(true)
    expect(await isReusableEvidenceLoopCorrection(await correction("provider_failed"),expected)).toBe(false)
    const failure={candidate_id:"context:current-window",reason:"provider_failed"}
    expect(await isReusableEvidenceLoopCorrection(await correction("no_valid_transition",[failure]),expected)).toBe(false)
    // A failed request followed by a successful commit is ordinary audit history.
    state.evidences=[{evidence_candidate_id:"context:current-window",candidate_type:"context",artifact_path:path.join(dir,"context.json"),artifact_sha256:"b".repeat(64)}]
    expect(await isReusableEvidenceLoopCorrection(await correction("no_valid_transition",[failure]),expected)).toBe(true)
  } finally {await rm(dir,{recursive:true,force:true})}
})

test("provider outage leaves test windows incomplete and recovery really invokes the healthy provider",async()=>{
  const provider=mockProvider({evidence:true})
  let outage=false,failedGenerations=0,healthyGenerations=0
  const proxy=Bun.serve({hostname:"127.0.0.1",port:0,async fetch(request){
    const body=await request.json() as any
    const generation=!body.messages[0].content.includes("FROZEN TASK AND CURRENT STATE:") &&
      body.messages[1]?.content.includes('"provisional_evidence_fusion"')
    if(generation && outage){failedGenerations++;return Response.json({error:{message:"injected provider outage"}},{status:503})}
    if(generation)healthyGenerations++
    return fetch(provider.baseURL+"/chat/completions",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})
  }})
  process.env.ADACAST_LITE_RECOVERY_TEST_KEY="test-credential"
  const f=await fixture({task:{reflection_enabled:true,model_families:["Naive"]},execution:{mode:"agent",window_llm_concurrency:2},
    llm:{base_url:`http://127.0.0.1:${proxy.port}/v1`,model:"mock-model",api_key_env:"ADACAST_LITE_RECOVERY_TEST_KEY",max_retries:0}})
  let store:Store|undefined
  try {
    await expect(runTask({configFile:f.configFile,workspace:f.workspace,signal:new AbortController().signal,
      emit(type,value:any){if(type==="phase.changed"&&value.phase==="post-forecast")throw new AppError("INTERRUPTED","prepared test checkpoint",130)}})).rejects.toThrow("prepared test checkpoint")
    store=new Store(path.join(f.workspace,".forecast"))
    const config=store.get<any>("config"),signal=new AbortController().signal
    const service=new ForecastService(config,store,signal)
    await service.testBest()
    const critic=new CriticController(config,store,new LLMClient(config,store),signal)
    outage=true
    await expect(critic.run("test")).rejects.toThrow("LLM window reasoning did not complete")
    expect(failedGenerations).toBeGreaterThan(0)
    expect(store.get("test_reflection")).toBeUndefined()
    const testRun=store.get<any>("test")
    const directory=path.join(store.root,"reflection/test",testRun.run_id,"raw")
    const failed=JSON.parse(await readFile(path.join(directory,"window-progress.json"),"utf8"))
    expect(failed.request_failed_windows).toBe(6)
    expect(failed.accepted_windows).toBe(0)
    const before=healthyGenerations
    outage=false
    await service.domain(ForecastWindowReflectionTool,{subcommand:"resume_test",run_id:testRun.run_id,validation_run_id:testRun.source_run_id})
    expect(healthyGenerations-before).toBeGreaterThanOrEqual(6)
    expect(store.get<any>("test_reflection").completed_windows).toBe(6)
    expect(store.get<any>("final_output").kind).toBe("window_refined")
    const completed=store.get<any>("final_output"),callsAfterRecovery=healthyGenerations
    // Simulate interruption between domain completion and the host's projection.
    store.db.query("DELETE FROM kv WHERE key IN ('test_reflection','final_output')").run()
    await critic.run("test")
    expect(healthyGenerations).toBe(callsAfterRecovery)
    expect(store.get<any>("final_output")).toEqual(completed)
    expect(store.get<any>("test_reflection").completed_windows).toBe(6)
    for(let i=0;i<6;i++) {
      const terminal=JSON.parse(await readFile(path.join(directory,"corrections",`window-${i.toString().padStart(6,"0")}.json`),"utf8"))
      expect(terminal.evidence_loop.termination_reason).toBe("evidence_sufficient")
      expect(terminal.evidence_loop.accepted_transitions).toBeGreaterThan(0)
    }
  } finally {store?.close();proxy.stop(true);provider.close();await f.cleanup();delete process.env.ADACAST_LITE_RECOVERY_TEST_KEY}
},360000)
