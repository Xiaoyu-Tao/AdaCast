import path from "node:path"
import { AppError, type Config, type Experiment } from "./contracts"
import type { Store } from "./storage"
import type { LLMClient } from "./llm"
import { ForecastService } from "./forecast"
import { ForecastWindowReflectionTool } from "./domain/forecast-window-reflection"
import { readJSON } from "./util"

/** The original window controller, hosted by the independent headless runtime. */
export class CriticController {
  constructor(public config:Config,public store:Store,public llm:LLMClient,public signal:AbortSignal){}
  async run(split:"val"|"test") {
    if(!this.config.task.reflection_enabled)return {status:"disabled"}
    const service=new ForecastService(this.config,this.store,this.signal)
    const representatives=this.store.get<Experiment[]>("family_representatives")
    if(!representatives?.length)throw new AppError("PHASE_GATE","Freeze validation family representatives before window reflection",3)
    const sources=split==="val"?representatives:[this.store.get<Experiment>("test")!]
    if(sources.some(x=>!x))throw new AppError("PHASE_GATE","Complete raw test inference first",3)
    const results=[]
    const active=split==="val"?(await service.domainState()).adjustment.active_skill_version:undefined
    const manifest=active?await readJSON(path.join(this.store.root,"adjustments/skills/active/manifest.json")):undefined
    const accepted=new Set<string>()
    if(manifest)for(const file of manifest.validation_results){const result=await readJSON(path.resolve(path.dirname(this.store.root),file));if(result.acceptance_decision?.accepted)accepted.add(result.run_id)}
    for(const source of sources){
      const candidateId=split==="test"?this.store.get<string>("test_adjustment_candidate_id"):undefined
      const ctx={sessionId:split==="val"?"forecaster:forecast":"critic:post-forecast",callId:"window-"+source.run_id,agent:split==="val"?"forecaster":"critic"}
      const result=await service.domain(ForecastWindowReflectionTool,{subcommand:split==="val"?"run_validation":"run_test",run_id:source.run_id,...(split==="test"?{validation_run_id:source.source_run_id}:{}),...(candidateId?{adjustment_candidate_id:candidateId}:{})},ctx)
      results.push(result)
      if(split==="val" && manifest && accepted.has(source.run_id))results.push(await service.domain(ForecastWindowReflectionTool,{subcommand:"run_validation",run_id:source.run_id,adjustment_candidate_id:manifest.candidate_id},ctx))
      if(split==="test") {
        const artifact=await readJSON(path.resolve(path.dirname(this.store.root),result.result_path))
        this.store.set("final_output",{predictions:path.resolve(path.dirname(this.store.root),artifact.final_predictions_path),metrics:artifact.final_metrics,kind:"window_refined"})
      }
    }
    const summary={status:"completed",records:results,completed_windows:(this.store.get<any>("data_info")?.window_counts[split]??0)*results.length}
    this.store.set(split==="val"?"validation_reflection":"test_reflection",summary)
    return summary
  }
}
