import { test, expect } from "bun:test"
import path from "node:path"
import { readFile } from "node:fs/promises"
import { fixture } from "./helpers"
import { runTask } from "../src/workflow"
import { Store } from "../src/storage"
import { ForecastService } from "../src/forecast"
import { AppError } from "../src/contracts"
import { ForecastAdjustmentPolicyTool as Policy } from "../src/domain/forecast-adjustment-policy"
import { ForecastAdjustmentSourceModelsTool as Sources } from "../src/domain/forecast-adjustment-source-models"
import { ForecastAdjustmentFitRunsTool as Fits } from "../src/domain/forecast-adjustment-fit-runs"
import { ForecastAdjustmentDiagnosticsTool as Diagnostics } from "../src/domain/forecast-adjustment-diagnostics"
import { ForecastAdjustmentCandidateTool as Candidate } from "../src/domain/forecast-adjustment-candidate"
import { ForecastAdjustmentApplyTool as Apply } from "../src/domain/forecast-adjustment-apply"
import { ForecastAdjustmentEvaluateTool as Evaluate } from "../src/domain/forecast-adjustment-evaluate"
import { ForecastAdjustmentSkillTool as Skill } from "../src/domain/forecast-adjustment-skill"

// All fitting, file hashes, split checks, acceptance and reuse run against real artifacts.
test("canonical adjustment freezes family sources, replays validation front, validates and locks a conditional skill", async()=>{
  const f=await fixture({task:{model_families:["Holt","Naive"]}})
  let store:Store|undefined
  try {
    await expect(runTask({configFile:f.configFile,workspace:f.workspace,signal:new AbortController().signal,emit(type,value:any){if(type==="phase.changed"&&value.phase==="forecast")throw new AppError("INTERRUPTED","test checkpoint",130)}})).rejects.toThrow("test checkpoint")
    store=new Store(path.join(f.workspace,".forecast"))
    const config=store.get<any>("config")
    config.decisions.adjustment.strategy="a_lightweight"
    const service=new ForecastService(config,store,new AbortController().signal)
    const stat=await service.experiment("Holt")
    await service.reflect(stat.run_id,"keep","Retain this real statistical baseline for family comparison")
    const deep=await service.experiment("Naive")
    await service.reflect(deep.run_id,"keep","Retain this independent baseline for the other family")
    await service.domain(Policy,{subcommand:"configure",policy:{enabled:true,mode:"rule_skill",strategy:"a_lightweight"}})
    await service.domain(Policy,{subcommand:"lock"})
    const sources=(await service.domain(Sources,{subcommand:"freeze"})).source_model_set
    expect(sources.members).toHaveLength(2)
    await service.domain(Fits,{subcommand:"prepare"})
    const fitted=await service.domain(Fits,{subcommand:"run"})
    expect(fitted.status).toBe("complete")
    const member=fitted.members.find((m:any)=>m.validation_run_id===stat.run_id)
    const diagnostics=await service.domain(Diagnostics,{run_id:member.fit_run_id})
    expect(diagnostics.split_scope).toBe("fit_only_validation_front")
    const evidence=`.forecast/adjustments/diagnostics/fit/${member.fit_run_id}/diagnostics.json`
    const candidate=await service.domain(Candidate,{subcommand:"write",candidate_id:"hourly-context",source_model_set_id:sources.source_model_set_id,rules:[{id:"hour-zero",condition:{kind:"timestamp_hour_in",values:[0]},action:{type:"add",value:0},rationale:"A zero correction preserves the evidence-backed initial baseline",evidence_refs:[evidence],bounds:{max_delta_ratio:.1}}],skill_markdown:"# Scoped hourly context\n\nApply a zero correction only at midnight; retain the immutable initial forecast."})
    expect(candidate.candidate_id).toBe("hourly-context")
    await service.domain(Apply,{run_id:stat.run_id,candidate_id:"hourly-context",split:"val"})
    const evaluated=await service.domain(Evaluate,{run_id:stat.run_id,candidate_id:"hourly-context",split:"val"})
    expect(evaluated.acceptance_decision.accepted).toBe(true)
    await expect(service.domain(Policy,{subcommand:"mark_no_effective"})).rejects.toThrow("cannot discard")
    await service.domain(Skill,{subcommand:"accept_candidate",candidate_id:"hourly-context",run_id:stat.run_id})
    await service.domain(Skill,{subcommand:"lock_active"})
    await expect(service.domain(Candidate,{subcommand:"write",source_model_set_id:sources.source_model_set_id,rules:[],skill_markdown:"frozen"})).rejects.toThrow("locked")
    await service.lockSelection()
    store.set("state",{...store.get<any>("state"),phase:"post-forecast"})
    await service.testBest()
    const tests=store.get<any[]>("test_collection")!
    expect(tests).toHaveLength(2)
    expect(store.get<any>("test").metrics.mse).toBe(Math.min(...tests.map(r=>r.metrics.mse)))
    const adjusted=await service.applyActiveAdjustment(tests.find(r=>r.source_run_id===stat.run_id)!)
    const adjustedEval=JSON.parse(await readFile(path.join(path.dirname(adjusted.predictions),"adjusted_eval.json"),"utf8"))
    expect(adjusted.metrics).toEqual(adjustedEval.adjusted_metrics)
    const decisions=store.get<any>("domain-state").adjustment.test_application_records
    expect(decisions.find((r:any)=>r.validation_run_id===stat.run_id).decision).toBe("apply")
    expect(decisions.find((r:any)=>r.validation_run_id===deep.run_id).decision).toBe("skip")
    const reuse=JSON.parse(await readFile(path.join(service.root,"adjustments/fit-runs/runs",member.fit_run_id,"model/reuse.json"),"utf8"))
    expect(reuse.training_performed).toBe(false)
  }finally{store?.close();await f.cleanup()}
},180000)
