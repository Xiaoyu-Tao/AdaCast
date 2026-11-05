import { readBoundModelArtifactIdentity } from "./domain/model-artifact"
import { EvalResult } from "./domain/types"
import { ForecastAdjustmentApplyTool } from "./domain/forecast-adjustment-apply"
import { ForecastAdjustmentEvaluateTool } from "./domain/forecast-adjustment-evaluate"
import { validationSpecMatchesPostForecastTestSpec } from "./domain/execution-contract"
import { LLMClient } from "./llm"
import { withDomain, type Tool } from "./domain/runtime"
import { readPhaseState, writePhaseState } from "./domain/forecast-adjustment-common"
import { ForecastWindowReflectionTool } from "./domain/forecast-window-reflection"
import { canonicalForecastModelName, inferForecastModelFamily } from "./domain/model-family"
import path from "node:path"
import { readFile, mkdir, readdir, cp } from "node:fs/promises"
import { z } from "zod"
import { AppError, type Config, type Experiment, type RunState } from "./contracts"
import type { Store } from "./storage"
import { compute } from "./python"
import { atomicWrite, canonical, checkAbort, digest, errorText, exists, now, readJSON, writeJSON } from "./util"

export class ForecastService {
  constructor(public config: Config, public store: Store, public signal: AbortSignal) {}
  async domain(tool:Tool.Info<any>,args:any,ctx?:{sessionId:string;callId:string;agent:string}) {
    const cancellation=new AbortController()
    const signal=AbortSignal.any([this.signal,cancellation.signal])
    return withDomain({config:this.config,store:this.store,llm:new LLMClient(this.config,this.store),signal,agent:ctx?.agent,onFatal:error=>cancellation.abort(error)},async()=>{
      if(tool.id.startsWith("forecast_adjustment_") && this.config.decisions.adjustment.strategy==="off") {
        const readOnly=["read","list","list_versions","read_active"].includes(args.subcommand)
        const disabledPolicy=tool.id==="forecast_adjustment_policy"&&(args.subcommand==="lock"||(args.subcommand==="configure"&&args.policy?.enabled===false))
        if(!readOnly&&!disabledPolicy)throw new AppError("ADJUSTMENT_DISABLED","Adjustment is disabled by the frozen configuration",3)
      }
      if(tool.id==="forecast_adjustment_policy"&&args.subcommand==="configure"&&this.config.decisions.adjustment.strategy!=="agent") {
        const strategy=this.config.decisions.adjustment.strategy
        if((args.policy?.enabled!==undefined&&args.policy.enabled!==(strategy!=="off"))||(args.policy?.strategy&&args.policy.strategy!==strategy))throw new AppError("CONFIG_CONFLICT","Adjustment policy conflicts with the frozen configuration",3)
      }
      const definition=await tool.init()
      let result
      try{result=await definition.execute(definition.parameters.parse(args),{sessionID:ctx?.sessionId??"domain",messageID:ctx?.callId??"domain",agent:ctx?.agent??"critic",abort:signal,messages:[],metadata:()=>{},ask:async()=>{throw new AppError("INTERACTION_UNAVAILABLE","All required decisions must come from the frozen task configuration",3)}})
      }catch(error){checkAbort(signal);throw error}
      const output=JSON.parse(result.output)
      if(tool.id==="forecast_window_reflection" && ["run_test","resume_test"].includes(args.subcommand) && output.status==="completed" &&
        output.source_run_id===this.store.get<Experiment>("test")?.run_id &&
        output.adjustment_candidate_id===(this.store.get<string>("test_adjustment_candidate_id")??null)) {
        // Canonical domain tools and the controller wrapper share the same report gates.
        this.store.transaction(()=>{
          this.store.set("final_output",{predictions:path.resolve(path.dirname(this.root),output.final_predictions_path),metrics:output.final_metrics,kind:"window_refined"})
          this.store.set("test_reflection",{status:"completed",records:[output],completed_windows:this.store.get<any>("data_info")?.window_counts.test??0})
        })
      }
      return output
    })
  }
  async domainState(update?:(state:any)=>any) {
    return withDomain({config:this.config,store:this.store,llm:new LLMClient(this.config,this.store),signal:this.signal},async()=>{
      const state=await readPhaseState();if(update){const next=update(state);await writePhaseState(next);return next}return state
    })
  }
  get root() { return this.store.root }
  get runs() { return this.store.get<Experiment[]>("runs") ?? [] }
  get validationRuns() { return this.runs.filter(r => r.split === "val" && r.status === "success") }
  get best() { return [...this.validationRuns].sort((a,b) => a.metrics.mse! - b.metrics.mse!)[0] }
  state() {
    return {state:this.store.get("state"),task:this.config.task,analysis:this.store.get("analysis"),
      domain:this.store.get("domain-state"),
      family_representatives:this.store.get("family_representatives"),
      experiments:this.validationRuns.map(r=>({run_id:r.run_id,model:r.model,spec:r.spec,metrics:r.metrics})),
      best:this.best ? {run_id:this.best.run_id,model:this.best.model,metrics:this.best.metrics} : null,
      attempts:this.store.get("attempts") ?? [],reflections:this.store.get("reflections") ?? {},
      skills:this.store.get("skills") ?? [],adjustment:this.store.get("adjustment"),
      adjustment_decision:this.store.get("adjustment_decision"),diagnostics:this.store.get("diagnostics"),candidates:this.store.get("candidates")??[],
      search_done:this.store.get("search_done"),final_report:this.store.get("final_report"),test_reflection:this.store.get("test_reflection"),
      validation_reflection:this.store.get("validation_reflection"),test:this.store.get("test"),
      final_output:this.store.get("final_output"),budgets:{llm_usage:this.store.usage(),limits:this.config.execution,
        experiments_remaining:Math.max(0,this.config.execution.max_experiments-new Set((this.store.get<any[]>("attempts")??[]).map(a=>a.run_id)).size),
        consecutive_crashes:this.store.get("consecutive_crashes")??0,no_improvement:this.store.get("no_improvement")??0}}
  }
  promptState() {
    const state=this.state()
    const compactRun=(r:any)=>({run_id:r.run_id,source_run_id:r.source_run_id,model:r.model,metrics:r.metrics,split:r.split,params:r.spec?.params})
    const analysis=state.analysis?{analysis_scope:state.analysis.analysis_scope,analysis:state.analysis.analysis,feature_columns:state.analysis.feature_columns,train_statistics:state.analysis.train_statistics,split_time_ranges:state.analysis.split_time_ranges}:undefined
    return {...state,analysis,experiments:this.validationRuns.map(compactRun),attempts:state.attempts.slice(-5),
      family_representatives:(this.store.get<any[]>("family_representatives")??[]).map(compactRun),test:state.test?compactRun(state.test):undefined,
      reflections:Object.fromEntries(Object.entries(state.reflections).slice(-5)),history_note:"Complete specs, attempts and reflections remain available through forecast_state and the immutable audit."}
  }
  async analyze() {
    if (this.store.get("analysis")) return this.store.get("analysis")
    const analysis = await compute(this.config,{operation:"analyze"},this.root,this.signal)
    await writeJSON(path.join(this.root,"reports/quantitative.json"),analysis)
    this.store.set("analysis",analysis); this.store.event("analysis.completed")
    return analysis
  }
  async preReport(content: string) {
    if(this.store.get<RunState>("state")!.phase!=="pre-forecast") throw new AppError("PHASE_GATE","Pre-forecast report is frozen after its phase",3)
    if (!this.store.get("analysis")) throw new AppError("PHASE_GATE","Run analyze_dataset first",3)
    if (content.length < 100) throw new AppError("REPORT_INVALID","Pre-forecast report must contain actual analysis",3)
    const headings=["Task Summary","Data Characteristics","Domain Context","Risk Flags","Modeling Directions"]
    const missing=headings.filter(h=>!content.split("\n").some(line=>line.trim()==="## "+h))
    if(missing.length) throw new AppError("REPORT_INVALID","Required level-2 report headings: "+missing.join(", "),3)
    await atomicWrite(path.join(this.root,"reports/pre-forecast.md"),"# Pre-Forecast Report\n\n"+content+"\n")
    this.store.set("pre_report",{sha256:digest(content),content})
    return {saved:true}
  }
  async saveSkill(name: string, description: string, content: string) {
    const phase = this.store.get<RunState>("state")!.phase
    if (phase !== "skills-review") throw new AppError("PHASE_GATE","Skills are generated during skills-review",3)
    if (!/^[a-z0-9][a-z0-9-]{0,63}$/.test(name) || content.length < 30 || description.length < 10)
      throw new AppError("SKILL_INVALID","Invalid skill name or insufficient description/content",3)
    const skill = {name,description,content,sha256:digest(content),created_at:now()}
    const skills = (this.store.get<any[]>("skills") ?? []).filter(s=>s.name!==name)
    await writeJSON(path.join(this.root,"skills",name,"skill.json"),skill)
    await atomicWrite(path.join(this.root,"skills",name,"SKILL.md"),`# ${name}\n\n${description}\n\n${content}\n`)
    this.store.set("skills",[...skills,skill]); return {name,sha256:skill.sha256}
  }
  async memoryList() {
    if (!this.config.memory.enabled || !await exists(this.config.memory.directory)) return []
    const files = (await readdir(this.config.memory.directory)).filter(f=>/^[a-f0-9]{64}\.json$/.test(f))
    return await Promise.all(files.map(async f=>{
      const skill=await readJSON(path.join(this.config.memory.directory,f))
      return {id:f.slice(0,-5),name:skill.name,description:skill.description}
    }))
  }
  async memoryImport(hash: string) {
    if (!this.config.memory.enabled || !/^[a-f0-9]{64}$/.test(hash)) throw new AppError("MEMORY_INVALID","Invalid memory identifier",3)
    const skill=await readJSON(path.join(this.config.memory.directory,hash+".json"))
    if(digest(skill.content)!==hash) throw new AppError("MEMORY_INVALID","Memory content hash mismatch",3)
    return await this.saveSkill(skill.name,skill.description,skill.content)
  }
  async publishMemory() {
    if (!this.config.memory.enabled || !this.config.memory.publish_on_forecast_complete) return
    await mkdir(this.config.memory.directory,{recursive:true})
    for(const skill of this.store.get<any[]>("skills") ?? [])
      await writeJSON(path.join(this.config.memory.directory,skill.sha256+".json"),{...skill,source_task:this.store.get<RunState>("state")!.taskId})
  }
  async configurePolicy() {
    if(this.store.get("policy")) return this.store.get("policy")
    if(!this.config.task.reflection_enabled) return {enabled:false}
    if(this.store.get<RunState>("state")!.phase!=="skills-review") throw new AppError("PHASE_GATE","Window policy must be frozen during skills-review",3)
    const policy=await this.domain(ForecastWindowReflectionTool,{subcommand:"configure"})
    this.store.set("policy",policy); return policy
  }

  async experiment(model: string, params: Record<string,unknown> = {}, prior = false): Promise<Experiment> {
    checkAbort(this.signal)
    model=canonicalForecastModelName(model)
    if(!prior && this.store.get<RunState>("state")!.phase!=="forecast") throw new AppError("PHASE_GATE","Experiments require forecast phase",3)
    if(!prior && this.config.decisions.prediction_prior.mode==="csv") throw new AppError("PRIOR_MODE","Model search is disabled for UserPrior tasks",3)
    if(!prior && !this.config.task.model_families.some(allowed=>allowed===inferForecastModelFamily(model)||(!["statistical","deep_learning","foundation","baseline","user_prior"].includes(allowed)&&canonicalForecastModelName(allowed)===model))) throw new AppError("MODEL_DENIED","Model is outside frozen allowed models",3)
    for(const [key,value] of Object.entries(this.config.task))if(key in params&&canonical(params[key])!==canonical(value))throw new AppError("SPEC_FROZEN","Hyperparameters cannot change frozen task field "+key,3)
    const hyperparams={...params}
    for(const [alias,key] of [["epochs","train_epochs"],["kernel","moving_avg"]])if(alias in hyperparams){if(key in hyperparams && hyperparams[key]!==hyperparams[alias])throw new AppError("PARAM_CONFLICT","Conflicting model parameter aliases",2);hyperparams[key]=hyperparams[alias];delete hyperparams[alias]}
    if(model==="ARIMA" && ["p","d","q"].some(k=>k in hyperparams)) {hyperparams.order=[hyperparams.p??1,hyperparams.d??1,hyperparams.q??1];delete hyperparams.p;delete hyperparams.d;delete hyperparams.q}
    for(const key of ["seed","freq","device","threads"])delete hyperparams[key]
    const spec={...this.config.task,model,params,hyperparams,seed:params.seed??42,freq:params.freq??this.config.task.frequency,input_contract_version:model==="UserPrior"?5:3,...(model==="UserPrior"?{source_type:"user_prediction_prior"}:{}),phase:"forecast",eval_split:"val",adjustment_fit_holdout_ratio:.5}
    const runId=digest(canonical(spec))
    const cached=this.validationRuns.find(r=>r.run_id===runId)
    if(cached) { await this.verifyRun(cached); return cached }
    if(this.store.get("selected")||this.store.get<any>("domain-state")?.adjustment.source_model_set_locked)throw new AppError("SEARCH_LOCKED","Raw model search has already been frozen",3)
    const reflections=this.store.get<Record<string,unknown>>("reflections") ?? {}
    const attempts=this.store.get<any[]>("attempts")??[]
    const last=attempts.at(-1)
    if(last && !(last.run_id===runId&&last.status==="started") && !(this.store.get<any[]>("reflection_records")??[]).some(r=>r.run_id===last.run_id&&r.attempt_index===last.attempt_index))throw new AppError("REFLECTION_REQUIRED","Reflect on the previous successful or failed attempt before starting another",3)
    const previousAttempts=attempts.filter(a=>a.run_id===runId)
    const existing=previousAttempts.at(-1)
    if(!existing&&new Set(attempts.map(a=>a.run_id)).size>=this.config.execution.max_experiments)throw new AppError("EXPERIMENT_BUDGET","Unique experiment budget exhausted; finish search",3)
    if(!existing&&(this.store.get<number>("no_improvement")??0)>=this.config.execution.max_consecutive_no_improvement)throw new AppError("NO_IMPROVEMENT","No-improvement threshold reached; finish search",3)
    if((this.store.get<number>("consecutive_crashes")??0)>=this.config.execution.max_consecutive_crashes)throw new AppError("CRASH_THRESHOLD","Consecutive crash threshold reached",4)
    const attempt=existing?.status==="started"?existing:{run_id:runId,spec,status:"started",attempt_index:(existing?.attempt_index??0)+1,started_at:now()}
    if(attempt!==existing)attempts.push(attempt)
    this.store.set("attempts",attempts)
    const output=path.join(this.root,"runs",runId)
    try {
      const resultFile=path.join(output,"result.json")
      let result:Experiment
      if(await exists(resultFile)) {
        result=await readJSON(resultFile)
        const recoveredSpec={...result.spec};delete recoveredSpec.prediction_grid_target_anchor_index
        if(canonical(recoveredSpec)!==canonical(spec) || result.run_id!==runId) throw new AppError("INTEGRITY_FAILED","Recovered experiment identity mismatch",3)
        await this.verifyRun(result)
      } else {
        result=await compute(this.config,{operation:"experiment",spec,split:"val",run_id:runId,output_dir:output,
          ...(prior && this.config.decisions.prediction_prior.mode==="csv" ? {prior_path:this.config.decisions.prediction_prior.path} : {})},this.root,this.signal)
      }
      const previous=this.best?.metrics.mse ?? Infinity
      this.store.transaction(()=>{
        const list=this.runs.filter(r=>r.run_id!==runId); this.store.set("runs",[...list,result])
        Object.assign(attempt,{status:"success",metrics:result.metrics,improved:result.metrics.mse!<previous,completed_at:now()});this.store.set("attempts",attempts)
        this.store.set("consecutive_crashes",0)
        this.store.set("no_improvement",result.metrics.mse!<previous ? 0 : (this.store.get<number>("no_improvement") ?? 0)+1)
        this.store.event("experiment.completed",{runId,model,metrics:result.metrics})
      })
      await this.syncCounters(attempt)
      await this.projectHistory()
      return result
    } catch(e) {
      if(this.signal.aborted) throw e
      Object.assign(attempt,{status:"failed",error:errorText(e),completed_at:now()})
      const failure=e instanceof AppError&&e.code.startsWith("IPC_")?"ipc_error":e instanceof AppError&&e.code==="TOOL_TIMEOUT"?"timeout":"crash"
      attempt.failure_status=failure;this.store.set("attempts",attempts)
      const crashes=failure==="crash"?(this.store.get<number>("consecutive_crashes")??0)+1:0
      this.store.set("consecutive_crashes",crashes)
      if(failure!=="crash")this.store.set("no_improvement",(this.store.get<number>("no_improvement")??0)+1)
      await this.syncCounters(attempt)
      await this.projectHistory()
      this.store.event("experiment.failed",{runId,attempt_index:attempt.attempt_index,status:failure,error:errorText(e)})
      if(crashes>=this.config.execution.max_consecutive_crashes) throw new AppError("CRASH_THRESHOLD","Consecutive crash threshold reached: "+errorText(e),4)
      throw e
    }
  }
  async syncCounters(attempt:any) {
    const noImprovement=this.store.get<number>("no_improvement")??0,crashes=this.store.get<number>("consecutive_crashes")??0
    const exhausted=new Set((this.store.get<any[]>("attempts")??[]).map(a=>a.run_id)).size>=this.config.execution.max_experiments
    await this.domainState(s=>({...s,consecutive_no_improvement:noImprovement,consecutive_crashes:crashes,
      stop_counter_last_run_id:attempt.run_id,stop_counter_last_attempt_index:attempt.attempt_index,budget_initialized:true,
      budget_thresholds:{max_experiments:this.config.execution.max_experiments,no_improvement_rounds:this.config.execution.max_consecutive_no_improvement,crash_threshold:this.config.execution.max_consecutive_crashes},
      stop_reason:exhausted?"budget_exhausted":crashes>=this.config.execution.max_consecutive_crashes?"crash_threshold":noImprovement>=this.config.execution.max_consecutive_no_improvement?"no_improvement":null}))
  }
  async projectHistory() {
    const records:any[]=[]
    for(const attempt of this.store.get<any[]>("attempts")??[]){
      if(attempt.status==="started")continue
      const run=this.runs.find(r=>r.run_id===attempt.run_id)
      const spec=run?await readJSON(path.join(this.root,"runs",run.run_id,"spec.json")):attempt.spec
      const reflection=(this.store.get<any[]>("reflection_records")??[]).find(r=>r.run_id===attempt.run_id&&r.attempt_index===attempt.attempt_index)
      records.push({run_id:attempt.run_id,attempt_index:attempt.attempt_index,model:spec.model,spec,status:attempt.status==="success"?"success":attempt.failure_status,
        metrics:attempt.metrics??null,elapsed_seconds:Math.max(0,(Date.parse(attempt.completed_at)-Date.parse(attempt.started_at))/1000),timestamp:attempt.completed_at,
        ...(attempt.error?{failure_reason:attempt.error}:{}),...(reflection?{decision:reflection.decision==="continue"||reflection.decision==="stop"?"keep":reflection.decision}:{})})
      if(reflection)records.push(reflection)
    }
    await atomicWrite(path.join(this.root,"history.jsonl"),records.map(r=>JSON.stringify(r)).join("\n")+"\n")
  }
  async verifyRun(run: Experiment) {
    try {
      if(digest(await readFile(run.artifact))!==run.artifact_sha256 || !await exists(run.predictions))throw new Error("Model or predictions changed")
      const metadata=await readJSON(run.artifact)
      for(const [file,hash] of Object.entries(metadata.files??{}))if(path.basename(file)!==file||digest(await readFile(path.join(path.dirname(run.artifact),file)))!==hash)throw new Error("Model file hash mismatch")
      const directory=path.join(this.root,"runs",run.run_id)
      const evaluated=EvalResult.parse(await readJSON(path.join(directory,"eval.json")))
      await readBoundModelArtifactIdentity(directory,run.model,evaluated)
      if(canonical(evaluated.metrics)!==canonical(run.metrics))throw new Error("Metric binding mismatch")
    }catch(error){throw new AppError("INTEGRITY_FAILED","Experiment artifact missing or changed: "+run.run_id+"; "+errorText(error),3)}
  }

  async reflect(runId:string,decision:string,rationale:string,details?:{effective?:boolean;next_direction?:string}) {
    const attempt=(this.store.get<any[]>("attempts")??[]).findLast(a=>a.run_id===runId)
    if(!attempt||attempt.status==="started"||rationale.length<10)throw new AppError("REFLECTION_INVALID","Unknown or unfinished experiment attempt, or missing rationale",3)
    if(attempt.status==="success"&&decision==="crash")throw new AppError("REFLECTION_INVALID","A successful experiment cannot be marked as a crash",3)
    const records=this.store.get<any[]>("reflection_records")??[]
    let record=records.find(r=>r.run_id===runId&&r.attempt_index===attempt.attempt_index)
    if(!record){
      record={type:"reflection",run_id:runId,attempt_index:attempt.attempt_index,decision,effective:details?.effective??!!attempt.improved,
        reason:rationale,rationale,next_direction:details?.next_direction??"Follow the frozen budget and evaluate the next evidence-backed direction.",timestamp:now()}
      records.push(record)
      this.store.transaction(()=>{this.store.set("reflection_records",records);this.store.set("reflections",{...(this.store.get<any>("reflections")??{}),[runId]:record})})
    }
    await this.projectHistory()
    return record
  }
  async evaluate(runId:string) {
    const run=this.runs.find(r=>r.run_id===runId)
    if(!run)throw new AppError("RUN_MISSING","Unknown completed experiment",3)
    if(run.split==="test"&&this.store.get<RunState>("state")?.phase!=="post-forecast")throw new AppError("PHASE_GATE","Test metrics require post-forecast",3)
    await this.verifyRun(run)
    const before=this.validationRuns.slice(0,this.validationRuns.findIndex(r=>r.run_id===runId))
    const best=[...before].sort((a,b)=>a.metrics.mse!-b.metrics.mse!)[0]
    return {status:"success",run_id:runId,metrics:run.metrics,eval_split:run.split,comparison:{improved:!best||run.metrics.mse!<best.metrics.mse!,has_baseline:!!best,
      delta:best?{mse_delta:run.metrics.mse!-best.metrics.mse!,mae_delta:run.metrics.mae!-best.metrics.mae!}:null}}
  }
  async replay(source: Experiment, split: "adjustment_fit"|"test") {
    await this.verifyRun(source)
    const runId=split+"_"+source.run_id
    const cached=this.runs.find(r=>r.run_id===runId)
    if(cached) {await this.verifyRun(cached);return cached}
    const result=await compute(this.config,{operation:"experiment",spec:source.spec,split,run_id:runId,
      source_artifact:source.artifact,source_sha256:source.artifact_sha256,source_run_id:source.run_id,
      output_dir:path.join(this.root,"runs",runId)},this.root,this.signal)
    this.store.set("runs",[...this.runs,result]);return result as Experiment
  }
  async lockSelection() {
    if((this.store.get<number>("consecutive_crashes")??0)>=this.config.execution.max_consecutive_crashes)
      throw new AppError("CRASH_THRESHOLD","Consecutive crash threshold prevents continuing this task",4)
    if(!this.best) throw new AppError("NO_MODEL","No successful validation experiment",3)
    const reflections=this.store.get<Record<string,unknown>>("reflections") ?? {}
    if((this.store.get<any[]>("attempts")??[]).some(a=>a.status!=="started"&&!(this.store.get<any[]>("reflection_records")??[]).some(r=>r.run_id===a.run_id&&r.attempt_index===a.attempt_index)))throw new AppError("REFLECTION_REQUIRED","Every completed attempt, including failures, requires a paired reflection",3)
    if(!this.store.get("family_representatives")) {
      const eligible=this.validationRuns.filter(r=>(reflections[r.run_id] as any)?.decision!=="discard")
      const groups=new Map<string,Experiment>()
      for(const run of eligible){const family=inferForecastModelFamily(run.model)??run.model;const best=groups.get(family);if(!best || run.metrics.mse!<best.metrics.mse!)groups.set(family,run)}
      if(!groups.size)throw new AppError("NO_MODEL","No retained validation experiment",3)
      const representatives=[...groups.values()]
      this.store.set("family_representatives",representatives)
      const validationBest=[...representatives].sort((a,b)=>a.metrics.mse!-b.metrics.mse!)[0]
      this.store.set("validation_best",validationBest);this.store.set("selected",validationBest)
      await this.projectHistory()
      this.store.event("models.locked",{runIds:representatives.map(r=>r.run_id),basis:"per_family_validation_mse"})
    }
    return this.store.get<Experiment>("selected")!
  }
  async testBest() {
    const representatives=this.store.get<Experiment[]>("family_representatives")
    if(!representatives?.length || this.store.get<RunState>("state")!.phase!=="post-forecast")throw new AppError("PHASE_GATE","Testing requires frozen validation family representatives and post-forecast phase",3)
    const tests:Experiment[]=[]
    for(const source of representatives)tests.push(await this.replay(source,"test"))
    const test=[...tests].sort((a,b)=>a.metrics.mse!-b.metrics.mse!)[0]
    const selected=representatives.find(r=>r.run_id===test.source_run_id)!
    this.store.transaction(()=>{this.store.set("test_collection",tests);this.store.set("selected",selected);this.store.set("test",test);this.store.set("selection_basis","raw_test_mse_among_validation_family_representatives")})
    const domain=await this.domainState()
    const applied=new Map<string,any>()
    if(domain.adjustment.active_skill_version)for(const raw of tests){const result=await this.applyActiveAdjustment(raw);if(result)applied.set(raw.run_id,result)}
    const adjustment=applied.get(test.run_id)
    if(adjustment){this.store.set("adjusted_test",adjustment);this.store.set("test_adjustment_candidate_id",adjustment.candidate_id)}
    const candidateId=adjustment?.candidate_id
    const baseline=this.store.get("adjusted_test")??test
    if(!this.store.get("final_output"))this.store.set("final_output",{...baseline,kind:candidateId?"adjusted":"raw"})
    this.store.event("model.selected",{runId:selected.run_id,testRunId:test.run_id,basis:"raw_test_mse_among_validation_family_representatives"})
    return {run_id:test.run_id,model:selected.model,metrics:test.metrics,family_tests:tests.map(r=>({run_id:r.run_id,model:r.model,metrics:r.metrics}))}
  }
  async applyActiveAdjustment(test:Experiment):Promise<any>{
    const manifest=await readJSON(path.join(this.root,"adjustments/skills/active/manifest.json"))
    if(manifest.status!=="locked")throw new AppError("ADJUSTMENT_UNLOCKED","The active skill must be locked before test",3)
    const testSpec=await readJSON(path.join(this.root,"runs",test.run_id,"spec.json"))
    let eligible=false
    for(const file of manifest.validation_results){const evidence=await readJSON(path.resolve(path.dirname(this.root),file));const spec=await readJSON(path.join(this.root,"runs",evidence.run_id,"spec.json"));if(evidence.acceptance_decision?.accepted && validationSpecMatchesPostForecastTestSpec(spec,testSpec))eligible=true}
    const state=await this.domainState()
    if(!state.adjustment.test_application_records.some((r:any)=>r.test_run_id===test.run_id))await this.domainState(s=>({...s,adjustment:{...s.adjustment,critic_user_confirmed:now(),test_collection_run_ids:(this.store.get<Experiment[]>("test_collection")??[]).map(r=>r.run_id),test_eligible_run_ids:[...new Set([...s.adjustment.test_eligible_run_ids,...(eligible?[test.run_id]:[])])],test_application_records:[...s.adjustment.test_application_records,{test_run_id:test.run_id,validation_run_id:test.source_run_id,model:test.model,decision:eligible?"apply":"skip",decision_at:now(),decision_actor:"auto_policy",decision_reason:"Frozen unattended auto_validate policy applies only a validation-accepted skill to the matching model spec."}]}}))
    if(!eligible)return
    await this.domain(ForecastAdjustmentApplyTool,{run_id:test.run_id,candidate_id:manifest.candidate_id,split:"test"})
    const evaluated=await this.domain(ForecastAdjustmentEvaluateTool,{run_id:test.run_id,candidate_id:manifest.candidate_id,split:"test"})
    return {candidate_id:manifest.candidate_id,metrics:evaluated.adjusted_metrics,predictions:path.join(this.root,"adjustments/test",test.run_id,manifest.candidate_id,"adjusted_predictions.csv")}
  }

  async finalReport(commentary: string) {
    const test=this.store.get<Experiment>("test"),selected=this.store.get<Experiment>("selected"),final=this.store.get<any>("final_output")
    if(!test || !selected || !final) throw new AppError("PHASE_GATE","Run test_best before writing the final report",3)
    if(this.config.task.reflection_enabled && !this.store.get("test_reflection")) throw new AppError("PHASE_GATE","Complete window reflection before writing the report",3)
    if(commentary.length<50) throw new AppError("REPORT_INVALID","Report commentary must explain actual findings",3)
    const familyTests=(this.store.get<Experiment[]>("test_collection")??[]).map(r=>`| ${r.model} | ${r.source_run_id} | ${r.metrics.mse} | ${r.run_id===test.run_id?"selected":"compared"} |`).join("\n")
    const body=`# Final Forecast Report\n\nExecution mode: ${this.config.execution.mode}\n\nModel: ${selected.model}\nValidation selection: ${selected.run_id}\nTest run: ${test.run_id}\nSelection criterion: validation MSE selects one representative per model family; raw test MSE selects the final model before adjustment or window refinement.\n\n`+
      `| Output | MSE | MAE | WAPE | MASE |\n| --- | --- | --- | --- | --- |\n`+
      [["Validation",selected.metrics],["Raw test",test.metrics],[`Final (${final.kind})`,final.metrics]].map(([label,m]:any)=>`| ${label} | ${m.mse} | ${m.mae} | ${m.wape ?? "undefined"} | ${m.mase ?? "undefined"} |`).join("\n")+
      `\n\n## Family representative raw test comparison\n\n| Model | Validation source | Raw test MSE | Result |\n| --- | --- | --- | --- |\n${familyTests}\n\nMetrics use the training-fitted standardized target scale. WAPE is a ratio on that scale. Undefined denominators are recorded as null.\n\n## Analysis\n\n${commentary}\n`
    await atomicWrite(path.join(this.root,"reports/final-report.md"),body)
    this.store.set("final_report",{sha256:digest(body),at:now()});return {saved:true}
  }
}
