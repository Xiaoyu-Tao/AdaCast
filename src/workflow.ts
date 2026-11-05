import path from "node:path"
import { mkdir, readFile } from "node:fs/promises"
import { AppError, VERSION, type Config, type Phase, type RunState } from "./contracts"
import { loadConfig, validateCredentials } from "./config"
import { Store, lockWorkspace } from "./storage"
import { ForecastService } from "./forecast"
import { CriticController } from "./critic"
import { LLMClient } from "./llm"
import { AgentRuntime } from "./runtime"
import { systemPrompt } from "./prompts"
import { registerTools } from "./tools"
import { buildDelivery, verifyBundle } from "./delivery"
import { compute } from "./python"
import { atomicWrite, canonical, checkAbort, digest, errorText, exists, id, now, sanitize, writeJSON } from "./util"

export async function validateTask(file:string,signal:AbortSignal) {
  const config=await loadConfig(file);validateCredentials(config)
  for(const input of [...config.knowledge.files,...config.instructions.files]) await readFile(input)
  if(config.decisions.prediction_prior.mode==="csv") await readFile(config.decisions.prediction_prior.path)
  const info=await compute(config,{operation:"inspect",skip_model_check:config.decisions.prediction_prior.mode==="csv"},undefined,signal)
  return {valid:true,execution_mode:config.execution.mode,task:config.task,...info}
}

export async function runTask(options:{configFile?:string;workspace:string;resume?:boolean;signal:AbortSignal;emit?:(type:string,value:unknown)=>void}) {
  const workspace=path.resolve(options.workspace),root=path.join(workspace,".forecast")
  if(options.resume && !await exists(path.join(root,"runtime/state.sqlite"))) throw new AppError("WORKSPACE_MISSING","No task exists at this workspace",2)
  const unlock=await lockWorkspace(workspace)
  let store:Store|undefined
  let heartbeat:ReturnType<typeof setInterval>|undefined
  let deadline:ReturnType<typeof setTimeout>|undefined
  let started=Date.now(),baseElapsed=0
  try {
    store=new Store(root)
    let config:Config
    if(options.resume) {
      config=store.get<Config>("config")!
      if(!config || !store.get("state")) throw new AppError("WORKSPACE_INVALID","Task initialization did not complete",3)
      for(const input of store.get<{path:string;sha256:string}[]>("inputs") ?? [])
        if(digest(await readFile(input.path))!==input.sha256) throw new AppError("INTEGRITY_FAILED","Frozen input changed: "+input.path,3)
      if(store.get<RunState>("state")!.lifecycle==="succeeded") {
        const manifest=await verifyBundle(path.join(root,"best-model"));return {status:"succeeded",resumed:true,manifest}
      }
    } else {
      if(store.get("state")) throw new AppError("WORKSPACE_EXISTS","Task already exists; use resume",3)
      if(!options.configFile) throw new AppError("CONFIG_MISSING","run requires --config",2)
      config=await loadConfig(options.configFile);validateCredentials(config)
      const locks:{path:string;sha256:string}[]=[]
      const snapshot=async(source:string,name:string)=>{
        const bytes=await readFile(source),target=path.join(root,"inputs",name)
        await atomicWrite(target,bytes);locks.push({path:target,sha256:digest(bytes)});return target
      }
      config.task.dataset_path=await snapshot(config.task.dataset_path,"dataset.csv")
      for(const category of ["knowledge","instructions"] as const) {
        const items=[]
        for(let i=0;i<config[category].files.length;i++) {
          const source=config[category].files[i],file=await snapshot(source,category+"-"+i+".txt")
          const content=await readFile(file,"utf8")
          if(Buffer.byteLength(content)>256000) throw new AppError("INPUT_TOO_LARGE","Knowledge/instruction input exceeds 256 KB",2)
          items.push({name:path.basename(source),content,sha256:digest(content)});config[category].files[i]=file
        }
        store.set(category,items)
      }
      if(config.decisions.prediction_prior.mode==="csv") config.decisions.prediction_prior.path=await snapshot(config.decisions.prediction_prior.path,"prior.csv")
      const fingerprint=digest(canonical({config,inputs:locks.map(i=>i.sha256),version:VERSION}))
      const checked=await compute(config,{operation:"inspect",skip_model_check:config.decisions.prediction_prior.mode==="csv"},root,options.signal)
      store.set("data_info",checked)
      const frozen={...config.task,train_ratio:config.task.train_ratio??checked.boundaries.train[1]/checked.rows,val_ratio:config.task.val_ratio??(checked.boundaries.test[0]-checked.boundaries.train[1])/checked.rows,test_ratio:config.task.test_ratio??(checked.rows-checked.boundaries.test[0])/checked.rows,adjustment_fit_holdout_ratio:.5,fingerprint,frozen_at:now()}
      await writeJSON(path.join(root,"task.json"),frozen)
      locks.push({path:path.join(root,"task.json"),sha256:digest(await readFile(path.join(root,"task.json")))})
      await writeJSON(path.join(root,"task-lock.json"),{version:1,task_sha256:locks.at(-1)!.sha256,frozen_at:frozen.frozen_at,fingerprint,inputs:locks})
      store.transaction(()=>{
        store!.set("config",config);store!.set("inputs",locks);store!.set("task_fingerprint",fingerprint)
        store!.set("state",{lifecycle:"pending",phase:"init",taskId:id("task"),createdAt:now(),elapsedMs:0} satisfies RunState)
        store!.event("task.frozen",{fingerprint,execution_mode:config.execution.mode})
      })
    }
    validateCredentials(config)
    const activeStore=store
    const state=store.get<RunState>("state")!
    started=Date.now();baseElapsed=state.elapsedMs
    const controller=new AbortController()
    const remaining=config.execution.task_timeout_seconds*1000-baseElapsed
    if(remaining<=0) throw new AppError("TASK_TIMEOUT","Task time budget exhausted",5)
    deadline=setTimeout(()=>controller.abort(new AppError("TASK_TIMEOUT","Task time budget exhausted",5,true)),remaining)
    const signal=AbortSignal.any([options.signal,controller.signal])
    heartbeat=setInterval(()=>{
      const current=activeStore.get<RunState>("state")!
      activeStore.set("state",{...current,elapsedMs:baseElapsed+Date.now()-started})
    },1000)
    store.set("state",{...state,lifecycle:"running",error:undefined})
    const llm=new LLMClient(config,store),forecast=new ForecastService(config,store,signal)
    const critic=new CriticController(config,store,llm,signal)
    const runtime=new AgentRuntime(config,store,llm,agent=>systemPrompt(agent,config,forecast,activeStore),signal)
    registerTools(runtime,forecast,critic,config)
    const emit=(type:string,value:unknown={})=>{activeStore.event(type,value);options.emit?.(type,value)}
    const transition=async(phase:Phase)=>{
      const current=activeStore.get<RunState>("state")!
      activeStore.transaction(()=>{activeStore.set("state",{...current,phase});activeStore.set("handoff",forecast.promptState());activeStore.event("phase.changed",{from:current.phase,to:phase})})
      await forecast.domainState(s=>({...s,phase,skills_confirmed:!["init","pre-forecast","skills-review"].includes(phase),skills_confirmed_at:s.skills_confirmed_at??(phase==="forecast"?now():null),pre_forecast_completed:phase!=="pre-forecast",pre_forecast_completed_at:s.pre_forecast_completed_at??(phase==="skills-review"?now():null)}))
      await writeJSON(path.join(root,"CONTEXT.json"),forecast.state());await activeStore.project();options.emit?.("phase.changed",{phase})
    }
    await mkdir(path.join(workspace,"scratch"),{recursive:true})
    while(true) {
      checkAbort(signal)
      const phase=store.get<RunState>("state")!.phase
      emit("phase.started",{phase,mode:config.execution.mode})
      if(phase==="init") {
        const info=await compute(config,{operation:"inspect",skip_model_check:config.decisions.prediction_prior.mode==="csv"},root,signal)
        store.set("data_info",info);await writeJSON(path.join(root,"reports/data-check.json"),info)
        await transition("pre-forecast")
      } else if(phase==="pre-forecast") {
        if(config.execution.mode==="deterministic") {
          const analysis=await forecast.analyze()
          await forecast.preReport(`## Task Summary\nPredict ${config.task.target_col} using ${config.task.seq_len} historical steps for ${config.task.pred_len} future steps.\n\n## Data Characteristics\nTraining-only statistics: ${JSON.stringify(analysis.train_statistics)}.\n\n## Domain Context\n${JSON.stringify(store.get("knowledge")??[])}\n\n## Risk Flags\nEvaluation uses training-standardized targets and rolling historical context.\n\n## Modeling Directions\nCompare explicitly configured models by held-out validation MSE. Execution mode is deterministic; no LLM reasoning is performed.`)
        } else await runtime.run("planner","planner:pre-forecast","Complete pre-forecast analysis and report now.",()=>[
          ...(!store!.get("analysis")?["analyze_dataset"]:[]),...(!store!.get("pre_report")?["pre_forecast_report"]:[])])
        await transition("skills-review")
      } else if(phase==="skills-review") {
        if(config.execution.mode==="deterministic") await forecast.saveSkill("baseline-models","Compare configured baseline forecasting models","Use the frozen split and training-only scale. Rank models by held-out validation MSE. Reuse fitted artifacts for test inference.")
        else await runtime.run("planner","planner:skills-review","Create and validate task-specific modeling Skills. Configure window policy if enabled.",()=>[
          ...(!(store!.get<any[]>("skills")??[]).length?["forecast_skill"]:[]),...(config.task.reflection_enabled&&!store!.get("policy")?["configure_window_policy"]:[])])
        await transition("forecast")
      } else if(phase==="forecast") {
        if(config.decisions.prediction_prior.mode==="csv" && !forecast.validationRuns.length) {
          const prior=await forecast.experiment("UserPrior",{},true);await forecast.reflect(prior.run_id,"keep","Imported and validated the explicitly supplied forecast prior; no model training or search occurred.")
        }
        if(config.execution.mode==="deterministic") {
          if(config.decisions.prediction_prior.mode!=="csv") for(const configuredModel of config.task.model_families) {
            const model=({statistical:"ARIMA",deep_learning:"DLinear",foundation:"Chronos",baseline:"Naive"} as Record<string,string>)[configuredModel]??configuredModel
            try {const r=await forecast.experiment(model);await forecast.reflect(r.run_id,"keep",`Validation MSE ${r.metrics.mse}; compare using the same frozen windows and scale.`)}
            catch(e) {if(signal.aborted || e instanceof AppError&&([5,130,143].includes(e.exitCode)||e.code==="CRASH_THRESHOLD")) throw e;const failed=(store!.get<any[]>("attempts")??[]).at(-1);if(failed?.status==="failed")await forecast.reflect(failed.run_id,"crash",`Experiment failed: ${errorText(e)}. Continue only within the frozen budget.`);emit("experiment.skipped",{model,error:errorText(e)})}
          }
          await forecast.lockSelection();store.set("search_done",{reason:"Configured baseline comparison complete"})
        } else await runtime.run("forecaster","forecaster:forecast","Execute forecasting search, reflections and configured adjustment, then call finish_search.",()=>store!.get("search_done")?[]:["finish_search after all required work"])
        if(config.task.reflection_enabled) await critic.run("val")
        await forecast.publishMemory();await transition("post-forecast")
      } else {
        if(config.execution.mode==="deterministic") {
          await forecast.testBest()
          await forecast.finalReport("The configured models were evaluated using identical held-out validation windows. Validation MSE selected one representative per model family. Raw test MSE selected the delivery model among those representatives. The exact fitted artifact was reused for testing. This run uses deterministic orchestration and performs no LLM inference or semantic prediction adjustment.")
        } else await runtime.run("critic","critic:post-forecast","Complete test inference, configured window reasoning and final report using the available tools.",()=>[
          ...(!store!.get("test")?["test_best"]:[]),...(config.task.reflection_enabled&&!store!.get("test_reflection")?["run_window_reflection"]:[]),...(!store!.get("final_report")?["final_report"]:[])])
        const manifest=await buildDelivery(config,store,signal)
        store.set("state",{...store.get<RunState>("state")!,lifecycle:"succeeded",completedAt:now(),elapsedMs:baseElapsed+Date.now()-started})
        emit("task.succeeded",{model:manifest.model,delivery:path.join(root,"best-model")});await store.project()
        return {status:"succeeded",workspace,mode:config.execution.mode,manifest}
      }
    }
  } catch(e) {
    if(store?.get("state")) {
      const error=e instanceof AppError?e:new AppError("RUN_FAILED",sanitize(errorText(e)))
      const state=store.get<RunState>("state")!
      // Invocation errors must not overwrite a completed or another frozen task's status.
      if(!["WORKSPACE_EXISTS","LLM_KEY_MISSING","LLM_CONFIG_MISSING"].includes(error.code) && state.lifecycle!=="succeeded") {
        store.set("state",{...state,lifecycle:[130,143].includes(error.exitCode)?"interrupted":error.exitCode===3?"blocked":"failed",
          elapsedMs:heartbeat?baseElapsed+Date.now()-started:state.elapsedMs,error:{code:error.code,message:sanitize(error.message)}})
        store.event("task.stopped",{code:error.code,message:sanitize(error.message)});await store.project()
      }
    }
    throw e
  } finally {if(heartbeat) clearInterval(heartbeat);if(deadline) clearTimeout(deadline);store?.close();await unlock()}
}
