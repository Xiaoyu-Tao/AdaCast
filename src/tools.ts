import { ForecastAdjustmentSourceModelsTool } from "./domain/forecast-adjustment-source-models"
import { ForecastAdjustmentFitRunsTool } from "./domain/forecast-adjustment-fit-runs"
import { ForecastAdjustmentDiagnosticsTool } from "./domain/forecast-adjustment-diagnostics"
import { ForecastAdjustmentCandidateTool } from "./domain/forecast-adjustment-candidate"
import { ForecastAdjustmentApplyTool } from "./domain/forecast-adjustment-apply"
import { ForecastAdjustmentEvaluateTool } from "./domain/forecast-adjustment-evaluate"
import { ForecastAdjustmentSkillTool } from "./domain/forecast-adjustment-skill"
import { ForecastAdjustmentPolicyTool } from "./domain/forecast-adjustment-policy"
import { ForecastWindowReflectionTool } from "./domain/forecast-window-reflection"
import { z } from "zod"
import path from "node:path"
import { mkdir, readFile, readdir } from "node:fs/promises"
import { AppError, type Config, type RunState, type ToolDefinition } from "./contracts"
import { ForecastService } from "./forecast"
import type { CriticController } from "./critic"
import type { AgentRuntime } from "./runtime"
import { atomicWrite, digest, inside, safePath } from "./util"
import { compute, runProcess } from "./python"

export function registerTools(runtime:AgentRuntime,service:ForecastService,critic:CriticController,config:Config) {
  const root=service.root,workspace=path.dirname(root),scratch=path.join(workspace,"scratch")
  const add=(name:string,description:string,schema:z.ZodType,roles:string[],execute:ToolDefinition["execute"],replaySafe=true)=>
    runtime.register({name,description,schema,roles,execute,replaySafe})
  for(const tool of [ForecastAdjustmentSourceModelsTool,ForecastAdjustmentFitRunsTool,ForecastAdjustmentDiagnosticsTool,ForecastAdjustmentCandidateTool,ForecastAdjustmentApplyTool,ForecastAdjustmentEvaluateTool,ForecastAdjustmentSkillTool,ForecastAdjustmentPolicyTool,ForecastWindowReflectionTool]) {
    // Domain definitions are pure, so initialization does not require a workspace scope.
    const info=tool as any
    const roles=tool.id==="forecast_window_reflection"?["planner","forecaster","critic"]:tool.id==="forecast_adjustment_apply"||tool.id==="forecast_adjustment_evaluate"?["forecaster","critic"]:["forecaster"]
    // Tool.init returns the retained schema, available before the first LLM request.
    const definition=(tool as any).definition
    add(tool.id,definition.description,definition.parameters,roles,async(a,ctx)=>service.domain(info,a,ctx))
  }
  const empty=z.object({}).strict()
  add("forecast_state","Read frozen task, phase, aggregate metrics, budgets, skills and decisions.",empty,["*"],async()=>service.state())
  add("list_models","List the independent model registry, optional dependency requirements and currently available implementations. This does not download model weights.",empty,["planner","forecaster"],async()=>compute(config,{operation:"models"},service.root,service.signal))
  add("analyze_dataset","Analyze training targets only; produce quantitative analysis.",empty,["planner"],async()=>service.analyze())
  add("pre_forecast_report","Write analysis-backed pre-forecast report. Include Task Summary, Data Characteristics, Domain Context, Risk Flags and Modeling Directions.",
    z.object({content:z.string().min(100)}).strict(),["planner"],async a=>service.preReport(a.content))
  add("forecast_skill","Create a task-specific modeling Skill during skills-review.",z.object({name:z.string(),description:z.string(),content:z.string()}).strict(),["planner"],async a=>service.saveSkill(a.name,a.description,a.content))
  add("list_skills","Read frozen task Skills.",empty,["*"],async()=>service.store.get("skills") ?? [])
  add("list_memory","Find cross-task skill memory metadata; memory is an unvalidated prior.",empty,["planner"],async()=>service.memoryList())
  add("import_memory","Import a relevant memory Skill for current task review.",z.object({id:z.string()}).strict(),["planner"],async a=>service.memoryImport(a.id))
  add("configure_window_policy","Freeze training/validation-front-derived window boundaries before forecasting.",empty,["planner"],async()=>service.configurePolicy())
  add("generate_model","Fit an allowed model on held-out validation. Accept the reference spec/hyperparams interface or the Lite model/params interface. Frozen task fields cannot change.",
    z.object({model:z.string().optional(),params:z.record(z.string(),z.unknown()).optional(),spec:z.record(z.string(),z.unknown()).optional()}).strict().refine(a=>!!a.model||typeof a.spec?.model==="string","A model or spec.model is required"),["forecaster"],async a=>{
      if(a.model&&a.spec?.model&&a.model!==a.spec.model)throw new AppError("SPEC_CONFLICT","Conflicting model identifiers",3)
      const spec=a.spec??{}
      for(const [key,value] of Object.entries(config.task))if(key in spec){
        const supplied=key==="dataset_path"?path.resolve(workspace,String(spec[key])):spec[key]
        if(JSON.stringify(supplied)!==JSON.stringify(value))throw new AppError("SPEC_FROZEN","Experiment cannot change frozen task field "+key,3)
      }
      if((spec.phase??"forecast")!=="forecast"||(spec.eval_split??"val")!=="val")throw new AppError("PHASE_GATE","Forecaster generation only supports forecast/val; use test_best for frozen test replays",3)
      const params={...(spec.hyperparams as Record<string,unknown>??{}),...(a.params??{})}
      for(const key of ["seed","freq","device","threads"])if(key in spec)params[key]=spec[key]
      for(const [key,value] of Object.entries(spec))if(![...Object.keys(config.task),"model","hyperparams","phase","eval_split","input_contract_version","seed","freq","device","threads","window_features","metrics"].includes(key))params[key]=value
      const r=await service.experiment(a.model??spec.model,params);return {status:"success",run_id:r.run_id,model:r.model,metrics:r.metrics}
    })
  add("evaluate_experiment","Read evaluator metrics after verifying the source model and prediction artifact bindings.",z.object({run_id:z.string()}).strict(),["forecaster","critic"],async a=>service.evaluate(a.run_id))
  add("forecast_task","Read the already frozen task definition.",z.object({subcommand:z.literal("read").default("read")}).strict(),["*"],async()=>config.task)
  add("forecast_reflect","Record a reflection after each successful or failed experiment attempt. Reference reason/effective/next_direction and Lite rationale are supported.",
    z.object({run_id:z.string(),decision:z.enum(["continue","keep","discard","crash","stop"]),rationale:z.string().min(10).optional(),reason:z.string().min(10).optional(),effective:z.boolean().optional(),next_direction:z.string().optional()}).strict().refine(a=>!!a.rationale||!!a.reason,"A reflection reason is required"),
    ["forecaster"],async a=>service.reflect(a.run_id,a.decision,a.rationale??a.reason,{effective:a.effective,next_direction:a.next_direction}))
  add("decide_adjustment","Choose off, lightweight or comprehensive adjustment with a reason. Explicit config choices take precedence.",
    z.object({strategy:z.enum(["off","a_lightweight","b_comprehensive"]),reason:z.string().min(10)}).strict(),["forecaster"],async a=>{
      if(service.store.get("selected")) throw new AppError("PHASE_GATE","Adjustment decision is frozen",3)
      const fixed=config.decisions.adjustment.strategy
      if(fixed!=="agent" && fixed!==a.strategy) throw new AppError("CONFIG_CONFLICT","The requested strategy conflicts with frozen configuration",3)
      const previous=service.store.get<any>("adjustment_decision")
      if(previous && previous.strategy!==a.strategy) throw new AppError("DECISION_FROZEN","Adjustment choice is already recorded",3)
      const actor=fixed==="agent"?"llm_auto":"auto_policy"
      const decision={...a,actor};service.store.set("adjustment_decision",decision)
      await service.domainState(s=>({...s,adjustment:{...s.adjustment,user_decision:a.strategy==="off"?"disabled":a.strategy==="a_lightweight"?"enabled_a":"enabled_b",user_decision_at:new Date().toISOString(),user_decision_actor:actor,user_decision_reason:a.reason}}))
      await service.domain(ForecastAdjustmentPolicyTool,{subcommand:"configure",policy:{enabled:a.strategy!=="off",mode:a.strategy==="off"?"off":"rule_skill",strategy:a.strategy==="off"?null:a.strategy}})
      await service.domain(ForecastAdjustmentPolicyTool,{subcommand:"lock"})
      return decision
    })
  add("finish_search","Freeze one validation representative per model family and the accepted adjustment skill. Call after all reflections and required adjustment work.",z.object({reason:z.string().min(10)}).strict(),["forecaster"],async a=>{
      const configured=config.decisions.adjustment.strategy
      const decision=service.store.get<any>("adjustment_decision")
      if(configured!=="off" && !decision) throw new AppError("DECISION_REQUIRED","Call decide_adjustment before finishing",3)
      const strategy=decision?.strategy ?? configured
      if(strategy!=="off") {
        const state=await service.domainState()
        if(!state.adjustment.forecaster_locked || !(state.adjustment.active_skill_version || state.adjustment.no_effective_skill))throw new AppError("ADJUSTMENT_REQUIRED","Complete the source-model set, fit diagnostics, candidate validation, and lock_active or mark_no_effective before finishing",3)
      }
      const selected=await service.lockSelection();service.store.set("search_done",{reason:a.reason,selected:selected.run_id});return {selected:selected.run_id}
    })
  add("test_best","Reuse each frozen family representative for test inference, choose the lowest raw test MSE, then apply the locked skill to eligible models.",empty,["critic"],async()=>service.testBest())
  add("run_window_reflection","Run or resume isolated test-window evidence reasoning. No actual future target is exposed to these calls.",empty,["critic"],async()=>critic.run("test"))
  add("final_report","Write final analysis. Deterministic model IDs and metric tables are included automatically.",z.object({content:z.string().min(50)}).strict(),["critic"],async a=>service.finalReport(a.content))
  add("read_tool_output","Read a bounded slice of a truncated tool result.",z.object({output_id:z.string().regex(/^[a-f0-9]{64}$/),offset:z.number().int().min(0).default(0)}).strict(),["*"],async a=>{
      const text=await readFile(path.join(root,"runtime/tool-outputs",a.output_id+".json"),"utf8")
      if(digest(text.trimEnd())!==a.output_id && digest(JSON.stringify(JSON.parse(text)))!==a.output_id) throw new AppError("INTEGRITY_FAILED","Tool output hash mismatch",3)
      return {text:text.slice(a.offset,a.offset+6000),next_offset:a.offset+6000<text.length?a.offset+6000:null}
    })
  add("read_file","Read an explicitly accessible file: scratch, reports, frozen knowledge, Skills, or extra configured read paths. Raw datasets, run predictions and runtime state are protected.",
    z.object({path:z.string(),offset:z.number().int().nonnegative().default(0)}).strict(),["*"],async a=>{
      const requested=path.resolve(workspace,a.path)
      const roots=[scratch,path.join(root,"reports"),path.join(root,"skills"),...config.tools.extra_read_paths]
      const allowed=roots.find(r=>inside(r,requested))
      if(!allowed) throw new AppError("PATH_DENIED","File is outside explicitly readable paths; use structured forecasting tools",3)
      const file=await safePath(allowed,path.relative(allowed,requested))
      const text=await readFile(file,"utf8");return {text:text.slice(a.offset,a.offset+6000),next_offset:a.offset+6000<text.length?a.offset+6000:null}
    })
  add("write_file","Write generated code or notes to scratch or configured output directories.",z.object({path:z.string(),content:z.string().max(100000)}).strict(),["planner","forecaster","critic","general"],async a=>{
      const requested=path.resolve(workspace,a.path)
      if(inside(root,requested)) throw new AppError("PATH_DENIED","Managed forecast artifacts are protected",3)
      const allowed=[scratch,...config.tools.extra_write_paths].find(r=>inside(r,requested))
      if(!allowed) throw new AppError("PATH_DENIED","File is outside writable paths",3)
      await mkdir(allowed,{recursive:true});const file=await safePath(allowed,path.relative(allowed,requested))
      await atomicWrite(file,a.content);return {path:a.path,sha256:digest(a.content)}
    })
  add("list_files","List entries within the task scratch directory.",z.object({path:z.string().default(".")}).strict(),["*"],async a=>{
      await mkdir(scratch,{recursive:true});return await readdir(await safePath(scratch,a.path))
    })
  add("shell","Execute an explicitly allowlisted executable with argument array, no interactive stdin. Shell tools are not an OS sandbox.",
    z.object({command:z.string(),args:z.array(z.string()).default([])}).strict(),["general","forecaster"],async(a,ctx)=>{
      if(!config.tools.allow_shell || !config.tools.allowed_commands.includes(a.command)) throw new AppError("COMMAND_DENIED","Executable is not allowed by task configuration",3)
      await mkdir(scratch,{recursive:true})
      const env:NodeJS.ProcessEnv={PATH:process.env.PATH,TMPDIR:process.env.TMPDIR}
      return await runProcess(a.command,a.args,config.execution.tool_timeout_seconds*1000,ctx.signal,{cwd:scratch,env,maxBytes:2*1024*1024})
    },false)
  add("web_fetch","Fetch an explicitly requested HTTP(S) knowledge source when network tools are enabled.",z.object({url:z.string().url()}).strict(),["planner","general","explore"],async(a,ctx)=>{
      if(!config.tools.allow_network) throw new AppError("NETWORK_DENIED","Network tools are disabled",3)
      const url=new URL(a.url);if(!["https:","http:"].includes(url.protocol) || url.username || url.password) throw new AppError("URL_DENIED","Invalid source URL",3)
      const response=await fetch(url,{signal:AbortSignal.any([ctx.signal,AbortSignal.timeout(20000)])})
      if(!response.ok) throw new AppError("FETCH_FAILED",`Source returned HTTP ${response.status}`)
      const text=await response.text();return {url:a.url,content:text.slice(0,10000),truncated:text.length>10000}
    })
  add("task","Run a bounded general/explore/custom subagent task with isolated messages; use task_id to resume a previous subtask.",
    z.object({agent:z.string(),prompt:z.string().min(1),task_id:z.string().optional()}).strict(),["planner","forecaster","critic","general"],async(a,ctx)=>{
      if(!["general","explore",...Object.keys(config.agents)].includes(a.agent)) throw new AppError("AGENT_DENIED","Unknown helper agent",3)
      const session=a.task_id ?? "sub_"+digest(ctx.sessionId+ctx.callId).slice(0,24)
      if(!/^sub_[a-f0-9]{24}$/.test(session)) throw new AppError("SESSION_INVALID","Invalid subtask identifier",3)
      const owner=service.store.get<any>("subtask:"+session)
      if(owner && (owner.parent!==ctx.sessionId || owner.agent!==a.agent)) throw new AppError("SESSION_DENIED","Subtask belongs to a different parent or role",3)
      service.store.set("subtask:"+session,{parent:ctx.sessionId,agent:a.agent})
      const result=await runtime.run(a.agent,session,a.prompt,undefined,ctx.depth+1,!!a.task_id);return {task_id:session,result}
    })
}
