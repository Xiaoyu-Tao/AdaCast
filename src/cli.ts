#!/usr/bin/env bun
import { compute } from "./python"
import { ConfigSchema } from "./contracts"
import path from "node:path"
import { parseArgs } from "node:util"
import { AppError, VERSION } from "./contracts"
import { runTask, validateTask } from "./workflow"
import { Store } from "./storage"
import { exportDelivery } from "./delivery"
import { errorText, exists, sanitize } from "./util"

const help=`adacast-lite ${VERSION}\n\nCommands:\n  models [--config TASK.json]\n  validate --config TASK.json\n  run --config TASK.json --workspace DIR\n  resume --workspace DIR\n  status --workspace DIR\n  export --workspace DIR --output DIR\n\nOptions: --json --python PATH --help --version\n\nAgent mode uses the configured LLM environment variables. Deterministic mode runs actual local baselines without LLM calls.\n`
export async function main(argv=process.argv.slice(2)):Promise<number> {
  let json=false
  const controller=new AbortController()
  const interrupt=()=>controller.abort(new AppError("INTERRUPTED","Interrupted by SIGINT",130,true))
  const terminate=()=>controller.abort(new AppError("INTERRUPTED","Interrupted by SIGTERM",143,true))
  process.on("SIGINT",interrupt);process.on("SIGTERM",terminate)
  try {
    const {values,positionals}=parseArgs({args:argv,allowPositionals:true,strict:true,options:{
      config:{type:"string"},workspace:{type:"string"},output:{type:"string"},python:{type:"string"},
      json:{type:"boolean"},help:{type:"boolean"},version:{type:"boolean"}}})
    json=values.json??false
    if(values.python) process.env.ADACAST_LITE_PYTHON=path.resolve(values.python)
    if(values.version) {console.log(VERSION);return 0}
    if(values.help || !positionals.length) {console.log(help);return 0}
    if(positionals.length!==1) throw new AppError("ARGUMENT_INVALID","Expected a single command",2)
    const command=positionals[0]
    const workspace=values.workspace?path.resolve(values.workspace):undefined
    const emit=(type:string,value:unknown)=>{
      if(json) console.log(JSON.stringify({type,data:value}))
      else if(["phase.changed","phase.started","task.succeeded","experiment.skipped"].includes(type)) console.error(type,JSON.stringify(value))
    }
    let result:unknown
    if(command==="models") {
      // Catalog queries do not require credentials or input data.
      const config=values.config?(await import("./config")).loadConfig(path.resolve(values.config)):Promise.resolve(ConfigSchema.parse({schema_version:1,task:{dataset_path:"CATALOG_ONLY",time_col:"date",target_col:"target",frequency:"h",seq_len:1,pred_len:1,train_ratio:.6,val_ratio:.2,test_ratio:.2,model_families:["Naive"],reflection_enabled:false},execution:{mode:"deterministic"},decisions:{adjustment:{strategy:"off"}}}))
      result=await compute(await config,{operation:"models"},undefined,controller.signal)
    } else if(command==="validate") {
      if(!values.config) throw new AppError("ARGUMENT_MISSING","validate requires --config",2)
      result=await validateTask(path.resolve(values.config),controller.signal)
    } else if(command==="run" || command==="resume") {
      if(!workspace || command==="run"&&!values.config) throw new AppError("ARGUMENT_MISSING","Specify --workspace and, for run, --config",2)
      result=await runTask({configFile:values.config?path.resolve(values.config):undefined,workspace,resume:command==="resume",signal:controller.signal,emit})
    } else if(command==="status" || command==="export") {
      if(!workspace || !await exists(path.join(workspace,".forecast/runtime/state.sqlite"))) throw new AppError("WORKSPACE_MISSING","Specify an existing task workspace",2)
      const store=new Store(path.join(workspace,".forecast"),true)
      try {
        if(command==="status") result={state:store.get("state"),usage:store.usage(),selected:store.get("selected"),delivery:store.get("delivery")}
        else {
          if(!values.output) throw new AppError("ARGUMENT_MISSING","export requires --output",2)
          if(store.get<any>("state")?.lifecycle!=="succeeded") throw new AppError("DELIVERY_INCOMPLETE","Only complete tasks can be exported",3)
          result=await exportDelivery(workspace,path.resolve(values.output))
        }
      } finally {store.close()}
    } else throw new AppError("COMMAND_UNKNOWN","Unknown command: "+command,2)
    console.log(json?JSON.stringify({type:"result",data:result}):JSON.stringify(result,null,2));return 0
  } catch(e) {
    const error=e instanceof AppError?e:new AppError("EXECUTION_FAILED",sanitize(errorText(e)),e instanceof TypeError?2:4)
    const value={error_code:error.code,message:sanitize(error.message),retryable:error.retryable}
    if(json) console.log(JSON.stringify({type:"error",data:value}));else console.error(JSON.stringify(value,null,2))
    return error.exitCode
  } finally {process.off("SIGINT",interrupt);process.off("SIGTERM",terminate)}
}
if(import.meta.main) process.exitCode=await main()
