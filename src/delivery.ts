import path from "node:path"
import { cp, mkdir, readdir, readFile, rename, rm } from "node:fs/promises"
import { AppError, VERSION, type Config, type Experiment, type RunState } from "./contracts"
import type { Store } from "./storage"
import { compute } from "./python"
import { atomicWrite, digest, exists, id, inside, readJSON, writeJSON } from "./util"

async function files(root:string,prefix=""):Promise<string[]> {
  const result:string[]=[]
  for(const entry of await readdir(path.join(root,prefix),{withFileTypes:true})) {
    if(entry.isSymbolicLink()) throw new AppError("DELIVERY_INVALID","Delivery cannot contain symlinks",3)
    const rel=path.posix.join(prefix,entry.name)
    if(entry.isDirectory()) result.push(...await files(root,rel));else result.push(rel)
  }
  return result.sort()
}
export async function verifyBundle(root:string) {
  const manifest=await readJSON(path.join(root,"manifest.json"))
  if(manifest.schema_version!==1 || !manifest.files || !manifest.final_prediction) throw new AppError("DELIVERY_INVALID","Invalid delivery manifest",3)
  const required=["model/model.json","predictions/raw.csv","report.md","USAGE.md",manifest.final_prediction]
  const actual=await files(root)
  if(JSON.stringify(actual.filter(f=>f!=="manifest.json"))!==JSON.stringify(Object.keys(manifest.files).sort()))
    throw new AppError("DELIVERY_INVALID","Delivery file inventory differs from manifest",3)
  if(manifest.model_artifact_sha256!==manifest.files["model/model.json"])
    throw new AppError("DELIVERY_INVALID","Model provenance differs from indexed artifact",3)
  for(const rel of required) if(!Object.hasOwn(manifest.files,rel)) throw new AppError("DELIVERY_INVALID","Required delivery artifact is not indexed: "+rel,3)
  for(const [rel,hash] of Object.entries(manifest.files)) {
    const file=path.resolve(root,rel)
    if(!inside(root,file) || digest(await readFile(file))!==hash) throw new AppError("DELIVERY_INVALID","Delivery hash mismatch: "+rel,3)
  }
  return manifest
}

export async function buildDelivery(config:Config,store:Store,signal:AbortSignal) {
  const selected=store.get<Experiment>("selected"),test=store.get<Experiment>("test"),final=store.get<any>("final_output")
  if(!selected || !test || !final || !store.get("final_report")) throw new AppError("DELIVERY_INCOMPLETE","Required evaluation or report missing",3)
  if(config.task.reflection_enabled && !store.get("test_reflection")) throw new AppError("DELIVERY_INCOMPLETE","Window work is incomplete",3)
  if(digest(await readFile(selected.artifact))!==selected.artifact_sha256) throw new AppError("INTEGRITY_FAILED","Selected model changed",3)
  const report=await readFile(path.join(store.root,"reports/final-report.md"))
  if(digest(report)!==store.get<any>("final_report").sha256) throw new AppError("INTEGRITY_FAILED","Final report changed after commit",3)
  const staging=path.join(store.root,".delivery-"+id("tmp")),target=path.join(store.root,"best-model")
  await mkdir(path.join(staging,"predictions"),{recursive:true})
  try {
    await cp(path.dirname(selected.artifact),path.join(staging,"model"),{recursive:true})
    await cp(test.predictions,path.join(staging,"predictions/raw.csv"))
    await cp(final.predictions,path.join(staging,"predictions/final.csv"))
    const adjusted=store.get<any>("adjusted_test")
    if(adjusted) await cp(adjusted.predictions,path.join(staging,"predictions/adjusted.csv"))
    const adjustment=store.get<any>("adjustment")
    if(adjustment) await writeJSON(path.join(staging,"adjustment.json"),{id:adjustment.id,rule:adjustment.rule,validation_metrics:adjustment.metrics})
    await atomicWrite(path.join(staging,"report.md"),report)
    for(const [directory,destination] of [["reflection","audit/reflection"],["adjustments","audit/adjustments"],["runtime/llm-calls","audit/llm-calls"],["runtime/context-snapshots","audit/context-snapshots"]])
      if(await exists(path.join(store.root,directory))) await cp(path.join(store.root,directory),path.join(staging,destination),{recursive:true})
    await writeJSON(path.join(staging,"audit/decisions.json"),{reflections:store.get("reflections"),adjustment:store.get("adjustment_decision"),
      domain:store.get("domain-state"),family_representatives:store.get("family_representatives"),test_collection:store.get("test_collection"),validation_reflection:store.get("validation_reflection"),test_reflection:store.get("test_reflection"),usage:store.usage()})
    await compute(config,{operation:"visualize",predictions:final.predictions,output:path.join(staging,"forecast.svg")},store.root,signal)
    await writeJSON(path.join(staging,"task.json"),{...config.task,dataset_path:"EXTERNAL_INPUT_REQUIRED",dataset_sha256:store.get<any>("data_info")?.dataset_sha256})
    await atomicWrite(path.join(staging,"USAGE.md"),`# Forecast delivery\n\nModel: ${selected.model}. Execution: ${config.execution.mode}.\n\nThe manifest indexes all files by relative path and SHA-256. Predictions contain standardized and original-unit values with window and horizon coordinates. The final output kind is ${final.kind}.\n\nValidation MSE selected one representative per model family. Raw test MSE selected the delivery model among these representatives, using the exact fitted artifacts. model/model.json records training scale, parameters and model file hashes. UserPrior artifacts contain supplied forecasts rather than fitted weights.\n\nInstall adacast-lite-ml with the appropriate model extra to load the model. Use its predict operation with model/model.json and original-unit history; source input data is not embedded in the package. Foundation models require the recorded immutable revision and its weights.\n`)
    const hashes:Record<string,string>={}
    for(const rel of await files(staging)) hashes[rel]=digest(await readFile(path.join(staging,rel)))
    const manifest={schema_version:1,version:VERSION,task_id:store.get<RunState>("state")!.taskId,
      execution_mode:config.execution.mode,model:selected.model,selection_basis:"raw_test_mse_among_validation_family_representatives",
      family_representatives:(store.get<Experiment[]>("family_representatives")??[]).map(r=>({run_id:r.run_id,model:r.model,metrics:r.metrics})),raw_test_collection:(store.get<Experiment[]>("test_collection")??[]).map(r=>({run_id:r.run_id,model:r.model,source_run_id:r.source_run_id,metrics:r.metrics})),validation:{run_id:selected.run_id,metrics:selected.metrics},test:{run_id:test.run_id,metrics:test.metrics},
      final_metrics:final.metrics,final_kind:final.kind,final_prediction:"predictions/final.csv",
      model_artifact_sha256:selected.artifact_sha256,prediction_outputs:{raw:"predictions/raw.csv",final:"predictions/final.csv",
        ...(adjusted?{adjusted:"predictions/adjusted.csv"}:{})},files:hashes}
    await writeJSON(path.join(staging,"manifest.json"),manifest)
    await verifyBundle(staging)
    if(await exists(target)) {
      const backup=target+".previous-"+id("tmp")
      await rename(target,backup)
      try {await rename(staging,target)} catch(e) {await rename(backup,target);throw e}
      await rm(backup,{recursive:true,force:true})
    } else await rename(staging,target)
    store.set("delivery",{manifest_sha256:digest(await readFile(path.join(target,"manifest.json"))),path:target})
    return manifest
  } finally {await rm(staging,{recursive:true,force:true})}
}

export async function exportDelivery(workspace:string,output:string) {
  const source=path.join(workspace,".forecast/best-model")
  const manifest=await verifyBundle(source)
  if(inside(workspace,output)) throw new AppError("OUTPUT_INVALID","Export destination must be outside the task workspace",3)
  if(await exists(output)) throw new AppError("OUTPUT_EXISTS","Export destination already exists",3)
  const staging=output+".tmp-"+id("export")
  await mkdir(path.dirname(output),{recursive:true})
  try {await cp(source,staging,{recursive:true});await verifyBundle(staging);await rename(staging,output)}
  finally {await rm(staging,{recursive:true,force:true})}
  return {output,model:manifest.model,files:Object.keys(manifest.files).length}
}
