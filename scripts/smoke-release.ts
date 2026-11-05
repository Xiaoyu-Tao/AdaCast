import { mockProvider } from "../tests/helpers"
import path from "node:path"
import os from "node:os"
import { mkdtemp,mkdir,copyFile,readFile,writeFile,rm } from "node:fs/promises"
const root=path.resolve(import.meta.dir,"..")
const temporary=await mkdtemp(path.join(os.tmpdir(),"adacast-lite-release-"))
async function command(args:string[],json=false) {
  const env:NodeJS.ProcessEnv={...process.env,UV_LINK_MODE:"copy"}
  for(const key of ["ADACAST_LITE_HOME","ADACAST_LITE_PYTHON","PYTHONPATH","PYTHONHOME"]) delete env[key]
  const child=Bun.spawn(args,{cwd:temporary,stdout:"pipe",stderr:"pipe",env})
  const [stdout,stderr,code]=await Promise.all([new Response(child.stdout).text(),new Response(child.stderr).text(),child.exited])
  if(code!==0) throw new Error(`${args[0]} exited ${code}: ${stderr}\n${stdout}`)
  return json?stdout.trim().split("\n").map(line=>JSON.parse(line)).at(-1).data:stdout
}
try {
  const prefix=path.join(temporary,"installed release")
  await command(["sh",path.join(root,"dist/install.sh"),prefix])
  const input=path.join(temporary,"input")
  await mkdir(input)
  await copyFile(path.join(root,"examples/series.csv"),path.join(input,"series.csv"))
  await copyFile(path.join(root,"examples/task.local.json"),path.join(input,"task.json"))
  const cli=path.join(prefix,"bin/adacast-lite"),workspace=path.join(temporary,"workspace")
  const catalog=await command([cli,"models","--json"],true)
  if(!catalog.models.some((m:any)=>m.name==="Holt"&&m.dependencies_available))throw new Error("Installed model registry failed")
  const validated=await command([cli,"validate","--config",path.join(input,"task.json"),"--json"],true)
  if(!validated.valid) throw new Error("Installed CLI could not validate input")
  const run=await command([cli,"run","--config",path.join(input,"task.json"),"--workspace",workspace,"--json"],true)
  const resumed=await command([cli,"resume","--workspace",workspace,"--json"],true)
  if(run.status!=="succeeded" || resumed.status!=="succeeded") throw new Error("Installed run/recovery failed")
  const exported=await command([cli,"export","--workspace",workspace,"--output",path.join(temporary,"delivery"),"--json"],true)
  const manifest=JSON.parse(await readFile(path.join(exported.output,"manifest.json"),"utf8"))
  const request=path.join(temporary,"predict.json")
  await writeFile(request,JSON.stringify({protocol_version:1,operation_id:"release-inference",operation:"predict",
    model_artifact:path.join(exported.output,"model/model.json"),history:Array.from({length:24},(_,i)=>20+i/10)}))
  const python=path.join(prefix,"python/.venv/bin/python")
  const predicted=JSON.parse(await command([python,"-I","-m","adacast_lite_ml.cli","--request",request]) as string)
  if(!predicted.ok || predicted.result.values.length!==12) throw new Error("Portable fitted model inference failed")
  const provider=mockProvider({strategy:"off"})
  const releaseKey="ADACAST_LITE_RELEASE_TEST_KEY"
  process.env[releaseKey]="release-protocol-fixture"
  try {
    const agentConfig=JSON.parse(await readFile(path.join(input,"task.json"),"utf8"))
    agentConfig.task.reflection_enabled=true
    agentConfig.task.model_families=["Naive"]
    agentConfig.execution={...agentConfig.execution,mode:"agent"}
    agentConfig.llm={base_url:provider.baseURL,model:"mock-model",api_key_env:releaseKey,max_retries:0}
    agentConfig.decisions={adjustment:{strategy:"off"}}
    const agentFile=path.join(input,"task.agent.json")
    await writeFile(agentFile,JSON.stringify(agentConfig))
    const agent=await command([cli,"run","--config",agentFile,"--workspace",path.join(temporary,"agent-workspace"),"--json"],true)
    if(agent.status!=="succeeded" || agent.manifest.final_kind!=="window_refined" || !provider.requests.length)throw new Error("Installed agent/window protocol failed")
  }finally{provider.close();delete process.env[releaseKey]}
  const result={installed_without_source:true,run:true,resume:true,export:true,portable_model_inference:true,mock_agent_window_protocol:true,model_registry:true,model:manifest.model,files:exported.files}
  console.log(JSON.stringify(result,null,2))
} finally {await rm(temporary,{recursive:true,force:true})}
