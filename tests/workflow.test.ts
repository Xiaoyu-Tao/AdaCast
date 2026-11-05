import { test,expect } from "bun:test"
import path from "node:path"
import { readFile,writeFile } from "node:fs/promises"
import { fixture,mockProvider } from "./helpers"
import { runTask } from "../src/workflow"
import { AppError } from "../src/contracts"
import { Store,lockWorkspace } from "../src/storage"
import { exportDelivery,verifyBundle } from "../src/delivery"

test("local task trains, selects on validation, reuses weights, resumes and exports a verified package",async()=>{
  const f=await fixture()
  try {
    const result=await runTask({configFile:f.configFile,workspace:f.workspace,signal:new AbortController().signal})
    expect(result.status).toBe("succeeded")
    const root=path.join(f.workspace,".forecast"),store=new Store(root,true)
    const selected=store.get<any>("selected"),testRun=store.get<any>("test"),usage=store.usage() as any
    expect(usage.calls).toBe(0)
    expect(testRun.source_run_id).toBe(selected.run_id)
    expect(testRun.artifact_sha256).toBe(selected.artifact_sha256)
    expect(store.get<any[]>("runs")!.filter(r=>r.split==="val")).toHaveLength(2)
    store.close()
    const resumed=await runTask({workspace:f.workspace,resume:true,signal:new AbortController().signal})
    expect(resumed.status).toBe("succeeded")
    const out=path.join(f.dir,"delivery")
    await exportDelivery(f.workspace,out)
    const manifest=await verifyBundle(out)
    expect(manifest.selection_basis).toBe("raw_test_mse_among_validation_family_representatives")
    await writeFile(path.join(out,"predictions/final.csv"),"tampered")
    await expect(verifyBundle(out)).rejects.toThrow("hash mismatch")
  } finally {await f.cleanup()}
})

test("interrupted phase recovery does not repeat completed experiments",async()=>{
  const f=await fixture()
  try {
    await expect(runTask({configFile:f.configFile,workspace:f.workspace,signal:new AbortController().signal,emit(type,value:any){
      if(type==="phase.changed"&&value.phase==="post-forecast") throw new AppError("INTERRUPTED","Injected boundary interruption",130,true)
    }})).rejects.toThrow("Injected boundary")
    const store=new Store(path.join(f.workspace,".forecast"),true)
    const before=store.get<any[]>("runs")!.filter(r=>r.split==="val").map(r=>r.run_id)
    expect(store.get<any>("state").lifecycle).toBe("interrupted");store.close()
    await runTask({workspace:f.workspace,resume:true,signal:new AbortController().signal})
    const after=new Store(path.join(f.workspace,".forecast"),true)
    expect(after.get<any[]>("runs")!.filter(r=>r.split==="val").map(r=>r.run_id)).toEqual(before)
    after.close()
  } finally {await f.cleanup()}
})

test("frozen input corruption is rejected without silently changing the task",async()=>{
  const f=await fixture()
  try {
    await expect(runTask({configFile:f.configFile,workspace:f.workspace,signal:new AbortController().signal,emit(type){
      if(type==="phase.changed") throw new AppError("INTERRUPTED","pause",130)
    }})).rejects.toThrow("pause")
    await writeFile(path.join(f.workspace,".forecast/inputs/dataset.csv"),"changed")
    await expect(runTask({workspace:f.workspace,resume:true,signal:new AbortController().signal})).rejects.toThrow("Frozen input changed")
  } finally {await f.cleanup()}
})

test("one workspace rejects concurrent executors using a transactional lease",async()=>{
  const f=await fixture()
  try {
    const unlock=await lockWorkspace(f.workspace)
    try {await expect(lockWorkspace(f.workspace)).rejects.toThrow("Another executor")}
    finally {await unlock()}
    const next=await lockWorkspace(f.workspace);await next()
  } finally {await f.cleanup()}
})

test("mock-provider agents execute real tools and the original structured evidence loop",async()=>{
  const provider=mockProvider({evidence:true,strategy:"off"})
  process.env.ADACAST_LITE_TEST_KEY="integration-test-credential"
  const f=await fixture({task:{reflection_enabled:true,model_families:["Naive"]},execution:{mode:"agent",window_llm_concurrency:2},
    llm:{base_url:provider.baseURL,model:"mock-model",api_key_env:"ADACAST_LITE_TEST_KEY",max_retries:0},
    decisions:{adjustment:{strategy:"off"}}})
  try {
    await runTask({configFile:f.configFile,workspace:f.workspace,signal:new AbortController().signal})
    const store=new Store(path.join(f.workspace,".forecast"),true)
    expect(store.get<any>("domain-state").reflection.configured).toBe(true)
    expect(store.get<any>("validation_reflection").completed_windows).toBe(3)
    expect(store.get<any>("test_reflection").completed_windows).toBe(6)
    expect(store.get<any>("state").lifecycle).toBe("succeeded")
    expect((store.usage() as any).calls).toBe(provider.requests.length)
    store.close()
    const isolated=provider.requests.filter(r=>!r.messages[0].content.includes("FROZEN TASK AND CURRENT STATE:"))
    expect(isolated.length).toBeGreaterThan(6)
    for(const request of isolated) {
      const input=request.messages[1].content
      expect(input).not.toContain('"actual"')
      expect(input).not.toContain('"actual_original"')
    }
    const report=await readFile(path.join(f.workspace,".forecast/reports/final-report.md"),"utf8")
    expect(report).toContain("mock-provider integration test")
  } finally {provider.close();await f.cleanup();delete process.env.ADACAST_LITE_TEST_KEY}
})

test("window cancellation remains resumable and terminal windows are reused",async()=>{
  const controller=new AbortController()
  let gaps=0,interrupted=false
  const provider=mockProvider({onGap(){if(++gaps===5&&!interrupted){interrupted=true;controller.abort(new AppError("INTERRUPTED","Injected window interruption",130,true))}}})
  process.env.ADACAST_LITE_TEST_KEY="integration-test-credential"
  const f=await fixture({task:{reflection_enabled:true,model_families:["Naive"]},execution:{mode:"agent",window_llm_concurrency:1},
    llm:{base_url:provider.baseURL,model:"mock-model",api_key_env:"ADACAST_LITE_TEST_KEY",max_retries:0}})
  try {
    await expect(runTask({configFile:f.configFile,workspace:f.workspace,signal:controller.signal})).rejects.toThrow("Injected window")
    await runTask({workspace:f.workspace,resume:true,signal:new AbortController().signal})
    const store=new Store(path.join(f.workspace,".forecast"),true)
    expect(store.get<any>("state").lifecycle).toBe("succeeded")
    expect(store.get<any>("test_reflection").completed_windows).toBe(6)
    // 3 validation + 6 test windows, plus the interrupted request only.
    expect(gaps).toBeGreaterThanOrEqual(6)
    store.close()
  } finally {provider.close();await f.cleanup();delete process.env.ADACAST_LITE_TEST_KEY}
})

test("UserPrior skips model search and preserves an independently supplied forecast source",async()=>{
  const f=await fixture({task:{model_families:["Naive"]}})
  try {
    const rows=(await readFile(f.data,"utf8")).trim().split("\n")
    const prior=path.join(f.dir,"prior.csv")
    for(let i=217;i<rows.length;i++) {
      const [time,target]=rows[i].split(",");rows[i]=time+","+(Number(target)+0.5)
    }
    await writeFile(prior,rows.join("\n")+"\n")
    const config={...f.config,decisions:{...f.config.decisions,prediction_prior:{mode:"csv",path:prior}}}
    await writeFile(f.configFile,JSON.stringify(config))
    await runTask({configFile:f.configFile,workspace:f.workspace,signal:new AbortController().signal})
    const store=new Store(path.join(f.workspace,".forecast"),true)
    expect(store.get<any>("selected").model).toBe("UserPrior")
    expect(store.get<any[]>("attempts")).toHaveLength(1)
    store.close()
    const bundle=await verifyBundle(path.join(f.workspace,".forecast/best-model"))
    expect(bundle.files["model/prior.csv"]).toBeDefined()
  } finally {await f.cleanup()}
})

test("invalid evidence proposals never replace committed predictions",async()=>{
  const mockOptions={evidence:true,invalidProposal:true}
  const provider=mockProvider(mockOptions)
  process.env.ADACAST_LITE_TEST_KEY="integration-test-credential"
  const f=await fixture({task:{reflection_enabled:true,model_families:["Naive"]},execution:{mode:"agent",window_llm_concurrency:2},
    llm:{base_url:provider.baseURL,model:"mock-model",api_key_env:"ADACAST_LITE_TEST_KEY",max_retries:0}})
  try {
    await expect(runTask({configFile:f.configFile,workspace:f.workspace,signal:new AbortController().signal})).rejects.toThrow("did not complete")
    const interrupted=new Store(path.join(f.workspace,".forecast"),true)
    expect(interrupted.get("final_output")).toBeUndefined()
    const originalRuns=interrupted.get<any[]>("runs")!.map(r=>r.artifact_sha256)
    interrupted.close()
    mockOptions.invalidProposal=false
    await runTask({workspace:f.workspace,resume:true,signal:new AbortController().signal})
    const store=new Store(path.join(f.workspace,".forecast"),true)
    for(const metric of ["mse","mae","wape","mase"])expect(store.get<any>("final_output").metrics[metric]).toBeCloseTo(store.get<any>("test").metrics[metric],7)
    const records=store.get<any>("domain-state").reflection.test_records
    expect(records).toHaveLength(1)
    expect(store.get<any[]>("runs")!.filter(r=>r.split==="val").map(r=>r.artifact_sha256)).toEqual(originalRuns)
    store.close()
  } finally {provider.close();await f.cleanup();delete process.env.ADACAST_LITE_TEST_KEY}
})
