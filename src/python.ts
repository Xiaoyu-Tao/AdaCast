import { resolveForecastModelEnvironment } from "./domain/model-environment"
import path from "node:path"
import { spawn } from "node:child_process"
import { mkdir, mkdtemp, rm, readFile } from "node:fs/promises"
import os from "node:os"
import { z } from "zod"
import { AppError, PROTOCOL_VERSION, type Config } from "./contracts"
import { checkAbort, errorText, exists, id, sanitize, writeJSON, atomicWrite } from "./util"

const MetricsSchema=z.object({mse:z.number().finite(),mae:z.number().finite(),wape:z.number().finite().nullable(),mase:z.number().finite().nullable()})
const ResultSchemas:Record<string,z.ZodType>={
  engine_module:z.unknown(),
  models:z.object({models:z.array(z.object({name:z.string(),family:z.string(),dependencies_available:z.boolean(),missing_dependencies:z.array(z.string())}).passthrough()),availability_scope:z.string()}),
  inspect:z.object({rows:z.number().int().positive(),dataset_sha256:z.string().regex(/^[a-f0-9]{64}$/),
    boundaries:z.record(z.string(),z.tuple([z.number().int(),z.number().int()])),window_counts:z.record(z.string(),z.number().int().nonnegative()),
    train_statistics:z.object({mean:z.number().finite(),std:z.number().finite()}).passthrough()}).passthrough(),
  analyze:z.object({analysis_scope:z.object({target_scope:z.literal("training_split_only")}).passthrough(),train_statistics:z.object({mean:z.number().finite(),std:z.number().finite()}).passthrough()}).passthrough(),
  experiment:z.object({run_id:z.string(),model:z.string(),spec:z.record(z.string(),z.unknown()),split:z.string(),metrics:MetricsSchema,
    artifact:z.string(),artifact_sha256:z.string().regex(/^[a-f0-9]{64}$/),predictions:z.string(),status:z.literal("success")}).passthrough(),
  diagnostics:z.object({scope:z.literal("adjustment_fit_only"),mean_residual_original:z.number().finite(),suggested_offset:z.number().finite()}).passthrough(),
  adjust:z.object({metrics:MetricsSchema,predictions:z.string()}).passthrough(),
  policy:z.object({version:z.literal(1),min:z.number().finite(),max:z.number().finite(),max_abs_step:z.number().positive()}).passthrough(),
  blind_windows:z.array(z.object({window_index:z.number().int().nonnegative(),
    history:z.array(z.object({timestamp:z.string(),value:z.number().finite()})),
    points:z.array(z.object({timestamp:z.string(),horizon_index:z.number().int().nonnegative(),value:z.number().finite()})),forecast_gap:z.number().int().nonnegative()})),
  finalize_windows:z.object({metrics:MetricsSchema,predictions:z.string(),windows:z.number().int().nonnegative()}),
  visualize:z.object({file:z.string()})
}

export async function compute(config: Config, request: Record<string, unknown>, root: string | undefined, signal: AbortSignal): Promise<any> {
  checkAbort(signal)
  const temporary = !root
  const work = root ?? await mkdtemp(path.join(os.tmpdir(), "adacast-lite-check-"))
  const operationId = id("compute")
  const requestDir = path.join(work, "runtime/requests")
  await mkdir(requestDir, { recursive: true })
  const requestFile = path.join(requestDir, operationId + ".json")
  await writeJSON(requestFile, { protocol_version: PROTOCOL_VERSION, operation_id: operationId, task: config.task, execution_mode:config.execution.mode, ...request })
  const home = process.env.ADACAST_LITE_HOME ?? path.resolve(import.meta.dir, "..")
  const source = path.join(home, "python/src")
  const venv = path.join(home, "python/.venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python")
  const python = process.env.ADACAST_LITE_PYTHON ?? (await exists(venv) ? venv : "python3")
  const args = await exists(path.join(source, "adacast_lite_ml/cli.py"))
    ? ["-I", "-c", "import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('adacast_lite_ml.cli',run_name='__main__')", source, "--request", requestFile]
    : ["-I", "-m", "adacast_lite_ml.cli", "--request", requestFile]
  try {
    const { stdout, stderr, code } = await runProcess(python, args, config.execution.tool_timeout_seconds * 1000, signal,{cwd:root?path.dirname(root):undefined,env:root?{...process.env,...resolveForecastModelEnvironment({directory:path.dirname(root),env:process.env}).env}:process.env})
    if (root) await atomicWrite(path.join(work, "runtime/requests", operationId + ".log"), sanitize(stderr))
    let response: any
    try { response = JSON.parse(stdout) } catch { throw new AppError("IPC_INVALID", "Computation returned invalid JSON: " + sanitize(stderr.slice(-2000))) }
    if (response.protocol_version !== PROTOCOL_VERSION || response.operation_id !== operationId)
      throw new AppError("IPC_MISMATCH", "Computation protocol or operation identity mismatch")
    if (!response.ok || code !== 0) throw new AppError(response.error?.code ?? "COMPUTATION_FAILED", sanitize(response.error?.message ?? stderr.slice(-2000)),response.error?.code==="INPUT_INVALID"?2:4)
    const schema=ResultSchemas[String(request.operation)]
    if(!schema) throw new AppError("IPC_OPERATION_UNKNOWN","No response contract for computation operation",3)
    const parsed=schema.safeParse(response.result)
    if(!parsed.success) throw new AppError("IPC_INVALID","Computation result does not match operation schema: "+parsed.error.message)
    return parsed.data
  } finally { if (temporary) await rm(work, { recursive: true, force: true }) }
}

export async function runProcess(command: string, args: string[], timeoutMs: number, signal: AbortSignal,
  options: { cwd?: string; env?: NodeJS.ProcessEnv; maxBytes?: number } = {}) {
  checkAbort(signal)
  return await new Promise<{stdout:string;stderr:string;code:number}>((resolve, reject) => {
    let stdout = "", stderr = "", failure: AppError | undefined
    const child = spawn(command, args, { cwd: options.cwd, env: options.env ?? process.env,
      detached: process.platform !== "win32", stdio: ["ignore", "pipe", "pipe"] })
    let killTimer: ReturnType<typeof setTimeout> | undefined
    const kill = () => {
      try { if (process.platform !== "win32" && child.pid) process.kill(-child.pid, "SIGTERM"); else child.kill("SIGTERM") } catch {}
      killTimer = setTimeout(() => { try { if (process.platform !== "win32" && child.pid) process.kill(-child.pid, "SIGKILL"); else child.kill("SIGKILL") } catch {} }, 1000)
      killTimer.unref()
    }
    const cancel = () => { failure = signal.reason instanceof AppError ? signal.reason : new AppError("INTERRUPTED", "Process interrupted", 130, true); kill() }
    signal.addEventListener("abort", cancel, { once: true })
    const timer = setTimeout(() => { failure = new AppError("TOOL_TIMEOUT", "Computation timeout", 4, true); kill() }, timeoutMs)
    const collect = (chunk: Buffer, target: "out" | "err") => {
      if (target === "out") stdout += chunk.toString(); else stderr += chunk.toString()
      if (Buffer.byteLength(stdout) + Buffer.byteLength(stderr) > (options.maxBytes ?? 64 * 1024 * 1024)) {
        failure = new AppError("OUTPUT_TOO_LARGE", "Process output exceeds its limit"); kill()
      }
    }
    child.stdout.on("data", chunk => collect(chunk, "out")); child.stderr.on("data", chunk => collect(chunk, "err"))
    const cleanup = () => { clearTimeout(timer); if (killTimer) clearTimeout(killTimer); signal.removeEventListener("abort", cancel) }
    child.on("error", e => { cleanup(); reject(new AppError("PROCESS_START_FAILED", errorText(e), 2)) })
    child.on("close", code => { cleanup(); if (failure) reject(failure); else resolve({stdout,stderr,code:code ?? 1}) })
  })
}
