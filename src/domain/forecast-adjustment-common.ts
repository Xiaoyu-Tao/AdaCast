import { mkdir, readFile, writeFile } from "node:fs/promises"
import path from "node:path"
import matter from "gray-matter"
import { domainScope, executeEngineModule, Instance } from "./runtime"
import { Filesystem } from "./runtime"
import { ForecastPhaseState } from "./types"
import type { ForecastPhaseState as ForecastPhaseStateValue } from "./types"

export function forecastRoot(): string {
  return path.join(Instance.directory, ".forecast")
}

export function adjustmentRoot(): string {
  return path.join(forecastRoot(), "adjustments")
}

export function projectPath(...parts: string[]): string {
  return path.join(Instance.directory, ...parts)
}

export function projectRelative(file: string): string {
  return path.relative(Instance.directory, file).replaceAll(path.sep, path.posix.sep)
}

export function forecastPath(...parts: string[]): string {
  return path.posix.join(".forecast", ...parts)
}

export function runIdIsSafe(runId: string): boolean {
  return /^[A-Za-z0-9_-]+$/.test(runId)
}

export async function ensureAdjustmentDirs(): Promise<void> {
  const root = adjustmentRoot()
  await Promise.all([
    mkdir(root, { recursive: true }),
    mkdir(path.join(root, "candidates"), { recursive: true }),
    mkdir(path.join(root, "skills", "active"), { recursive: true }),
    mkdir(path.join(root, "skills", "versions"), { recursive: true }),
    mkdir(path.join(root, "validation"), { recursive: true }),
    mkdir(path.join(root, "test"), { recursive: true }),
    mkdir(path.join(root, "fit-runs"), { recursive: true }),
  ])
}

export async function readPhaseState(): Promise<ForecastPhaseStateValue> {
  const {store,config}=domainScope()
  const state=store.get<any>("domain-state") ?? {skills_confirmed:false,skills_confirmed_at:null,pre_forecast_completed:false,pre_forecast_completed_at:null,task_frozen_at:null,transitions:[],reflection:{enabled:config.task.reflection_enabled}}
  return ForecastPhaseState.parse({...state,phase:store.get<any>("state")?.phase??"init"})
}

export async function writePhaseState(state: ForecastPhaseStateValue, logEntry?: string): Promise<void> {
  const {store}=domainScope()
  store.set("domain-state",ForecastPhaseState.parse(state))
  if(logEntry)store.event("domain.state",{entry:logEntry})
  await mkdir(forecastRoot(),{recursive:true})
  await writeFile(path.join(forecastRoot(),"STATE.md"),matter.stringify("\n# Forecast State\n",state),"utf8")
}

export async function readJsonFile<T>(file: string): Promise<T> {
  const raw = await readFile(file, "utf-8")
  return JSON.parse(raw.replace(/^\uFEFF/, "")) as T
}

export async function writeJsonFile(file: string, value: unknown): Promise<void> {
  await mkdir(path.dirname(file), { recursive: true })
  await Filesystem.writeJson(file, value)
}

export const FORECAST_PYTHON_TIMEOUT_MS = 5 * 60 * 1000
const FORECAST_PYTHON_TIMEOUT_MIN_MS = 1_000
const FORECAST_PYTHON_TIMEOUT_MAX_MS = 30 * 60 * 1000

/** @internal Exported for deterministic timeout configuration tests. */
export function resolveForecastPythonTimeoutMs(value = process.env.ADACAST_LITE_FORECAST_PYTHON_TIMEOUT_MS): number {
  if (value === undefined || value.trim() === "") return FORECAST_PYTHON_TIMEOUT_MS
  const parsed = Number(value)
  if (
    !Number.isInteger(parsed) ||
    parsed < FORECAST_PYTHON_TIMEOUT_MIN_MS ||
    parsed > FORECAST_PYTHON_TIMEOUT_MAX_MS
  ) {
    throw new Error(
      `ADACAST_LITE_FORECAST_PYTHON_TIMEOUT_MS must be an integer between ${FORECAST_PYTHON_TIMEOUT_MIN_MS} and ${FORECAST_PYTHON_TIMEOUT_MAX_MS}`,
    )
  }
  return parsed
}

export async function runAdjustmentPython(module: string, args: string[]): Promise<unknown> {
  return executeEngineModule(module,args,resolveForecastPythonTimeoutMs())
}
