import { mkdir, readFile, writeFile } from "fs/promises"
import path from "path"
import matter from "gray-matter"
import { Instance } from "@/project/instance"
import { Filesystem } from "@/util/filesystem"
import { resolveForecastPythonProject } from "./python-project"
import { normalizeForecastStateFrontmatter } from "./forecast-progress"
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
  const statePath = path.join(forecastRoot(), "STATE.md")
  try {
    const parsed = matter(await readFile(statePath, "utf-8"))
    return ForecastPhaseState.parse(normalizeForecastStateFrontmatter(parsed.data as Record<string, unknown>))
  } catch {
    return ForecastPhaseState.parse({
      phase: "init",
      skills_confirmed: false,
      skills_confirmed_at: null,
      pre_forecast_completed: false,
      pre_forecast_completed_at: null,
      task_frozen_at: null,
      transitions: [],
    })
  }
}

export async function writePhaseState(state: ForecastPhaseStateValue, logEntry?: string): Promise<void> {
  const statePath = path.join(forecastRoot(), "STATE.md")
  let body = "\n# Forecast State Log\n"
  try {
    const existing = matter(await readFile(statePath, "utf-8"))
    body = existing.content || body
  } catch {
    // STATE.md will be created if missing.
  }
  if (logEntry) body += `\n${logEntry}\n`
  await writeFile(statePath, matter.stringify(body, state), "utf-8")
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
export function resolveForecastPythonTimeoutMs(value = process.env.ADACAST_FORECAST_PYTHON_TIMEOUT_MS): number {
  if (value === undefined || value.trim() === "") return FORECAST_PYTHON_TIMEOUT_MS
  const parsed = Number(value)
  if (
    !Number.isInteger(parsed) ||
    parsed < FORECAST_PYTHON_TIMEOUT_MIN_MS ||
    parsed > FORECAST_PYTHON_TIMEOUT_MAX_MS
  ) {
    throw new Error(
      `ADACAST_FORECAST_PYTHON_TIMEOUT_MS must be an integer between ${FORECAST_PYTHON_TIMEOUT_MIN_MS} and ${FORECAST_PYTHON_TIMEOUT_MAX_MS}`,
    )
  }
  return parsed
}

export async function runAdjustmentPython(module: string, args: string[]): Promise<unknown> {
  const pythonDir = await resolveForecastPythonProject(Instance.directory)
  const proc = Bun.spawn(["uv", "run", "--project", pythonDir, "python", "-m", module, ...args], {
    cwd: Instance.directory,
    env: {
      ...process.env,
      PYTHONUTF8: "1",
      PYTHONIOENCODING: "utf-8",
    },
    stdout: "pipe",
    stderr: "pipe",
  })
  const timeoutMs = resolveForecastPythonTimeoutMs()
  let timeout: ReturnType<typeof setTimeout> | undefined
  let exited = false
  const exitPromise = proc.exited.then((exitCode) => {
    exited = true
    return exitCode
  })
  const completion = Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    exitPromise,
  ])
  const timeoutPromise = new Promise<never>((_, reject) => {
    timeout = setTimeout(() => {
      try {
        proc.kill("SIGTERM")
      } catch {
        // The process may have exited between the timeout and signal delivery.
      }
      setTimeout(() => {
        if (!exited) {
          try {
            proc.kill("SIGKILL")
          } catch {
            // The process may have exited after SIGTERM.
          }
        }
      }, 2_000)
      reject(new Error(`${module} timed out after ${timeoutMs}ms`))
    }, timeoutMs)
  })
  const [stdout, stderr, exitCode] = await Promise.race([completion, timeoutPromise]).finally(() => {
    if (timeout) clearTimeout(timeout)
  })
  if (exitCode !== 0) {
    throw new Error(
      `${module} failed with exit code ${exitCode}: ${stderr
        .split("\n")
        .filter(Boolean)
        .slice(-10)
        .join("\n")}`,
    )
  }
  return JSON.parse(stdout)
}
