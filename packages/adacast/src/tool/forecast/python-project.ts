import { readdir } from "fs/promises"
import path from "path"
import { Filesystem } from "@/util/filesystem"

const PYTHON_PROJECT_MARKER = path.join("python", "pyproject.toml")
const PYPROJECT_FILENAME = "pyproject.toml"

type ResolveForecastPythonProjectOptions = {
  execPath?: string
  bundledSearchFrom?: string
  includeBundledFallback?: boolean
}

function normalizeProjectCandidate(candidate: string): string {
  return candidate.endsWith(PYPROJECT_FILENAME) ? path.dirname(candidate) : candidate
}

async function resolveFirstExistingProject(candidates: Array<string | undefined>): Promise<string | undefined> {
  for (const candidate of candidates) {
    if (!candidate) continue
    const normalized = normalizeProjectCandidate(path.resolve(candidate))
    if (await Filesystem.exists(path.join(normalized, PYPROJECT_FILENAME))) {
      return normalized
    }
  }
}

function installedRuntimeCandidates(execPath: string): string[] {
  const execDir = path.dirname(path.resolve(execPath))
  return Array.from(
    new Set([
      path.resolve(execDir, "python"),
      path.resolve(execDir, "..", "python"),
      path.resolve(execDir, "..", "share", "adacast", "python"),
    ]),
  )
}

function runtimeExecutableCandidates(explicitExecPath?: string): string[] {
  if (explicitExecPath) {
    return [explicitExecPath]
  }

  return Array.from(
    new Set(
      [process.env.ADACAST_BIN_PATH, process.execPath, process.argv0].filter((value): value is string => !!value),
    ),
  )
}

async function installedNodeModulesRuntimeCandidates(directory: string): Promise<string[]> {
  const results: string[] = []
  const seen = new Set<string>()
  let current = path.resolve(directory)

  for (;;) {
    const nodeModulesDir = path.join(current, "node_modules")
    if (await Filesystem.exists(nodeModulesDir)) {
      const entries = await readdir(nodeModulesDir, { withFileTypes: true }).catch(() => [])
      for (const entry of entries) {
        if (!entry.name.startsWith("adacast")) {
          continue
        }
        if (!entry.isDirectory() && !entry.isSymbolicLink()) {
          continue
        }

        const candidate = path.join(nodeModulesDir, entry.name, "bin", "python")
        if (seen.has(candidate)) {
          continue
        }

        seen.add(candidate)
        results.push(candidate)
      }
    }

    const parent = path.dirname(current)
    if (parent === current) {
      return results
    }

    current = parent
  }
}

export async function resolveForecastPythonProject(
  directory: string,
  options: ResolveForecastPythonProjectOptions = {},
): Promise<string> {
  const configured = process.env.ADACAST_PYTHON_PROJECT
  const installedNodeModulesCandidates = await installedNodeModulesRuntimeCandidates(directory)
  const resolved =
    (await resolveFirstExistingProject([
      configured,
      path.join(directory, "python"),
      ...runtimeExecutableCandidates(options.execPath).flatMap(installedRuntimeCandidates),
      ...installedNodeModulesCandidates,
    ])) ??
    undefined

  if (resolved) {
    return resolved
  }

  if (options.includeBundledFallback !== false) {
    const bundled = await Filesystem.findUp(PYTHON_PROJECT_MARKER, options.bundledSearchFrom ?? import.meta.dir)
    if (bundled[0]) {
      return path.dirname(bundled[0])
    }
  }

  if (configured) {
    throw new Error(
      `Forecast Python project not found at ADACAST_PYTHON_PROJECT=${configured}. Expected a directory containing pyproject.toml.`,
    )
  }

  throw new Error(
    "Forecast Python project not found. Expected python/pyproject.toml in the workspace, beside the installed executable, or under ../share/adacast/python.",
  )
}
