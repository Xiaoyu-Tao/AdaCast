import { mkdirSync, rmSync, writeFileSync } from "fs"
import path from "path"

export const STANDARD_HF_CACHE_ENV_KEYS = [
  "HF_HOME",
  "HF_HUB_CACHE",
  "HF_XET_CACHE",
  "HF_ASSETS_CACHE",
  "TRANSFORMERS_CACHE",
  "HUGGINGFACE_HUB_CACHE",
  "PYTORCH_TRANSFORMERS_CACHE",
  "PYTORCH_PRETRAINED_BERT_CACHE",
]

type ForecastModelEnvironmentSource = "operator" | "project_default" | "hf_default"

export type ForecastModelEnvironment = {
  env: Record<string, string>
  cacheRoot: string | null
  hubCache: string | null
  xetCache: string | null
  assetsCache: string | null
  transformersCache: string | null
  endpoint: string | null
  source: ForecastModelEnvironmentSource
  warnings: Array<{ code: string; message: string }>
}

const PROJECT_CACHE_WARNING = {
  code: "project_cache_unwritable",
  message: "Project model cache .forecast/model-cache is not writable; using Hugging Face defaults.",
}

function present(env: NodeJS.ProcessEnv | undefined, key: string): string | undefined {
  const value = env?.[key]
  return typeof value === "string" && value.length > 0 ? value : undefined
}

function hasOperatorCacheEnv(env: NodeJS.ProcessEnv | undefined): boolean {
  return STANDARD_HF_CACHE_ENV_KEYS.some((key) => present(env, key) !== undefined)
}

function copyPresent(env: NodeJS.ProcessEnv | undefined, output: Record<string, string>, key: string) {
  const value = present(env, key)
  if (value !== undefined) output[key] = value
}

function cacheRootWritable(cacheRoot: string): boolean {
  try {
    mkdirSync(cacheRoot, { recursive: true })
    const probePath = path.join(cacheRoot, `.adacast-probe-${process.pid}-${Date.now()}`)
    writeFileSync(probePath, "ok")
    rmSync(probePath, { force: true })
    return true
  } catch {
    return false
  }
}

function operatorEnvironment(env: NodeJS.ProcessEnv | undefined, endpoint: string | null): ForecastModelEnvironment {
  const output: Record<string, string> = {}
  if (endpoint) output.HF_ENDPOINT = endpoint
  for (const key of STANDARD_HF_CACHE_ENV_KEYS) copyPresent(env, output, key)

  const transformersCache = present(env, "TRANSFORMERS_CACHE") ?? null
  if (transformersCache && !present(env, "HF_HUB_CACHE")) {
    output.HF_HUB_CACHE = transformersCache
  }

  return {
    env: output,
    cacheRoot: present(env, "HF_HOME") ?? null,
    hubCache: output.HF_HUB_CACHE ?? present(env, "HUGGINGFACE_HUB_CACHE") ?? null,
    xetCache: output.HF_XET_CACHE ?? null,
    assetsCache: output.HF_ASSETS_CACHE ?? null,
    transformersCache,
    endpoint,
    source: "operator",
    warnings: [],
  }
}

export function resolveForecastModelEnvironment(input: {
  directory: string
  env?: NodeJS.ProcessEnv
}): ForecastModelEnvironment {
  const endpoint = present(input.env, "HF_ENDPOINT") ?? null

  if (hasOperatorCacheEnv(input.env)) {
    return operatorEnvironment(input.env, endpoint)
  }

  const output: Record<string, string> = {}
  if (endpoint) output.HF_ENDPOINT = endpoint

  const cacheRoot = path.join(input.directory, ".forecast", "model-cache")
  if (!cacheRootWritable(cacheRoot)) {
    return {
      env: output,
      cacheRoot: null,
      hubCache: null,
      xetCache: null,
      assetsCache: null,
      transformersCache: null,
      endpoint,
      source: "hf_default",
      warnings: [PROJECT_CACHE_WARNING],
    }
  }

  const hubCache = path.join(cacheRoot, "hub")
  const xetCache = path.join(cacheRoot, "xet")
  const assetsCache = path.join(cacheRoot, "assets")
  output.HF_HUB_CACHE = hubCache
  output.HF_XET_CACHE = xetCache
  output.HF_ASSETS_CACHE = assetsCache

  return {
    env: output,
    cacheRoot,
    hubCache,
    xetCache,
    assetsCache,
    transformersCache: null,
    endpoint,
    source: "project_default",
    warnings: [],
  }
}
