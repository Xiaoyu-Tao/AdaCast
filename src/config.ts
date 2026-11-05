import { resolveLLMPolicy } from "./llm-policy"
import path from "node:path"
import { readFile } from "node:fs/promises"
import { AppError, ConfigSchema, type Config } from "./contracts"
import { errorText } from "./util"

export async function loadConfig(file: string): Promise<Config> {
  try {
    const config = ConfigSchema.parse(JSON.parse(await readFile(file, "utf8")))
    const base = path.dirname(path.resolve(file))
    config.task.dataset_path = path.resolve(base, config.task.dataset_path)
    for (const group of [config.knowledge, config.instructions]) group.files = group.files.map(p => path.resolve(base, p))
    config.memory.directory = path.resolve(base, config.memory.directory)
    config.tools.extra_read_paths = config.tools.extra_read_paths.map(p => path.resolve(base, p))
    config.tools.extra_write_paths = config.tools.extra_write_paths.map(p => path.resolve(base, p))
    if (config.decisions.prediction_prior.mode === "csv") config.decisions.prediction_prior.path = path.resolve(base, config.decisions.prediction_prior.path)
    if (config.llm) {
      config.llm.model = config.llm.model ?? process.env[config.llm.model_env]
      config.llm.base_url = config.llm.base_url ?? process.env[config.llm.base_url_env]
    }
    return config
  } catch (e) { throw new AppError("CONFIG_INVALID", errorText(e), 2) }
}
export function validateCredentials(config: Config) {
  if (config.execution.mode !== "agent") return
  const llm = config.llm!
  if (!llm.model || !llm.base_url) throw new AppError("LLM_CONFIG_MISSING", "Configure the model and base URL, directly or through the configured environment variables", 2)
  try {
    const url = new URL(llm.base_url)
    if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash)
      throw new Error("Expected HTTP(S) URL without credentials, query, or fragment")
  } catch { throw new AppError("LLM_URL_INVALID", "Invalid LLM base URL", 2) }
  for(const role of ["planner","forecaster","critic"]){const policy=resolveLLMPolicy(config,role);if(!policy.supportsTools)throw new AppError("LLM_TOOLS_UNSUPPORTED",`Model ${policy.model} cannot execute required ${role} tools`,2)}
  if (!process.env[llm.api_key_env]) throw new AppError("LLM_KEY_MISSING", `Set ${llm.api_key_env}`, 2)
}
