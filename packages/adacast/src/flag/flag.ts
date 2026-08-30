import { Config } from "effect"

function truthy(key: string) {
  const value = process.env[key]?.toLowerCase()
  return value === "true" || value === "1"
}

function falsy(key: string) {
  const value = process.env[key]?.toLowerCase()
  return value === "false" || value === "0"
}

export namespace Flag {
  export const ADACAST_AUTO_SHARE = truthy("ADACAST_AUTO_SHARE")
  export const ADACAST_GIT_BASH_PATH = process.env["ADACAST_GIT_BASH_PATH"]
  export const ADACAST_CONFIG = process.env["ADACAST_CONFIG"]
  export declare const ADACAST_TUI_CONFIG: string | undefined
  export declare const ADACAST_CONFIG_DIR: string | undefined
  export const ADACAST_CONFIG_CONTENT = process.env["ADACAST_CONFIG_CONTENT"]
  export const ADACAST_DISABLE_AUTOUPDATE = truthy("ADACAST_DISABLE_AUTOUPDATE")
  export const ADACAST_ALWAYS_NOTIFY_UPDATE = truthy("ADACAST_ALWAYS_NOTIFY_UPDATE")
  export const ADACAST_DISABLE_PRUNE = truthy("ADACAST_DISABLE_PRUNE")
  export const ADACAST_DISABLE_TERMINAL_TITLE = truthy("ADACAST_DISABLE_TERMINAL_TITLE")
  export const ADACAST_PERMISSION = process.env["ADACAST_PERMISSION"]
  export const ADACAST_DISABLE_DEFAULT_PLUGINS = truthy("ADACAST_DISABLE_DEFAULT_PLUGINS")
  export const ADACAST_DISABLE_LSP_DOWNLOAD = truthy("ADACAST_DISABLE_LSP_DOWNLOAD")
  export const ADACAST_ENABLE_EXPERIMENTAL_MODELS = truthy("ADACAST_ENABLE_EXPERIMENTAL_MODELS")
  export const ADACAST_DISABLE_AUTOCOMPACT = truthy("ADACAST_DISABLE_AUTOCOMPACT")
  export const ADACAST_DISABLE_MODELS_FETCH = truthy("ADACAST_DISABLE_MODELS_FETCH")
  export const ADACAST_DISABLE_CLAUDE_CODE = truthy("ADACAST_DISABLE_CLAUDE_CODE")
  export const ADACAST_DISABLE_CLAUDE_CODE_PROMPT =
    ADACAST_DISABLE_CLAUDE_CODE || truthy("ADACAST_DISABLE_CLAUDE_CODE_PROMPT")
  export const ADACAST_DISABLE_CLAUDE_CODE_SKILLS =
    ADACAST_DISABLE_CLAUDE_CODE || truthy("ADACAST_DISABLE_CLAUDE_CODE_SKILLS")
  export const ADACAST_DISABLE_EXTERNAL_SKILLS =
    ADACAST_DISABLE_CLAUDE_CODE_SKILLS || truthy("ADACAST_DISABLE_EXTERNAL_SKILLS")
  export declare const ADACAST_DISABLE_PROJECT_CONFIG: boolean
  export const ADACAST_FAKE_VCS = process.env["ADACAST_FAKE_VCS"]
  export declare const ADACAST_CLIENT: string
  export const ADACAST_SERVER_PASSWORD = process.env["ADACAST_SERVER_PASSWORD"]
  export const ADACAST_SERVER_USERNAME = process.env["ADACAST_SERVER_USERNAME"]
  export const ADACAST_ENABLE_QUESTION_TOOL = truthy("ADACAST_ENABLE_QUESTION_TOOL")

  // Experimental
  export const ADACAST_EXPERIMENTAL = truthy("ADACAST_EXPERIMENTAL")
  export const ADACAST_EXPERIMENTAL_FILEWATCHER = Config.boolean("ADACAST_EXPERIMENTAL_FILEWATCHER").pipe(
    Config.withDefault(false),
  )
  export const ADACAST_EXPERIMENTAL_DISABLE_FILEWATCHER = Config.boolean(
    "ADACAST_EXPERIMENTAL_DISABLE_FILEWATCHER",
  ).pipe(Config.withDefault(false))
  export const ADACAST_EXPERIMENTAL_ICON_DISCOVERY =
    ADACAST_EXPERIMENTAL || truthy("ADACAST_EXPERIMENTAL_ICON_DISCOVERY")

  const copy = process.env["ADACAST_EXPERIMENTAL_DISABLE_COPY_ON_SELECT"]
  export const ADACAST_EXPERIMENTAL_DISABLE_COPY_ON_SELECT =
    copy === undefined ? process.platform === "win32" : truthy("ADACAST_EXPERIMENTAL_DISABLE_COPY_ON_SELECT")
  export const ADACAST_ENABLE_EXA =
    truthy("ADACAST_ENABLE_EXA") || ADACAST_EXPERIMENTAL || truthy("ADACAST_EXPERIMENTAL_EXA")
  export const ADACAST_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS = number("ADACAST_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS")
  export const ADACAST_EXPERIMENTAL_OUTPUT_TOKEN_MAX = number("ADACAST_EXPERIMENTAL_OUTPUT_TOKEN_MAX")
  export const ADACAST_EXPERIMENTAL_OXFMT = ADACAST_EXPERIMENTAL || truthy("ADACAST_EXPERIMENTAL_OXFMT")
  export const ADACAST_EXPERIMENTAL_LSP_TY = truthy("ADACAST_EXPERIMENTAL_LSP_TY")
  export const ADACAST_EXPERIMENTAL_LSP_TOOL = ADACAST_EXPERIMENTAL || truthy("ADACAST_EXPERIMENTAL_LSP_TOOL")
  export const ADACAST_DISABLE_FILETIME_CHECK = Config.boolean("ADACAST_DISABLE_FILETIME_CHECK").pipe(
    Config.withDefault(false),
  )
  export const ADACAST_EXPERIMENTAL_PLAN_MODE = ADACAST_EXPERIMENTAL || truthy("ADACAST_EXPERIMENTAL_PLAN_MODE")
  export const ADACAST_EXPERIMENTAL_WORKSPACES = ADACAST_EXPERIMENTAL || truthy("ADACAST_EXPERIMENTAL_WORKSPACES")
  export const ADACAST_EXPERIMENTAL_MARKDOWN = !falsy("ADACAST_EXPERIMENTAL_MARKDOWN")
  export const ADACAST_MODELS_URL = process.env["ADACAST_MODELS_URL"]
  export const ADACAST_MODELS_PATH = process.env["ADACAST_MODELS_PATH"]
  export const ADACAST_DB = process.env["ADACAST_DB"]
  export const ADACAST_DISABLE_CHANNEL_DB = truthy("ADACAST_DISABLE_CHANNEL_DB")
  export const ADACAST_SKIP_MIGRATIONS = truthy("ADACAST_SKIP_MIGRATIONS")
  export const ADACAST_STRICT_CONFIG_DEPS = truthy("ADACAST_STRICT_CONFIG_DEPS")

  function number(key: string) {
    const value = process.env[key]
    if (!value) return undefined
    const parsed = Number(value)
    return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined
  }
}

// Dynamic getter for ADACAST_DISABLE_PROJECT_CONFIG
// This must be evaluated at access time, not module load time,
// because external tooling may set this env var at runtime
Object.defineProperty(Flag, "ADACAST_DISABLE_PROJECT_CONFIG", {
  get() {
    return truthy("ADACAST_DISABLE_PROJECT_CONFIG")
  },
  enumerable: true,
  configurable: false,
})

// Dynamic getter for ADACAST_TUI_CONFIG
// This must be evaluated at access time, not module load time,
// because tests and external tooling may set this env var at runtime
Object.defineProperty(Flag, "ADACAST_TUI_CONFIG", {
  get() {
    return process.env["ADACAST_TUI_CONFIG"]
  },
  enumerable: true,
  configurable: false,
})

// Dynamic getter for ADACAST_CONFIG_DIR
// This must be evaluated at access time, not module load time,
// because external tooling may set this env var at runtime
Object.defineProperty(Flag, "ADACAST_CONFIG_DIR", {
  get() {
    return process.env["ADACAST_CONFIG_DIR"]
  },
  enumerable: true,
  configurable: false,
})

// Dynamic getter for ADACAST_CLIENT
// This must be evaluated at access time, not module load time,
// because some commands override the client at runtime
Object.defineProperty(Flag, "ADACAST_CLIENT", {
  get() {
    return process.env["ADACAST_CLIENT"] ?? "cli"
  },
  enumerable: true,
  configurable: false,
})
