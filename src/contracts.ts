import { z } from "zod"

export const VERSION = "0.3.0"
export const PROTOCOL_VERSION = 1
const positive = z.number().int().positive()
export const TaskSchema = z.object({
  dataset_path: z.string().min(1), time_col: z.string().min(1), target_col: z.string().min(1),
  frequency: z.string().min(1), seq_len: positive, pred_len: positive,
  strict_frequency: z.boolean().default(false),
  forecast_gap: z.number().int().nonnegative().default(0), label_len: z.number().int().nonnegative().default(0),
  window_stride: positive.default(1), split_mode: z.enum(["ratio", "timestamp"]).default("ratio"),
  train_ratio: z.number().gt(0).lt(1).optional(), val_ratio: z.number().gt(0).lt(1).optional(),
  test_ratio: z.number().gt(0).lt(1).optional(), train_end_timestamp: z.string().optional(),
  validation_start_timestamp: z.string().optional(),
  prediction_target_start_timestamp: z.string().min(1).optional(),
  validation_end_timestamp: z.string().optional(),
  eval_metrics: z.array(z.enum(["mse", "mae", "wape", "mase"])).min(1).default(["mse", "mae"]),
  model_families: z.array(z.string().min(1)).min(1), reflection_enabled: z.boolean().default(true),
  holiday_effects: z.string().default(""), anomaly_periods: z.string().default(""),
  adjustment_fit_holdout_ratio:z.literal(0.5).default(0.5),
  business_context: z.string().default(""), csv_semantics: z.string().default(""),
  exogenous_columns: z.array(z.string().min(1)).optional(),
  exogenous_history_steps: positive.optional(),
  region_info: z.string().default(""), seasonality_hints: z.string().default(""),
  adjustment_skill_priors: z.string().default("")
}).strict().superRefine((v, ctx) => {
  if(v.exogenous_history_steps!==undefined && v.exogenous_history_steps>v.seq_len)
    ctx.addIssue({code:"custom",message:"exogenous_history_steps cannot exceed seq_len"})
  if(v.exogenous_columns && (new Set(v.exogenous_columns).size!==v.exogenous_columns.length || v.exogenous_columns.some(c=>c===v.target_col||c===v.time_col)))
    ctx.addIssue({code:"custom",message:"exogenous_columns must be unique and exclude target/time columns"})
  if (v.label_len > v.seq_len) ctx.addIssue({ code: "custom", message: "label_len cannot exceed seq_len" })
  if (v.split_mode === "ratio") {
    if (v.train_end_timestamp || v.validation_start_timestamp || v.prediction_target_start_timestamp || v.validation_end_timestamp ||
      [v.train_ratio, v.val_ratio, v.test_ratio].some(x => x === undefined) ||
      Math.abs((v.train_ratio ?? 0) + (v.val_ratio ?? 0) + (v.test_ratio ?? 0) - 1) > 1e-9)
      ctx.addIssue({ code: "custom", message: "ratio split requires three ratios summing to 1 and no timestamps" })
  } else if (!v.train_end_timestamp || !v.validation_end_timestamp ||
    [v.train_ratio, v.val_ratio, v.test_ratio].some(x => x !== undefined)) {
    ctx.addIssue({ code: "custom", message: "timestamp split requires exactly two boundaries, without ratios" })
  }
})

export const ModelCapabilitiesSchema = z.object({
  context_window_tokens: positive.optional(), input_limit_tokens: positive.optional(),
  max_output_tokens: positive.optional(), supports_temperature: z.boolean().optional(),
  supports_tools: z.boolean().optional(), supports_top_k: z.boolean().optional(),
}).strict()
export const SamplingSchema = z.object({
  temperature: z.number().min(0).max(2).optional(), top_p: z.number().gt(0).max(1).optional(),
  top_k: positive.optional(), seed: z.number().int().optional(),
  reasoning_effort: z.enum(["none","minimal","low","medium","high","xhigh"]).optional(),
})
export const LLMSchema = SamplingSchema.extend({
  provider: z.literal("openai-compatible").default("openai-compatible"),
  base_url: z.string().url().optional(), base_url_env: z.string().default("ADACAST_LITE_LLM_BASE_URL"),
  api_key_env: z.string().default("ADACAST_LITE_LLM_API_KEY"),
  model: z.string().optional(), model_env: z.string().default("ADACAST_LITE_LLM_MODEL"),
  context_window_tokens: positive.default(64000), max_output_tokens: positive.default(32000),
  max_retries: z.number().int().min(0).max(5).default(0),
  model_capabilities: z.record(z.string(),ModelCapabilitiesSchema).default({}),
}).strict()
export const ConfigSchema = z.object({
  schema_version: z.literal(1), task: TaskSchema,
  knowledge: z.object({ files: z.array(z.string()).default([]) }).strict().default({ files: [] }),
  instructions: z.object({ files: z.array(z.string()).default([]) }).strict().default({ files: [] }),
  llm: LLMSchema.optional(),
  context: z.object({
    auto: z.boolean().default(true), prune: z.boolean().default(true), reserved_tokens: z.number().int().nonnegative().optional(),
    prune_minimum_tokens: positive.default(20000), prune_protect_tokens: positive.default(40000),
    tool_output_max_lines: positive.default(2000), tool_output_max_bytes: positive.default(50*1024),
  }).strict().prefault({}),
  execution: z.object({
    mode: z.enum(["agent", "deterministic"]).default("agent"),
    max_experiments: positive.default(20), max_consecutive_crashes: positive.default(3),
    max_consecutive_no_improvement: positive.default(3), max_steps_per_agent: positive.default(100),
    max_total_llm_calls: positive.default(500), max_total_tokens: positive.default(2000000),
    task_timeout_seconds: positive.default(7200), tool_timeout_seconds: positive.default(1800),
    llm_timeout_seconds: positive.default(300), window_llm_concurrency: positive.max(10).default(4),
    max_subagent_depth: z.number().int().min(0).max(5).default(2)
  }).strict().prefault({}),
  decisions: z.object({
    prediction_prior: z.discriminatedUnion("mode", [z.object({ mode: z.literal("none") }).strict(),
      z.object({ mode: z.literal("csv"), path: z.string() }).strict()]).default({ mode: "none" }),
    skills: z.object({ review: z.literal("auto_validate") }).default({ review: "auto_validate" }),
    adjustment: z.object({ strategy: z.enum(["off", "a_lightweight", "b_comprehensive", "agent"]).default("agent") }).prefault({}),
    no_improvement: z.literal("finish_search").default("finish_search"),
    experiment_budget_exhausted: z.literal("finish_search").default("finish_search")
  }).strict().prefault({}),
  tools: z.object({ profile: z.literal("forecast").default("forecast"),
    extra_read_paths: z.array(z.string()).default([]), extra_write_paths: z.array(z.string()).default([]),
    allow_shell: z.boolean().default(false), allow_network: z.boolean().default(false),
    allowed_commands: z.array(z.string()).default([])
  }).strict().prefault({}),
  memory: z.object({ enabled: z.boolean().default(true), directory: z.string().default("./memory"),
    publish_on_forecast_complete: z.boolean().default(true) }).strict().prefault({}),
  agents: z.record(z.string(), SamplingSchema.extend({ prompt: z.string().default(""), model: z.string().optional(),
    max_output_tokens: positive.optional(), tools: z.array(z.string()).optional() }).strict()).default({})
}).strict().superRefine((v, ctx) => {
  if (v.execution.mode === "agent" && !v.llm) ctx.addIssue({ code: "custom", message: "Agent mode requires llm configuration" })
  if(v.task.prediction_target_start_timestamp && v.decisions.prediction_prior.mode!=="csv")
    ctx.addIssue({code:"custom",message:"prediction_target_start_timestamp requires a supplied CSV prior"})
  if (v.llm && v.llm.max_output_tokens + 1024 >= v.llm.context_window_tokens)
    ctx.addIssue({ code: "custom", message: "Context window must exceed output allowance plus 1024 tokens" })
  if (v.execution.mode === "deterministic" && (v.task.reflection_enabled || v.decisions.adjustment.strategy !== "off"))
    ctx.addIssue({ code: "custom", message: "Deterministic mode requires reflection_enabled=false and adjustment.strategy=off" })
})
export type Config = z.infer<typeof ConfigSchema>
export type Task = z.infer<typeof TaskSchema>
export type Phase = "init" | "pre-forecast" | "skills-review" | "forecast" | "post-forecast"
export type Lifecycle = "pending" | "running" | "interrupted" | "blocked" | "failed" | "succeeded"
export type TokenUsage = { input?:number; output?:number; cache_read?:number; cache_write?:number; reasoning?:number; total?:number }
export type Message = { role: "system" | "user" | "assistant" | "tool"; content: string | null;
  reasoning_content?: string | null; audit?: { usage?:TokenUsage; model?:string; finish_reason?:string|null };
  tool_calls?: { id: string; type: "function"; function: { name: string; arguments: string } }[]; tool_call_id?: string }
export type ToolDefinition = { name: string; description: string; schema: z.ZodType;
  roles: string[]; replaySafe?: boolean; execute: (args: any, ctx: ToolContext) => Promise<unknown> }
export type ToolContext = { sessionId: string; agent: string; callId: string; signal: AbortSignal; depth: number }
export type RunState = { lifecycle: Lifecycle; phase: Phase; taskId: string; createdAt: string;
  elapsedMs: number; error?: { code: string; message: string }; completedAt?: string }
export type Experiment = { run_id: string; model: string; spec: Record<string, unknown>; split: string;
  metrics: Record<string, number | null>; artifact: string; artifact_sha256: string;
  predictions: string; source_run_id?: string; status: string }

export class AppError extends Error {
  constructor(public code: string, message: string, public exitCode = 4, public retryable = false) {
    super(message); this.name = "AppError"
  }
}
