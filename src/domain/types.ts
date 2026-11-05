import z from "zod"

const nullableMetrics = z.object({
  mse: z.number().nullable().describe("Mean squared error for the experiment or cached result"),
  mae: z.number().nullable().describe("Mean absolute error for the experiment or cached result"),
  wape: z.number().nullable().describe("Weighted absolute percentage error for the experiment or cached result"),
  mase: z.number().nullable().describe("Mean absolute scaled error for the experiment or cached result"),
})

const evalMetrics = z.object({
  mse: z.number().describe("Mean squared error produced by the fixed evaluator"),
  mae: z.number().describe("Mean absolute error produced by the fixed evaluator"),
  wape: z.number().nullable().describe("Weighted absolute percentage error, or null when its denominator is zero"),
  mase: z.number().nullable().describe("Mean absolute scaled error, or null when the train scale is unavailable"),
})

export const FIXED_ADJUSTMENT_FIT_HOLDOUT_RATIO = 1 / 2

export const TaskConfig = z
  .object({
    dataset_path: z.string().describe("Absolute or project-relative path to the source CSV dataset"),
    target_col: z.string().describe("Name of the target column to forecast"),
    time_col: z.string().describe("Name of the timestamp column in the dataset"),
    pred_len: z.number().int().positive().describe("Prediction horizon in time steps"),
    forecast_gap: z
      .number()
      .int()
      .nonnegative()
      .default(0)
      .describe("Unevaluated future steps between the history window and the evaluated prediction horizon"),
    seq_len: z.number().int().positive().describe("Input sequence length in time steps"),
    label_len: z.number().int().nonnegative().describe("Decoder warmup length in time steps"),
    window_stride: z
      .number()
      .int()
      .positive()
      .default(1)
      .describe("Unified sliding-window step for training and evaluation; omitted legacy tasks use 1"),
    split_mode: z
      .enum(["ratio", "timestamp"])
      .optional()
      .describe("Frozen split strategy; omitted legacy tasks use ratio mode"),
    train_ratio: z.number().positive().lt(1).describe("Fraction of rows reserved for the training split"),
    val_ratio: z.number().positive().lt(1).describe("Fraction of rows reserved for the validation split"),
    test_ratio: z.number().positive().lt(1).describe("Fraction of rows reserved for the test split"),
    train_end_timestamp: z
      .string()
      .min(1)
      .optional()
      .describe("Inclusive final training history/origin timestamp in timestamp mode"),
    validation_start_timestamp: z
      .string()
      .min(1)
      .optional()
      .describe("Inclusive first validation label timestamp; intervening post-training rows are history only"),
    prediction_target_start_timestamp: z.string().min(1).optional().describe("Explicit first UserPrior target timestamp anchoring the global sliding grid independently of first changed CSV value"),
    validation_end_timestamp: z
      .string()
      .min(1)
      .optional()
      .describe("Inclusive final validation history/origin timestamp in timestamp mode"),
    adjustment_fit_holdout_ratio: z
      .number()
      .default(FIXED_ADJUSTMENT_FIT_HOLDOUT_RATIO)
      .refine((value) => Math.abs(value - FIXED_ADJUSTMENT_FIT_HOLDOUT_RATIO) <= 1e-12, {
        message: "adjustment_fit_holdout_ratio is fixed at 1/2 of the validation split",
      })
      .describe("Fixed front fraction of the validation split reserved for optional adjustment evidence"),
    eval_metrics: z
      .array(z.enum(["mse", "mae", "wape", "mase"]))
      .describe("Evaluation metrics the fixed evaluator should report"),
    model_families: z.array(z.string()).min(1).describe("Model families allowed for experiment generation"),
    exogenous_columns: z.array(z.string().min(1)).optional().describe("Allowlisted columns for adjustment and blind exogenous reasoning; source CSV and prior remain intact"),
    exogenous_history_steps: z.number().int().positive().optional().describe("Maximum past exogenous rows visible to blind reasoning; target history retains seq_len rows"),
    business_context: z.string().optional().describe("Business background and operating context for the forecast"),
    csv_semantics: z.string().optional().describe("Column semantics or data dictionary notes for the CSV"),
    region_info: z.string().optional().describe("Regional market or geography details relevant to the forecast"),
    seasonality_hints: z.string().optional().describe("Known seasonality patterns the agent should account for"),
    holiday_effects: z.string().optional().describe("Holiday or calendar effects that may shift the target series"),
    anomaly_periods: z.string().optional().describe("Known anomaly periods or incidents that changed the series"),
    adjustment_skill_priors: z
      .string()
      .optional()
      .describe(
        "User-supplied or synthesized expert rules whose missing numeric parameters are completed in an independent validation-gated mirror candidate",
      ),
    reflection_enabled: z
      .boolean()
      .default(true)
      .describe("Whether the independent Critic window-reflection safety controller is enabled"),
    frozen_at: z.string().describe("ISO-8601 timestamp recording when the task definition was frozen"),
  })
  .superRefine((task, ctx) => {
    const splitMode = task.split_mode ?? "ratio"
    const hasTrainTimestamp = task.train_end_timestamp !== undefined
    const hasValidationTimestamp = task.validation_end_timestamp !== undefined
    if (splitMode === "ratio" && (hasTrainTimestamp || hasValidationTimestamp || task.validation_start_timestamp !== undefined || task.prediction_target_start_timestamp !== undefined)) {
      ctx.addIssue({
        code: "custom",
        path: ["split_mode"],
        message: "ratio split mode cannot include timestamp boundaries",
      })
    }
    if (splitMode === "timestamp" && (!hasTrainTimestamp || !hasValidationTimestamp)) {
      ctx.addIssue({
        code: "custom",
        path: ["split_mode"],
        message: "timestamp split mode requires train_end_timestamp and validation_end_timestamp",
      })
    }
    const ratioSum = task.train_ratio + task.val_ratio + task.test_ratio
    if (Math.abs(ratioSum - 1) > 1e-9) {
      ctx.addIssue({
        code: "custom",
        path: ["test_ratio"],
        message: `train_ratio + val_ratio + test_ratio must equal 1 (received ${ratioSum})`,
      })
    }
  })
export type TaskConfig = z.infer<typeof TaskConfig>

export const WindowFeatures = z.object({
  trend: z
    .enum(["upward", "downward", "stable"])
    .optional()
    .describe("Dominant trend direction in the look-back window"),
  seasonality: z.enum(["strong", "weak", "none"]).optional().describe("Seasonality strength assessed via STL or ACF"),
  stationary: z.boolean().optional().describe("Whether the series is stationary per ADF test (p < 0.05)"),
  volatility: z
    .enum(["high", "medium", "low"])
    .optional()
    .describe("Volatility category based on coefficient of variation (high: CV>0.5, medium: 0.2-0.5, low: <0.2)"),
  model_family: z
    .enum(["statistical", "deep_learning", "foundation"])
    .optional()
    .describe("Model family selected for this window based on its conditions"),
})
export type WindowFeatures = z.infer<typeof WindowFeatures>

export const ExperimentRecord = z.object({
  run_id: z.string().describe("Spec hash identifying the experiment run"),
  attempt_index: z
    .number()
    .int()
    .positive()
    .optional()
    .describe("One-based execution attempt for this deterministic run id; omitted legacy records are attempt 1"),
  model: z.string().describe("Model name used by the Python runner"),
  status: z.enum(["success", "crash", "timeout", "ipc_error"]).describe("Terminal status for the experiment attempt"),
  metrics: nullableMetrics.nullable().describe("Recorded evaluation metrics, or null when no evaluation completed"),
  spec: z.record(z.string(), z.unknown()).describe("Experiment specification persisted for replay and inspection"),
  window_features: WindowFeatures.optional().describe(
    "Time-series condition signals for the look-back window that informed model family selection for this experiment",
  ),
  decision: z
    .enum(["keep", "discard", "crash", "ipc_error"])
    .optional()
    .describe("Post-evaluation decision recorded by the agent"),
  failure_reason: z.string().optional().describe("Crash or IPC failure reason captured for the run"),
  elapsed_seconds: z.number().describe("Total wall-clock duration spent on the experiment"),
  timestamp: z.string().describe("ISO-8601 timestamp for when the record was appended"),
})
export type ExperimentRecord = z.infer<typeof ExperimentRecord>

export const ReflectionRecord = z.object({
  type: z.literal("reflection").describe("Record type discriminator for history.jsonl parsing"),
  run_id: z.string().describe("Spec hash of the experiment this reflection evaluates"),
  effective: z.boolean().describe("Whether the last experiment improved on the current best"),
  reason: z.string().describe("Explanation of the outcome"),
  next_direction: z.string().describe("What to try next and why"),
  decision: z.enum(["keep", "discard", "crash", "stop"]).describe("Outcome action"),
  timestamp: z.string().describe("ISO-8601 timestamp when the reflection was recorded"),
})
export type ReflectionRecord = z.infer<typeof ReflectionRecord>

export const ExpertFeedbackRecord = z.object({
  type: z.literal("expert_feedback").describe("Record type discriminator for history.jsonl parsing"),
  text: z.string().describe("Free-form expert guidance text as entered in the conversation"),
  timestamp: z.string().describe("ISO-8601 timestamp when the feedback was recorded"),
})
export type ExpertFeedbackRecord = z.infer<typeof ExpertFeedbackRecord>

export const EvalArtifactBinding = z.object({
  model_artifact: z.object({
    kind: z.enum([
      "baseline_model",
      "torch_state_dict",
      "statsmodels_arima_results",
      "statsmodels_statistical_results",
      "foundation_model_reference",
      "external_prediction_prior",
    ]),
    sha256: z.string().length(64),
    source_run_id: z.string().min(1),
    foundation_revision: z.string().nullable(),
  }),
  prediction_artifacts: z.object({
    predictions_sha256: z.string().length(64),
    actual_sha256: z.string().length(64),
    pred_sha256: z.string().length(64),
    train_scale_sha256: z.string().length(64),
  }),
})
export type EvalArtifactBinding = z.infer<typeof EvalArtifactBinding>

export const EvalResult = z.object({
  run_id: z.string().describe("Spec hash identifying the evaluated run"),
  evaluator_hash: z.string().describe("SHA-256 hash of the fixed evaluator used for scoring"),
  spec_hash: z.string().describe("SHA-256 hash of the canonical experiment spec"),
  metrics: evalMetrics.describe("Fixed evaluator metrics for the chosen evaluation split"),
  eval_split: z.string().describe("Dataset split that was evaluated, typically val during forecasting"),
  timestamp: z.string().describe("ISO-8601 timestamp for when evaluation completed"),
  artifact_binding: EvalArtifactBinding.optional().describe(
    "Immutable binding between metrics, model artifact, and prediction outputs",
  ),
})
export type EvalResult = z.infer<typeof EvalResult>

export const BestModelArtifact = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("checkpoint"),
    path: z.string().describe("Package-relative path to the copied checkpoint"),
    format: z.literal("torch_state_dict").describe("Serialized checkpoint format"),
    sha256: z.string().length(64).describe("SHA-256 identity of the copied checkpoint"),
    model: z.string().describe("Model name that produced the checkpoint"),
    model_family: z.string().nullable().describe("Model family selected by the Critic, when known"),
  }),
  z.object({
    kind: z.literal("arima_results"),
    path: z.string().describe("Package-relative path to the copied fitted ARIMA results"),
    format: z.literal("statsmodels_arima_results").describe("Serialized ARIMA result format"),
    sha256: z.string().length(64).describe("SHA-256 identity of the fitted ARIMA results"),
    model: z.string().describe("Model name that produced the fitted results"),
    model_family: z.string().nullable().describe("Model family selected by the Critic, when known"),
  }),
  z.object({
    kind: z.literal("statistical_results"),
    path: z.string().describe("Package-relative path to the copied fitted statistical-model results"),
    format: z.literal("statsmodels_results_pickle").describe("Serialized statsmodels fitted-result format"),
    sha256: z.string().length(64).describe("SHA-256 identity of the fitted statistical-model results"),
    model: z.string().describe("Statistical model name that produced the fitted results"),
    model_family: z.string().nullable().describe("Model family selected by the Critic, when known"),
  }),
  z.object({
    kind: z.literal("no_checkpoint"),
    path: z.string().describe("Package-relative path to the explicit no-checkpoint marker"),
    reason: z.string().describe("Why no reusable local checkpoint was available"),
    model: z.string().describe("Model name selected for export"),
    model_family: z.string().nullable().describe("Model family selected by the Critic, when known"),
  }),
  z.object({
    kind: z.literal("external_reference"),
    reference: z.string().describe("External model or cache reference required to reuse the model"),
    reason: z.string().describe("Why the export points to an external artifact instead of a checkpoint"),
    model: z.string().describe("Model name selected for export"),
    model_family: z.string().nullable().describe("Model family selected by the Critic, when known"),
    external_model_id: z.string().optional().describe("Provider-specific model id, when known"),
    revision: z.string().optional().describe("Immutable provider revision used for inference"),
    sha256: z.string().length(64).optional().describe("SHA-256 identity of the external model revision"),
    cache_metadata: z.record(z.string(), z.unknown()).optional().describe("Local cache metadata, when available"),
  }),
])
export type BestModelArtifact = z.infer<typeof BestModelArtifact>

export const BestModelManifest = z.object({
  generated_at: z.string().describe("ISO-8601 timestamp when the best-model package was generated"),
  model: z.string().describe("Selected model name"),
  model_family: z.string().nullable().describe("Selected model family, when known"),
  validation_run_id: z.string().describe("Validation run used for model selection"),
  test_run_id: z.string().describe("Held-out test run copied into the package"),
  validation_metrics: z.record(z.string(), z.number().nullable()).describe("Validation metrics from model selection"),
  test_metrics: z.record(z.string(), z.number().nullable()).describe("Final test metrics for the exported run"),
  task: TaskConfig.describe("Frozen forecast task that the selected run belongs to"),
  spec_hash: z.string().describe("Spec hash for the selected test run"),
  validation_spec_hash: z.string().nullable().describe("Spec hash for the validation run, when available"),
  selection_reason: z.string().nullable().describe("Critic rationale for selecting this model, when provided"),
  source_paths: z.object({
    validation_run: z.string().describe("Project-relative source validation run directory"),
    test_run: z.string().describe("Project-relative source test run directory"),
  }),
  artifacts: z.array(BestModelArtifact).describe("Reusable model artifacts or explicit no-checkpoint references"),
  prediction_outputs: z
    .object({
      raw: z.object({
        predictions: z.string().describe("Package-relative raw predictions path"),
        metrics: z.record(z.string(), z.number().nullable()).describe("Raw test metrics"),
      }),
      adjusted: z
        .object({
          enabled: z.boolean().describe("Whether adjusted outputs were exported"),
          predictions: z.string().nullable().describe("Package-relative adjusted predictions path"),
          metrics: z.record(z.string(), z.number().nullable()).nullable().describe("Adjusted test metrics"),
          delta_vs_raw: z
            .record(z.string(), z.number().nullable())
            .nullable()
            .describe("Adjusted metric minus raw metric"),
          skill_version: z.string().nullable().describe("Adjustment skill version used for test apply"),
          leakage_audit_passed: z.boolean().nullable().describe("Whether adjusted test leakage audit passed"),
          confirmed_before_test_metrics: z
            .boolean()
            .nullable()
            .describe("Whether adjustment was confirmed before adjusted test metrics were computed"),
          decision_actor: z
            .enum(["user", "llm_auto", "auto_policy"])
            .nullable()
            .default(null)
            .describe("Who decided to apply the locked adjustment to the exported test run"),
          decision_reason: z
            .string()
            .nullable()
            .default(null)
            .describe("Recorded rationale for the test-stage adjustment decision"),
        })
        .optional(),
      window_reflection: z
        .object({
          enabled: z.boolean().describe("Whether the independent window-reflection controller ran"),
          predictions: z.string().describe("Package-relative final prediction path"),
          metrics: z.record(z.string(), z.number().nullable()).describe("Final metrics after reflection/fallback"),
          baseline_metrics: z
            .record(z.string(), z.number().nullable())
            .describe("Metrics for the complete pre-reflection baseline"),
          delta_mse: z.number().describe("Final MSE minus pre-reflection baseline MSE"),
          baseline_kind: z.enum(["raw_anchor", "skill_adjusted"]),
          validation_status: z.enum([
            "pending",
            "enabled_no_trigger",
            "enabled_improved",
            "enabled_unchanged",
            "disabled_negative_mse",
            "failed",
          ]),
          llm_refinement_enabled: z.boolean(),
          trigger_count: z.number().int().nonnegative(),
          skill_match_count: z.number().int().nonnegative().default(0),
          llm_required_count: z.number().int().nonnegative().default(0),
          corrected_count: z.number().int().nonnegative(),
          fallback_count: z.number().int().nonnegative(),
          deterministic_fallback_count: z.number().int().nonnegative().default(0),
          raw_fallback_count: z.number().int().nonnegative().default(0),
          llm_protocol: z.string().nullable().default(null),
          leakage_audit_passed: z.boolean(),
        })
        .optional(),
    })
    .optional()
    .describe("Raw and optional adjusted prediction output metadata"),
})
export type BestModelManifest = z.infer<typeof BestModelManifest>

export const BudgetState = z.object({
  experiment_count: z.number().int().nonnegative().describe("Number of experiments counted against the current budget"),
  total_elapsed_seconds: z.number().nonnegative().describe("Total elapsed experiment time consumed so far"),
  last_updated: z.string().describe("ISO-8601 timestamp of the most recent budget update"),
})
export type BudgetState = z.infer<typeof BudgetState>

export const BudgetThresholds = z.object({
  max_experiments: z.number().int().positive().default(20).describe("Maximum total experiments before forced stop (K)"),
  no_improvement_rounds: z
    .number()
    .int()
    .positive()
    .default(3)
    .describe("Consecutive no-improvement rounds before HITL trigger (N)"),
  crash_threshold: z.number().int().positive().default(3).describe("Consecutive crashes before forced stop (M)"),
})
export type BudgetThresholds = z.infer<typeof BudgetThresholds>

export const StopCheckResult = z.object({
  stop: z.boolean().describe("Whether the experiment loop should terminate"),
  reason: z
    .enum(["no_improvement", "budget_exhausted", "crash_threshold", "continue"])
    .describe("Stop reason or continue"),
  hitl_triggered: z.boolean().describe("Whether HITL was triggered by the no-improvement counter reaching threshold"),
  counters: z.object({
    consecutive_no_improvement: z.number().describe("Current no-improvement counter after update"),
    consecutive_crashes: z.number().describe("Current crash counter after update"),
    experiments_remaining: z.number().describe("Experiments remaining in budget"),
  }),
})
export type StopCheckResult = z.infer<typeof StopCheckResult>

export const RunnerOutput = z.object({
  status: z.enum(["success", "cached", "error"]).describe("Runner exit classification returned on stdout"),
  run_id: z.string().describe("Spec hash identifying the run directory"),
  out_dir: z.string().describe("Run artifact directory produced by the Python runner"),
  metrics: nullableMetrics.optional().describe("Evaluation metrics returned by the runner when available"),
  error: z.string().optional().describe("Human-readable error summary when the runner fails"),
  error_type: z.string().optional().describe("Python exception class or failure category returned by the runner"),
})
export type RunnerOutput = z.infer<typeof RunnerOutput>

const cacheSummaryWarning = z.object({
  code: z.string().describe("Machine-readable cache diagnostic warning code"),
  message: z.string().describe("Human-readable cache diagnostic warning message"),
})

const cacheSummaryDependencyMissing = z.object({
  module: z.string().describe("Missing Python module reported by the runner"),
  package: z.string().describe("Install package that should provide the missing module"),
  guidance: z.string().describe("Actionable installation guidance for the missing dependency"),
})

export const CacheSummary = z.object({
  source: z.enum(["operator", "project_default", "hf_default"]).optional().describe("Source of cache_config values"),
  endpoint: z.string().nullable().optional().describe("Effective HF_ENDPOINT value from cache_config"),
  cache_root: z.string().nullable().optional().describe("Effective Hugging Face cache root from cache_config"),
  hits: z.number().int().nonnegative().describe("Number of cache_hit diagnostics observed"),
  misses: z.number().int().nonnegative().describe("Number of cache_miss diagnostics observed"),
  warning_count: z
    .number()
    .int()
    .nonnegative()
    .describe("Number of cache_warning diagnostics plus cache_config warnings"),
  dependency_missing_count: z
    .number()
    .int()
    .nonnegative()
    .describe("Number of dependency_missing diagnostics observed"),
  warnings: z.array(cacheSummaryWarning).describe("Concise cache warnings reported by cache_config and cache_warning"),
  dependencies_missing: z
    .array(cacheSummaryDependencyMissing)
    .describe("Missing dependency diagnostics with install guidance"),
  log_path: z.string().optional().describe("Relative path to the run train.log containing full diagnostics"),
})
export type CacheSummary = z.infer<typeof CacheSummary>

export const CompactSummary = z.object({
  model: z.string().describe("Model name surfaced back to the LLM"),
  status: z
    .enum(["success", "crash", "timeout", "ipc_error", "cached"])
    .describe("Compact experiment status returned to the LLM"),
  run_id: z.string().describe("Spec hash identifying the experiment"),
  metrics: nullableMetrics.nullable().describe("Compact metric summary for the experiment"),
  training_duration_s: z.number().describe("Training duration in seconds for the run or cached lookup"),
  cached: z.boolean().describe("Whether the summary came from an existing completed run"),
  cache_summary: CacheSummary.optional().describe(
    "Concise cache and dependency diagnostics for fresh runner execution",
  ),
})
export type CompactSummary = z.infer<typeof CompactSummary>

export const StderrEvent = z.object({
  type: z
    .enum([
      "train_start",
      "epoch",
      "predict_start",
      "done",
      "cache_config",
      "cache_hit",
      "cache_miss",
      "cache_warning",
      "dependency_missing",
    ])
    .describe("Progress event emitted on stderr by the runner"),
  epoch: z.number().optional().describe("One-based epoch number when the event reports training progress"),
  total_epochs: z.number().optional().describe("Configured total number of epochs for the current training run"),
  loss: z.number().optional().describe("Average training loss reported for the current epoch"),
  cache_root: z.string().nullable().optional().describe("Effective Hugging Face cache root for cache_config events"),
  hub_cache: z.string().nullable().optional().describe("Effective HF_HUB_CACHE value for cache_config events"),
  xet_cache: z.string().nullable().optional().describe("Effective HF_XET_CACHE value for cache_config events"),
  assets_cache: z.string().nullable().optional().describe("Effective HF_ASSETS_CACHE value for cache_config events"),
  transformers_cache: z
    .string()
    .nullable()
    .optional()
    .describe("Effective TRANSFORMERS_CACHE value for cache_config events"),
  endpoint: z.string().nullable().optional().describe("Effective HF_ENDPOINT value for cache_config events"),
  source: z.enum(["operator", "project_default", "hf_default"]).optional().describe("Source of cache_config values"),
  warnings: z
    .array(z.object({ code: z.string(), message: z.string() }))
    .optional()
    .describe("Structured cache_config warnings"),
  repo_id: z.string().optional().describe("Hugging Face repo id for cache_hit or cache_miss events"),
  code: z.string().optional().describe("Warning or diagnostic code for cache_warning events"),
  message: z.string().optional().describe("Human-readable warning or diagnostic message"),
  module: z.string().optional().describe("Missing Python module for dependency_missing events"),
  package: z.string().optional().describe("Install package for dependency_missing events"),
  guidance: z.string().optional().describe("Actionable dependency or cache guidance"),
  elapsed_s: z.number().describe("Elapsed seconds since experiment start when the event was emitted"),
})
export type StderrEvent = z.infer<typeof StderrEvent>

export const ForecastPhase = z.enum(["init", "pre-forecast", "skills-review", "forecast", "post-forecast"])
export type ForecastPhase = z.infer<typeof ForecastPhase>

export const PhaseTransitionRecord = z.object({
  from: ForecastPhase,
  to: ForecastPhase,
  at: z.string().describe("ISO-8601 timestamp of the transition"),
})
export type PhaseTransitionRecord = z.infer<typeof PhaseTransitionRecord>

export const ForecastAutoModeAgent = z.enum(["planner", "forecaster", "critic"])
export type ForecastAutoModeAgent = z.infer<typeof ForecastAutoModeAgent>

export const ForecastAutoModeWorkflowState = z.enum([
  "clarifying",
  "skills-review",
  "running",
  "handoff",
  "waiting",
  "paused",
])
export type ForecastAutoModeWorkflowState = z.infer<typeof ForecastAutoModeWorkflowState>

export const ForecastAutoModeHandoff = z.object({
  from: ForecastAutoModeAgent.describe("Agent handing work off to the next pipeline stage"),
  to: ForecastAutoModeAgent.describe("Agent receiving the next pipeline stage"),
  at: z.string().describe("ISO-8601 timestamp when the handoff was recorded"),
  reason: z.string().describe("Why Auto Mode moved work between agents"),
})
export type ForecastAutoModeHandoff = z.infer<typeof ForecastAutoModeHandoff>

export const ForecastAutoModeState = z.object({
  enabled: z.boolean().default(false).describe("Whether Auto Mode is enabled for the current forecast session"),
  active_agent: ForecastAutoModeAgent.nullable()
    .default(null)
    .describe("Current active pipeline agent while Auto Mode is orchestrating the workflow"),
  state: ForecastAutoModeWorkflowState.default("waiting").describe("Current Auto Mode workflow state"),
  pause_reason: z
    .string()
    .nullable()
    .default(null)
    .describe("Reason Auto Mode paused or is waiting for manual intervention"),
  auto_approve_permissions: z
    .boolean()
    .default(false)
    .describe("Whether routine tool permission prompts are auto-approved in Auto Mode"),
  auto_approve_skills: z
    .boolean()
    .default(false)
    .describe("Whether routine skills review prompts are auto-approved in Auto Mode"),
  handoffs: z
    .array(ForecastAutoModeHandoff)
    .default([])
    .describe("Ordered handoff log recorded while Auto Mode progresses through the pipeline"),
})
export type ForecastAutoModeState = z.infer<typeof ForecastAutoModeState>

export const AdjustmentMode = z.enum(["off", "rule_skill"])
export type AdjustmentMode = z.infer<typeof AdjustmentMode>

export const AdjustmentStrategy = z.enum(["a_lightweight", "b_comprehensive"])
export type AdjustmentStrategy = z.infer<typeof AdjustmentStrategy>

export const AdjustmentDecisionActor = z.enum(["user", "llm_auto", "auto_policy"])
export type AdjustmentDecisionActor = z.infer<typeof AdjustmentDecisionActor>

export const AdjustmentMemoryScope = z.enum(["task", "dataset", "scenario", "transferable"])
export type AdjustmentMemoryScope = z.infer<typeof AdjustmentMemoryScope>

export const AdjustmentPolicy = z.object({
  enabled: z.boolean().default(false).describe("Whether post-forecast adjustment is enabled for this task"),
  mode: AdjustmentMode.default("off").describe("Adjustment execution mode"),
  strategy: AdjustmentStrategy.nullable()
    .default(null)
    .describe("Selected adjustment strategy: lightweight prediction reasoning or deep prediction reasoning"),
  step: z.number().int().positive().default(288).describe("Window step used by adjustment context builders"),
  primary_metric: z
    .literal("mse")
    .default("mse")
    .describe("Validation acceptance metric; AdaCast currently supports the workflow-wide MSE objective"),
  min_relative_mse_gain: z
    .literal(-1)
    .default(-1)
    .describe("Fixed validation MSE floor of -100%; expert-prior mirrors are retained regardless of held-out metric change"),
  max_mae_regression_ratio: z.number().min(0).default(0.005).describe("Allowed MAE regression versus baseline"),
  max_wape_regression_ratio: z.number().min(0).default(0.005).describe("Allowed WAPE regression versus baseline"),
  max_point_delta_ratio: z.number().min(0).max(1).default(0.1).describe("Maximum absolute point adjustment ratio"),
  max_rules: z.number().int().positive().default(8).describe("Maximum number of executable adjustment rules"),
  max_candidate_iterations: z
    .number()
    .int()
    .positive()
    .default(5)
    .describe("Maximum LLM-exploratory candidates; the frozen user-prior mirror has a separate one-candidate budget"),
  memory_scope: AdjustmentMemoryScope.default("task").describe("Scope for accepted skill reuse"),
  auto_confirm_if_validation_passes: z
    .boolean()
    .default(false)
    .describe(
      "Legacy compatibility flag; an explicit manual or Auto Mode test-application decision is still required before adjusted test metrics",
    ),
  lock_before_test: z.boolean().default(true).describe("Whether the adjustment protocol must be locked before test"),
  preserve_raw_predictions: z
    .boolean()
    .default(true)
    .describe("Whether raw prediction artifacts must remain immutable"),
  configured_at: z.string().optional().describe("ISO-8601 timestamp when the policy was configured"),
  locked_at: z.string().optional().describe("ISO-8601 timestamp when the policy was locked"),
})
export type AdjustmentPolicy = z.infer<typeof AdjustmentPolicy>

export function parseStoredAdjustmentPolicy(input: unknown): AdjustmentPolicy {
  const stored = input && typeof input === "object" && !Array.isArray(input) ? input : {}
  // Normalize every stored task to the current frozen -100% validation floor.
  return AdjustmentPolicy.parse({ ...stored, min_relative_mse_gain: -1 })
}

export type AdjustmentRuleCondition =
  | { kind: "always" }
  | { kind: "horizon_index_in"; values: number[] }
  | { kind: "timestamp_hour_in"; values: number[] }
  | { kind: "timestamp_month_in"; values: number[] }
  | { kind: "timestamp_time_between"; start: string; end: string }
  | { kind: "raw_predicted_original_between"; min: number; max: number }
  | { kind: "exogenous_numeric_between"; column: string; min: number; max: number; day_offset?: number }
  | { kind: "exogenous_text_in"; column: string; values: string[]; day_offset?: number }
  | { kind: "exogenous_text_contains"; column: string; values: string[]; day_offset?: number }
  | { kind: "exogenous_bool_is"; column: string; value: boolean; day_offset?: number }
  | { kind: "all"; conditions: AdjustmentRuleCondition[] }
  | { kind: "any"; conditions: AdjustmentRuleCondition[] }

export const AdjustmentRuleCondition: z.ZodType<AdjustmentRuleCondition> = z.lazy(() =>
  z.discriminatedUnion("kind", [
    z.object({ kind: z.literal("always") }),
    z.object({
      kind: z.literal("horizon_index_in"),
      values: z.array(z.number().int().nonnegative()).min(1),
    }),
    z.object({
      kind: z.literal("timestamp_hour_in"),
      values: z.array(z.number().int().min(0).max(23)).min(1),
    }),
    z.object({
      kind: z.literal("timestamp_month_in"),
      values: z.array(z.number().int().min(1).max(12)).min(1),
    }),
    z.object({
      kind: z.literal("timestamp_time_between"),
      start: z.string().regex(/^\d{2}:\d{2}$/, "start must be HH:mm"),
      end: z.string().regex(/^\d{2}:\d{2}$/, "end must be HH:mm"),
    }),
    z.object({
      kind: z.literal("raw_predicted_original_between"),
      min: z.number(),
      max: z.number(),
    }),
    z.object({
      kind: z.literal("exogenous_numeric_between"),
      column: z.string().min(1),
      min: z.number(),
      max: z.number(),
      day_offset: z.number().int().min(-14).max(14).optional(),
    }),
    z.object({
      kind: z.literal("exogenous_text_in"),
      column: z.string().min(1),
      values: z.array(z.string().min(1)).min(1),
      day_offset: z.number().int().min(-14).max(14).optional(),
    }),
    z.object({
      kind: z.literal("exogenous_text_contains"),
      column: z.string().min(1),
      values: z.array(z.string().min(1)).min(1),
      day_offset: z.number().int().min(-14).max(14).optional(),
    }),
    z.object({
      kind: z.literal("exogenous_bool_is"),
      column: z.string().min(1),
      value: z.boolean(),
      day_offset: z.number().int().min(-14).max(14).optional(),
    }),
    z.object({
      kind: z.literal("all"),
      conditions: z.array(AdjustmentRuleCondition).min(1).max(8),
    }),
    z.object({
      kind: z.literal("any"),
      conditions: z.array(AdjustmentRuleCondition).min(1).max(8),
    }),
  ]),
)

export const AdjustmentRuleAction = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("add"),
    value: z.number().describe("Additive adjustment in original target units"),
  }),
  z.object({
    type: z.literal("multiply"),
    value: z.number().positive().describe("Multiplicative adjustment factor in original target units"),
  }),
  z.object({
    type: z.literal("clip"),
    min: z.number().optional().describe("Minimum original target value"),
    max: z.number().optional().describe("Maximum original target value"),
  }),
  z.object({
    type: z.literal("add_profile"),
    by: z.literal("horizon_index").default("horizon_index"),
    values: z
      .array(z.number())
      .min(1)
      .describe("Additive adjustment profile in original target units, indexed by horizon_index"),
  }),
])
export type AdjustmentRuleAction = z.infer<typeof AdjustmentRuleAction>

export const AdjustmentRule = z.object({
  id: z.string().min(1).describe("Stable rule id within the candidate"),
  condition: AdjustmentRuleCondition.describe("Condition selecting rows to adjust"),
  action: AdjustmentRuleAction.describe("Deterministic adjustment action"),
  rationale: z.string().min(1).describe("Why this rule is expected to help"),
  evidence_refs: z.array(z.string()).min(1).describe("Validation-front evidence references supporting the rule"),
  bounds: z
    .object({
      max_delta_ratio: z.number().min(0).max(1).optional().describe("Rule-specific max point delta ratio"),
      max_delta_original: z.number().nonnegative().optional().describe("Rule-specific absolute max point delta"),
      min_delta_original: z.number().optional().describe("Rule-specific absolute min point delta, e.g. 0 for up-only"),
    })
    .default({}),
})
export type AdjustmentRule = z.infer<typeof AdjustmentRule>

export const AdjustmentCandidateOrigin = z.enum(["llm_exploratory", "user_prior_mirror"])
export type AdjustmentCandidateOrigin = z.infer<typeof AdjustmentCandidateOrigin>

export const AdjustmentUserPriorReference = z.object({
  requested_path: z.string().min(1),
  resolved_path: z.string().min(1),
  sha256: z.string().length(64),
  size_bytes: z.number().int().nonnegative(),
  content: z.string().describe("Read-only UTF-8 text captured without importing or executing the file"),
})
export type AdjustmentUserPriorReference = z.infer<typeof AdjustmentUserPriorReference>

export const AdjustmentUserPriorEvidence = z.object({
  version: z.literal(1),
  source_text: z.string().min(1).describe("Frozen adjustment_skill_priors text from task.json"),
  source_sha256: z.string().length(64),
  references: z.array(AdjustmentUserPriorReference),
  inspected_at: z.string(),
  artifact_sha256: z.string().length(64),
})
export type AdjustmentUserPriorEvidence = z.infer<typeof AdjustmentUserPriorEvidence>

export const AdjustmentUserPriorParameterValue = z.union([
  z.number(),
  z.string(),
  z.boolean(),
  z.array(z.number()).min(1),
  z.array(z.string()).min(1),
])
export type AdjustmentUserPriorParameterValue = z.infer<typeof AdjustmentUserPriorParameterValue>

export const AdjustmentUserPriorParameterLocator = z.object({
  rule_id: z.string().min(1),
  field: z
    .string()
    .regex(
      /^(condition|action|bounds)(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|\d+))+$/,
      "field must be a safe condition/action/bounds property path",
    ),
})
export type AdjustmentUserPriorParameterLocator = z.infer<typeof AdjustmentUserPriorParameterLocator>

export const AdjustmentUserPriorParameterCompletion = AdjustmentUserPriorParameterLocator.extend({
  value: AdjustmentUserPriorParameterValue,
  source: z.enum(["explicit_prompt", "readonly_reference", "validation_front_fit", "llm_inferred"]),
  evidence_ref: z.string().min(1).optional(),
  rationale: z.string().min(1).describe("Why this immutable field or missing numeric value has the recorded source"),
})
export type AdjustmentUserPriorParameterCompletion = z.infer<typeof AdjustmentUserPriorParameterCompletion>

export const AdjustmentUserPriorRuleMapping = z.object({
  rule_id: z.string().min(1),
  source_excerpt: z.string().min(1).describe("User-prior excerpt or summary represented by this executable rule"),
})
export type AdjustmentUserPriorRuleMapping = z.infer<typeof AdjustmentUserPriorRuleMapping>

export const AdjustmentUserPriorContract = z.object({
  evidence_path: z
    .string()
    .regex(
      /^\.forecast\/adjustments\/user-prior\/evidence-[a-f0-9]{64}\.json$/,
      "evidence_path must reference a frozen user-prior inspection artifact",
    ),
  source_sha256: z.string().length(64),
  validation_front_match: z.enum(["full", "partial", "none"]),
  rule_mappings: z.array(AdjustmentUserPriorRuleMapping),
  unresolved_parameters_before_llm: z.array(AdjustmentUserPriorParameterLocator).default([]),
  parameter_completions: z.array(AdjustmentUserPriorParameterCompletion).min(1),
})
export type AdjustmentUserPriorContract = z.infer<typeof AdjustmentUserPriorContract>

export const AdjustmentCandidate = z.object({
  candidate_id: z.string().min(1).describe("Stable candidate id"),
  source_model_set_id: z.string().min(1).describe("Source model set this candidate was fit from"),
  parent_skill_version: z.string().nullable().default(null).describe("Active skill version this candidate improves on"),
  status: z.enum(["candidate", "accepted", "rejected"]).default("candidate"),
  origin: AdjustmentCandidateOrigin.default("llm_exploratory"),
  user_prior_contract: AdjustmentUserPriorContract.nullable().default(null),
  rules: z.array(AdjustmentRule).min(1).describe("Executable adjustment rules"),
  skill_markdown: z.string().min(1).describe("Human-readable SKILL.md body for the candidate"),
  created_at: z.string().describe("ISO-8601 timestamp when the candidate was created"),
  llm_provenance: z
    .record(z.string(), z.unknown())
    .default({})
    .describe("LLM model/prompt hashes and evidence metadata"),
})
export type AdjustmentCandidate = z.infer<typeof AdjustmentCandidate>

export const AdjustmentSourceModelSet = z.object({
  source_model_set_id: z.string().min(1).describe("Stable source set id"),
  created_at: z.string().describe("ISO-8601 timestamp when the set was frozen"),
  created_from: z.string().default("post_raw_forecast_search").describe("How the source set was selected"),
  members: z
    .array(
      z.object({
        family: z.string().describe("Model family or grouping key"),
        validation_run_id: z.string().describe("Validation run id for the final source model"),
        history_run_id: z
          .string()
          .optional()
          .describe("Original history.jsonl run id when it differs from the artifact-complete validation run"),
        model: z.string().describe("Model name"),
        spec_hash: z.string().describe("Hash of the stored validation spec"),
        validation_metrics: z.record(z.string(), z.number().nullable()).describe("Validation metrics"),
        artifacts: z
          .object({
            spec_path: z.string().describe("Project-relative validation spec artifact"),
            eval_path: z.string().describe("Project-relative validation eval artifact"),
            predictions_path: z.string().describe("Project-relative validation predictions artifact"),
          })
          .optional()
          .describe("Artifact-complete validation source paths used by adjustment tools"),
      }),
    )
    .min(1),
})
export type AdjustmentSourceModelSet = z.infer<typeof AdjustmentSourceModelSet>

export const AdjustmentFitRunStatus = z.enum(["prepared", "success", "cached", "crash", "timeout", "ipc_error"])
export type AdjustmentFitRunStatus = z.infer<typeof AdjustmentFitRunStatus>

export const AdjustmentFitRunMember = z.object({
  family: z.string().describe("Source model family or grouping key"),
  validation_run_id: z.string().describe("Validation run id this fit replay is derived from"),
  model: z.string().describe("Model name"),
  validation_spec_hash: z.string().describe("Hash of the frozen validation source spec"),
  fit_spec_hash: z.string().describe("Hash of the validation-front fit replay spec"),
  fit_run_id: z.string().describe("Run id for the validation-front fit replay"),
  status: AdjustmentFitRunStatus.describe("Fit replay status"),
  fit_method: z
    .enum([
      "runner_reuse_validation_front_half",
      "user_prediction_prior_validation_front_half",
      "runner_train_internal_holdout",
      "runner_train_split_eval",
      "arima_in_sample",
    ])
    .describe("How the validation-front fit evidence was generated"),
  spec_path: z.string().describe("Project-relative fit replay spec path"),
  fit_run_path: z.string().describe("Project-relative fit replay run directory"),
  predictions_path: z.string().nullable().default(null).describe("Project-relative validation-front predictions path"),
  eval_path: z.string().nullable().default(null).describe("Project-relative validation-front eval path"),
  metrics: evalMetrics.nullable().default(null).describe("Fit replay metrics, when available"),
  error: z.string().nullable().default(null).describe("Failure reason, when the replay did not complete"),
})
export type AdjustmentFitRunMember = z.infer<typeof AdjustmentFitRunMember>

export const AdjustmentFitRunManifest = z.object({
  source_model_set_id: z.string().min(1).describe("Source set used to build the fit replays"),
  created_at: z.string().describe("ISO-8601 timestamp when the manifest was first prepared"),
  updated_at: z.string().describe("ISO-8601 timestamp when the manifest was last written"),
  status: z.enum(["prepared", "complete", "partial", "failed"]).describe("Aggregate fit replay status"),
  members: z.array(AdjustmentFitRunMember).min(1).describe("One validation-front fit replay per selected source model"),
})
export type AdjustmentFitRunManifest = z.infer<typeof AdjustmentFitRunManifest>

export const AdjustmentAcceptanceDecision = z.object({
  accepted: z.boolean().describe("Whether the candidate passed validation acceptance gates"),
  reason: z.string().describe("Decision explanation"),
  checks: z.record(z.string(), z.unknown()).default({}).describe("Detailed gate checks"),
})
export type AdjustmentAcceptanceDecision = z.infer<typeof AdjustmentAcceptanceDecision>

export const AdjustmentApplicationSummary = z.object({
  matched_rows: z.number().int().nonnegative().describe("Unique prediction rows matched by at least one rule"),
  changed_rows: z.number().int().nonnegative().describe("Prediction rows whose final value differs from raw"),
  triggered_rule_count: z.number().int().nonnegative().describe("Rules that matched at least one row"),
  zero_trigger: z.boolean().describe("Whether no rule matched any validation row"),
  row_count: z.number().int().nonnegative().describe("Rows covered by the adjusted artifact"),
  metrics_reused_from_raw: z
    .boolean()
    .describe("Whether exact raw metrics were reused after proving a zero-trigger identity result"),
})
export type AdjustmentApplicationSummary = z.infer<typeof AdjustmentApplicationSummary>

export const AdjustmentEvalResult = z.object({
  run_id: z.string().describe("Raw source run id"),
  candidate_id: z.string().describe("Adjustment candidate or active skill id"),
  split: z.enum(["fit", "val", "test"]).describe("Evaluation split for the adjusted artifact"),
  raw_eval_path: z.string().nullable().describe("Path to raw eval.json when available"),
  adjusted_predictions_path: z.string().describe("Path to adjusted_predictions.csv"),
  raw_metrics: evalMetrics.nullable().describe("Raw metrics used as baseline"),
  adjusted_metrics: evalMetrics.describe("Adjusted metrics"),
  delta_vs_raw: z.record(z.string(), z.number().nullable()).describe("Adjusted metric minus raw metric"),
  application_summary: AdjustmentApplicationSummary.optional().describe("Audited rule trigger and change counts"),
  acceptance_decision: AdjustmentAcceptanceDecision.optional().describe("Validation acceptance decision"),
  timestamp: z.string().describe("ISO-8601 timestamp when adjusted evaluation completed"),
})
export type AdjustmentEvalResult = z.infer<typeof AdjustmentEvalResult>

export const AdjustmentSkillManifest = z.object({
  version: z.string().describe("Adjustment skill version"),
  status: z.enum(["active", "candidate", "rejected", "locked"]).describe("Skill status"),
  candidate_id: z.string().describe("Candidate id this skill was accepted from"),
  source_model_set_id: z.string().describe("Source model set used for fitting"),
  accepted_at: z.string().describe("ISO-8601 timestamp when accepted"),
  locked_at: z.string().nullable().default(null).describe("ISO-8601 timestamp when locked for Critic"),
  validation_results: z.array(z.string()).describe("Project-relative adjusted validation result paths"),
  policy_path: z.string().describe("Project-relative adjustment policy path"),
  parent_version: z.string().nullable().default(null),
})
export type AdjustmentSkillManifest = z.infer<typeof AdjustmentSkillManifest>

export const AdjustmentUserDecision = z.enum(["pending", "disabled", "enabled_a", "enabled_b"])
export type AdjustmentUserDecision = z.infer<typeof AdjustmentUserDecision>

export const AdjustmentTestApplicationDecision = z.enum(["pending", "apply", "skip"])
export type AdjustmentTestApplicationDecision = z.infer<typeof AdjustmentTestApplicationDecision>

export const AdjustmentTestApplicationRecord = z.object({
  test_run_id: z.string().describe("Post-forecast test run eligible for the locked adjustment skill"),
  validation_run_id: z.string().describe("Accepted validation run whose spec matches this test run"),
  model: z.string().nullable().default(null).describe("Model name read from the test spec"),
  decision: AdjustmentTestApplicationDecision.default("pending"),
  decision_at: z.string().nullable().default(null),
  decision_actor: AdjustmentDecisionActor.nullable().default(null),
  decision_reason: z.string().nullable().default(null),
  adjustment_locked: z.boolean().default(false),
  adjustment_candidate_id: z.string().nullable().default(null),
  adjustment_evaluated: z.boolean().default(false),
  adjusted_eval_path: z.string().nullable().default(null),
})
export type AdjustmentTestApplicationRecord = z.infer<typeof AdjustmentTestApplicationRecord>

export const ForecastPredictionPriorDecision = z.enum(["pending", "declined", "enabled"])
export type ForecastPredictionPriorDecision = z.infer<typeof ForecastPredictionPriorDecision>

export const ForecastPredictionPriorState = z.object({
  enabled: z.boolean().default(false),
  user_decision: ForecastPredictionPriorDecision.default("pending"),
  user_decision_at: z.string().nullable().default(null),
  user_decision_actor: AdjustmentDecisionActor.nullable().default(null),
  user_decision_reason: z.string().nullable().default(null),
  prior_dataset_path: z.string().nullable().default(null),
  prior_dataset_sha256: z.string().length(64).nullable().default(null),
  manifest_path: z.string().nullable().default(null),
  source_model_set_id: z.string().nullable().default(null),
  validation_run_id: z.string().nullable().default(null),
  validation_metrics: evalMetrics.nullable().default(null),
  fit_run_id: z.string().nullable().default(null),
  fit_metrics: evalMetrics.nullable().default(null),
  test_run_id: z.string().nullable().default(null),
  test_metrics: evalMetrics.nullable().default(null),
  test_materialized: z.boolean().default(false),
})
export type ForecastPredictionPriorState = z.infer<typeof ForecastPredictionPriorState>

export const ForecastAdjustmentState = z.object({
  enabled: z.boolean().default(false),
  mode: AdjustmentMode.default("off"),
  strategy: AdjustmentStrategy.nullable().default(null),
  user_decision: AdjustmentUserDecision.default("pending"),
  user_decision_at: z.string().nullable().default(null),
  user_decision_actor: AdjustmentDecisionActor.nullable().default(null),
  user_decision_reason: z.string().nullable().default(null),
  policy_path: z.string().nullable().default(null),
  source_model_set_locked: z.boolean().default(false),
  source_model_set_path: z.string().nullable().default(null),
  fit_run_manifest_path: z.string().nullable().default(null),
  fit_runs_ready: z.boolean().default(false),
  active_skill_version: z.string().nullable().default(null),
  active_skill_path: z.string().nullable().default(null),
  forecaster_locked: z.boolean().default(false),
  no_effective_skill: z.boolean().default(false),
  no_effective_skill_reason: z.string().nullable().default(null),
  test_collection_run_ids: z.array(z.string()).default([]),
  test_eligible_run_ids: z.array(z.string()).default([]),
  test_application_records: z.array(AdjustmentTestApplicationRecord).default([]),
  critic_user_confirmed: z.string().nullable().default(null),
  test_application_decision: AdjustmentTestApplicationDecision.default("pending"),
  test_application_decision_at: z.string().nullable().default(null),
  test_application_decision_actor: AdjustmentDecisionActor.nullable().default(null),
  test_application_decision_reason: z.string().nullable().default(null),
  test_application_run_id: z.string().nullable().default(null),
  test_adjustment_locked: z.boolean().default(false),
  test_adjustment_run_id: z.string().nullable().default(null),
  test_adjustment_candidate_id: z.string().nullable().default(null),
})
export type ForecastAdjustmentState = z.infer<typeof ForecastAdjustmentState>

export const ForecastReflectionValidationStatus = z.enum([
  "pending",
  "enabled_no_trigger",
  "enabled_improved",
  "enabled_unchanged",
  "disabled_negative_mse",
  "failed",
])
export type ForecastReflectionValidationStatus = z.infer<typeof ForecastReflectionValidationStatus>

export const ForecastWindowReflectionPolicy = z
  .object({
    controller_version: z.string(),
    enabled: z.boolean(),
    max_reflection_rounds: z.number().int().min(1).max(3),
    max_format_retries: z.number().int().min(0).max(2),
    trigger_policy: z.object({
      hard_violation_count: z.number().int().min(1),
      independent_soft_violation_count: z.number().int().min(2),
    }),
    constraints: z
      .object({
        nonnegative: z.boolean(),
        absolute_min: z.number().finite(),
        absolute_max: z.number().finite(),
        soft_absolute_min: z.number().finite().optional(),
        soft_absolute_max: z.number().finite().optional(),
        soft_abs_step: z.number().positive(),
        max_abs_step: z.number().positive(),
        soft_abs_curvature: z.number().positive().optional(),
        max_abs_curvature: z.number().positive().optional(),
        hard_min_window_range: z.number().nonnegative().optional(),
        min_window_range: z.number().nonnegative().optional(),
        window_mean_abs_delta_limit: z.number().positive(),
        point_emergency_delta_limit: z.number().positive(),
      })
      .refine((value) => value.absolute_min < value.absolute_max, "absolute_min must be below absolute_max")
      .refine((value) => value.soft_abs_step <= value.max_abs_step, "soft_abs_step must not exceed max_abs_step")
      .refine(
        (value) => value.window_mean_abs_delta_limit <= value.point_emergency_delta_limit,
        "window mean correction limit must not exceed the point emergency limit",
      )
      .refine(
        (value) =>
          value.soft_absolute_min === undefined ||
          value.soft_absolute_max === undefined ||
          (value.absolute_min <= value.soft_absolute_min &&
            value.soft_absolute_min < value.soft_absolute_max &&
            value.soft_absolute_max <= value.absolute_max),
        "soft absolute range must be ordered inside the hard absolute range",
      )
      .refine(
        (value) =>
          value.soft_abs_curvature === undefined ||
          value.max_abs_curvature === undefined ||
          value.soft_abs_curvature <= value.max_abs_curvature,
        "soft curvature limit must not exceed hard curvature limit",
      )
      .refine(
        (value) =>
          value.hard_min_window_range === undefined ||
          value.min_window_range === undefined ||
          value.hard_min_window_range <= value.min_window_range,
        "hard minimum window range must not exceed the soft minimum window range",
      ),
    calibration: z
      .object({
        method: z.literal("training_baseline_validation_front_calibration_v2"),
        source_split: z.literal("validation_front"),
        training_row_count: z.number().int().positive(),
        validation_front_row_count: z.number().int().positive(),
        validation_front_window_count: z.number().int().positive(),
        validation_back_used: z.literal(false),
        test_used: z.literal(false),
        validation_front_window_statistics: z.object({
          window_count: z.number().int().positive(),
          range: z.object({
            min: z.number().nonnegative(),
            q10: z.number().nonnegative(),
            q50: z.number().nonnegative(),
            q90: z.number().nonnegative(),
            q99: z.number().nonnegative(),
            max: z.number().nonnegative(),
          }),
          max_abs_step: z.object({
            min: z.number().positive(),
            q10: z.number().positive(),
            q50: z.number().positive(),
            q90: z.number().positive(),
            q99: z.number().positive(),
            max: z.number().positive(),
          }),
          max_abs_curvature: z.object({
            min: z.number().positive(),
            q10: z.number().positive(),
            q50: z.number().positive(),
            q90: z.number().positive(),
            q99: z.number().positive(),
            max: z.number().positive(),
          }),
        }),
        validation_front_reference_replay: z.object({
          max_reference_trigger_rate: z.number().min(0).max(1),
          initial_trigger_count: z.number().int().nonnegative(),
          initial_trigger_rate: z.number().min(0).max(1),
          final_trigger_count: z.number().int().nonnegative(),
          final_trigger_rate: z.number().min(0).max(1),
          hard_violation_window_count: z.literal(0),
          soft_threshold_adjustment_rounds: z.number().int().nonnegative().max(8),
        }),
        calibrated_rules: z.array(z.string()).min(1),
      })
      .optional(),
    validation_gate: z.object({
      metric: z.literal("mse"),
      accept_if: z.literal("final_mse <= baseline_mse"),
      zero_trigger: z.literal("enabled_no_trigger"),
    }),
  })
  .superRefine((value, ctx) => {
    if (value.controller_version !== "2") return
    if (!value.calibration) {
      ctx.addIssue({
        code: "custom",
        path: ["calibration"],
        message: "controller v2 requires validation-front calibration",
      })
    }
    for (const field of [
      "soft_absolute_min",
      "soft_absolute_max",
      "soft_abs_curvature",
      "max_abs_curvature",
      "hard_min_window_range",
      "min_window_range",
    ] as const) {
      if (value.constraints[field] === undefined) {
        ctx.addIssue({ code: "custom", path: ["constraints", field], message: `controller v2 requires ${field}` })
      }
    }
  })
export type ForecastWindowReflectionPolicy = z.infer<typeof ForecastWindowReflectionPolicy>

export const CriticPredictionEvidence = z.object({
  evidence_candidate_id: z.string().min(1).describe("Selected evidence-candidate identity"),
  candidate_type: z.enum(["experience", "context", "knowledge"]),
  artifact_path: z.string().min(1).describe("Path to the independently stored evidence candidate"),
  artifact_sha256: z.string().length(64).describe("Immutable identity of the selected evidence artifact"),
})
export type CriticPredictionEvidence = z.infer<typeof CriticPredictionEvidence>

export const CriticPredictionPoint = z.object({
  horizon_index: z.number().int().nonnegative(),
  timestamp: z.string(),
  value: z.number().finite().describe("Current prediction in original dataset units"),
})
export type CriticPredictionPoint = z.infer<typeof CriticPredictionPoint>

export const CriticPredictionState = z.object({
  version: z.literal(1),
  window_index: z.number().int().nonnegative(),
  evidences: z.array(CriticPredictionEvidence).describe("Evidence candidates already selected and used"),
  predictions: z.array(CriticPredictionPoint).min(1).describe("Current prediction values"),
  evidence_gap: z.union([z.literal(0), z.literal(1)]).describe("0 means sufficient; 1 means more evidence is needed"),
  needed_evidence: z
    .string()
    .default("")
    .describe("Natural-language description of evidence that may close the current evidence gap"),
})
export type CriticPredictionState = z.infer<typeof CriticPredictionState>

export const ForecastWindowReflectionArtifact = z.object({
  controller_version: z.string(),
  source_run_id: z.string(),
  split: z.enum(["val", "test"]),
  baseline_kind: z.enum(["raw_anchor", "skill_adjusted"]),
  adjustment_candidate_id: z.string().nullable().default(null),
  status: z.union([ForecastReflectionValidationStatus.exclude(["pending", "failed"]), z.literal("completed")]),
  trigger_count: z.number().int().nonnegative(),
  skill_match_count: z.number().int().nonnegative().default(0),
  llm_required_count: z.number().int().nonnegative().default(0),
  corrected_count: z.number().int().nonnegative(),
  fallback_count: z.number().int().nonnegative(),
  deterministic_fallback_count: z.number().int().nonnegative().default(0),
  raw_fallback_count: z.number().int().nonnegative().default(0),
  baseline_metrics: evalMetrics,
  final_metrics: evalMetrics,
  delta_mse: z.number(),
  final_predictions_path: z.string().min(1),
  final_predictions_sha256: z.string().length(64),
  leakage_audit_sha256: z.string().length(64),
  llm_protocol_path: z.string().min(1).nullable().default(null),
  llm_protocol_sha256: z.string().length(64).nullable().default(null),
  gap_predictions_sha256: z.string().length(64).nullable().default(null),
  completed_at: z.string(),
})
export type ForecastWindowReflectionArtifact = z.infer<typeof ForecastWindowReflectionArtifact>

export const ForecastReflectionValidationRecord = z.object({
  validation_run_id: z.string(),
  baseline_kind: z.enum(["raw_anchor", "skill_adjusted"]),
  adjustment_candidate_id: z.string().nullable().default(null),
  status: ForecastReflectionValidationStatus.default("pending"),
  trigger_count: z.number().int().nonnegative().default(0),
  corrected_count: z.number().int().nonnegative().default(0),
  fallback_count: z.number().int().nonnegative().default(0),
  baseline_metrics: evalMetrics.nullable().default(null),
  final_metrics: evalMetrics.nullable().default(null),
  delta_mse: z.number().nullable().default(null),
  result_path: z.string().nullable().default(null),
  completed_at: z.string().nullable().default(null),
})
export type ForecastReflectionValidationRecord = z.infer<typeof ForecastReflectionValidationRecord>

export const ForecastReflectionTestRecord = z.object({
  test_run_id: z.string(),
  validation_run_id: z.string(),
  baseline_kind: z.enum(["raw_anchor", "skill_adjusted"]),
  adjustment_candidate_id: z.string().nullable().default(null),
  validation_status: ForecastReflectionValidationStatus,
  llm_refinement_enabled: z.boolean(),
  trigger_count: z.number().int().nonnegative().default(0),
  skill_match_count: z.number().int().nonnegative().default(0),
  llm_required_count: z.number().int().nonnegative().default(0),
  corrected_count: z.number().int().nonnegative().default(0),
  fallback_count: z.number().int().nonnegative().default(0),
  deterministic_fallback_count: z.number().int().nonnegative().default(0),
  raw_fallback_count: z.number().int().nonnegative().default(0),
  baseline_metrics: evalMetrics.nullable().default(null),
  final_metrics: evalMetrics.nullable().default(null),
  result_path: z.string().nullable().default(null),
  completed_at: z.string().nullable().default(null),
})
export type ForecastReflectionTestRecord = z.infer<typeof ForecastReflectionTestRecord>

export const ForecastReflectionState = z.object({
  enabled: z.boolean().default(true),
  configured: z.boolean().default(false),
  policy_path: z.string().nullable().default(null),
  policy_sha256: z.string().length(64).nullable().default(null),
  controller_version: z.string().default("1"),
  max_reflection_rounds: z.number().int().min(1).max(3).default(3),
  max_format_retries: z.number().int().min(0).max(2).default(2),
  validation_records: z.array(ForecastReflectionValidationRecord).default([]),
  test_records: z.array(ForecastReflectionTestRecord).default([]),
})
export type ForecastReflectionState = z.infer<typeof ForecastReflectionState>

export const ForecastPhaseState = z.object({
  phase: ForecastPhase.describe("Current phase of the forecasting pipeline"),
  skills_confirmed: z.boolean().describe("Whether the user has confirmed the skill set"),
  skills_confirmed_at: z.string().nullable().describe("ISO-8601 timestamp of skill confirmation"),
  skill_memory_published_at: z
    .string()
    .nullable()
    .optional()
    .describe("ISO-8601 timestamp when final Forecast skills were promoted to global Forecast memory"),
  skill_memory_manifest_path: z
    .string()
    .nullable()
    .optional()
    .describe("Path to the global Forecast memory publish manifest for the final Forecast skills"),
  skill_memory_published_count: z
    .number()
    .int()
    .nonnegative()
    .nullable()
    .optional()
    .describe("Number of final Forecast skills promoted to global Forecast memory"),
  skill_memory_publish_error: z
    .string()
    .nullable()
    .optional()
    .describe("Best-effort global Forecast memory publish error, if automatic promotion failed"),
  pre_forecast_completed: z.boolean().describe("Whether the pre-forecast analysis is complete"),
  pre_forecast_completed_at: z.string().nullable().describe("ISO-8601 timestamp of pre-forecast completion"),
  task_frozen_at: z.string().nullable().describe("ISO-8601 timestamp of task.json freeze"),
  transitions: z.array(PhaseTransitionRecord).describe("Ordered log of all phase transitions"),
  auto_mode: ForecastAutoModeState.default({
    enabled: false,
    active_agent: null,
    state: "waiting",
    pause_reason: null,
    auto_approve_permissions: false,
    auto_approve_skills: false,
    handoffs: [],
  }).describe("Durable Auto Mode workflow state stored alongside the forecast phase machine"),
  consecutive_no_improvement: z
    .number()
    .int()
    .nonnegative()
    .default(0)
    .describe("Consecutive rounds where best MSE/MAE did not improve"),
  consecutive_crashes: z
    .number()
    .int()
    .nonnegative()
    .default(0)
    .describe("Consecutive experiment crashes (excludes ipc_error per Phase 2 D-13)"),
  stop_counter_last_run_id: z
    .string()
    .nullable()
    .default(null)
    .describe("Run id of the latest experiment attempt already incorporated into the stop counters"),
  stop_counter_last_attempt_index: z
    .number()
    .int()
    .positive()
    .nullable()
    .default(null)
    .describe("Attempt index paired with stop_counter_last_run_id for idempotent retry accounting"),
  stop_reason: z
    .enum(["no_improvement", "budget_exhausted", "crash_threshold", "user_stop"])
    .nullable()
    .default(null)
    .describe("Reason the experiment loop was terminated, or null if still running"),
  budget_initialized: z.boolean().default(false).describe("Whether budget thresholds have been frozen from CAST.md"),
  budget_thresholds: BudgetThresholds.optional().describe(
    "Frozen budget thresholds — set on first forecast read from CAST.md or defaults",
  ),
  adjustment: ForecastAdjustmentState.default({
    enabled: false,
    mode: "off",
    strategy: null,
    user_decision: "pending",
    user_decision_at: null,
    user_decision_actor: null,
    user_decision_reason: null,
    policy_path: null,
    source_model_set_locked: false,
    source_model_set_path: null,
    fit_run_manifest_path: null,
    fit_runs_ready: false,
    active_skill_version: null,
    active_skill_path: null,
    forecaster_locked: false,
    no_effective_skill: false,
    no_effective_skill_reason: null,
    test_collection_run_ids: [],
    test_eligible_run_ids: [],
    test_application_records: [],
    critic_user_confirmed: null,
    test_application_decision: "pending",
    test_application_decision_at: null,
    test_application_decision_actor: null,
    test_application_decision_reason: null,
    test_application_run_id: null,
    test_adjustment_locked: false,
    test_adjustment_run_id: null,
    test_adjustment_candidate_id: null,
  }).describe("Durable adjustment workflow state"),
  reflection: ForecastReflectionState.default({
    enabled: true,
    configured: false,
    policy_path: null,
    policy_sha256: null,
    controller_version: "1",
    max_reflection_rounds: 3,
    max_format_retries: 2,
    validation_records: [],
    test_records: [],
  }).describe("Independent deterministic window checks and isolated LLM refinement state"),
  prediction_prior: ForecastPredictionPriorState.default({
    enabled: false,
    user_decision: "pending",
    user_decision_at: null,
    user_decision_actor: null,
    user_decision_reason: null,
    prior_dataset_path: null,
    prior_dataset_sha256: null,
    manifest_path: null,
    source_model_set_id: null,
    validation_run_id: null,
    validation_metrics: null,
    fit_run_id: null,
    fit_metrics: null,
    test_run_id: null,
    test_metrics: null,
    test_materialized: false,
  }).describe("Optional user-submitted prediction-prior workflow state"),
})
export type ForecastPhaseState = z.infer<typeof ForecastPhaseState>

export const ForecastContext = z.object({
  generated_at: z.string().describe("ISO-8601 timestamp when this context was generated"),
  constraints: z.string().describe("CAST.md constraint text, or empty string if no CAST.md exists"),
  pre_forecast_summary: z.string().describe("Structured conclusions extracted from pre-forecast.md"),
  modeling_directions: z.string().describe("Recommended model families and approaches from pre-forecast analysis"),
})
export type ForecastContext = z.infer<typeof ForecastContext>
