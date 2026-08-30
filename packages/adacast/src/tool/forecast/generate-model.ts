import { appendFile, readFile } from "fs/promises"
import path from "path"
import { createInterface } from "node:readline"
import matter from "gray-matter"
import z from "zod"
import { Instance } from "@/project/instance"
import { Filesystem } from "@/util/filesystem"
import { Process } from "@/util/process"
import { Tool } from "../tool"
import { IPCError, TaskNotFoundError } from "./errors"
import {
  normalizeGenerateModelExecutionContract,
  normalizeValidationExecutionContract,
  validationSpecMatchesPostForecastTestSpec,
} from "./execution-contract"
import DESCRIPTION from "./generate-model.txt"
import { normalizeForecastStateFrontmatter } from "./forecast-progress"
import { deduplicateExperimentsByRunId, experimentAttemptIndex } from "./forecast-history"
import { computeSpecHash } from "./hash"
import { inferForecastModelFamily, isStatisticalForecastModel } from "./model-family"
import { readBoundModelArtifactIdentity, type ModelArtifactIdentity } from "./model-artifact"
import { resolveForecastModelEnvironment } from "./model-environment"
import { resolveForecastPythonProject } from "./python-project"
import { assertFrozenTaskIntegrity } from "./forecast-task-integrity"
import {
  BudgetThresholds,
  CacheSummary,
  CompactSummary,
  EvalResult,
  ExperimentRecord as ExperimentRecordSchema,
  ExpertFeedbackRecord as ExpertFeedbackRecordSchema,
  ForecastPhaseState,
  ReflectionRecord as ReflectionRecordSchema,
  RunnerOutput,
  StderrEvent,
  TaskConfig,
  WindowFeatures,
} from "./types"

const DEFAULT_TIMEOUT_MS = 30 * 60 * 1000
const SIGKILL_GRACE_MS = 10_000
const FOUNDATION_MODEL_NAMES = new Set(["Chronos", "Chronos2", "TimesFM", "Moirai", "Sundial", "TiRex", "TimeMoE"])
const FOUNDATION_MODEL_NAME_BY_CASEFOLD = new Map(
  [...FOUNDATION_MODEL_NAMES].map((name) => [name.toLocaleLowerCase("en-US"), name]),
)

function isFoundationModelName(model: string) {
  return FOUNDATION_MODEL_NAME_BY_CASEFOLD.has(model.toLocaleLowerCase("en-US"))
}

type RunnerOutputValue = z.infer<typeof RunnerOutput>
type CompactSummaryValue = z.infer<typeof CompactSummary>
type CacheSummaryValue = z.infer<typeof CacheSummary>
type CachedCompactSummaryValue = CompactSummaryValue & { status: "cached" | "crash"; cached: true }
type FreshCompactSummaryValue = CompactSummaryValue & { status: "success" | "crash"; cached: false }
type EvalResultValue = z.infer<typeof EvalResult>
export type StderrEventValue = z.infer<typeof StderrEvent>
type ForecastContext = {
  current_best: { run_id: string; model: string; metrics: Record<string, number | null> } | null
  recent_failures: Array<{ run_id: string; model: string; failure_reason: string; decision?: string }>
  budget: { remaining: number; total: number; consecutive_no_improvement: number; consecutive_crashes: number }
  expert_feedback: { text: string; timestamp: string } | null
}

type ReuseSource = {
  runId: string
  runDir: string
  artifact: ModelArtifactIdentity
}

type GenerateModelMetadata = {
  runId: string
  status: "running" | "success" | "cached" | "crash" | "timeout" | "ipc_error"
  reuseModelFromRunId?: string
  cached?: boolean
  reason?: string
  exitCode?: number
  progress?: StderrEventValue
  cache_summary?: CacheSummaryValue
  forecast_context?: ForecastContext | null
}

const parameters = z.object({
  spec: z
    .record(z.string(), z.unknown())
    .describe(
      "Experiment spec object containing at minimum: model (string), pred_len (number), seq_len (number), label_len (number), and optionally hyperparams, dataset_path, etc.",
    ),
  reuse_model_from_run_id: z
    .string()
    .optional()
    .describe(
      "Forecast-phase validation run id whose final fitted model must be reused for post-forecast test inference without retraining",
    ),
})

async function readForecastContext(directory: string): Promise<ForecastContext | null> {
  try {
    const forecastRoot = path.join(directory, ".forecast")

    let noImprovement = 0
    let crashes = 0
    let thresholds = BudgetThresholds.parse({})

    try {
      const stateRaw = await readFile(path.join(forecastRoot, "STATE.md"), "utf-8")
      const parsed = ForecastPhaseState.parse(
        normalizeForecastStateFrontmatter(matter(stateRaw).data as Record<string, unknown>),
      )
      noImprovement = parsed.consecutive_no_improvement ?? 0
      crashes = parsed.consecutive_crashes ?? 0
      if (parsed.budget_thresholds) {
        thresholds = BudgetThresholds.parse(parsed.budget_thresholds)
      }
    } catch {
      // Use defaults when the state file or budget thresholds are not available yet.
    }

    let best: ForecastContext["current_best"] = null
    try {
      const bestRaw = JSON.parse(await readFile(path.join(forecastRoot, "best.json"), "utf-8"))
      const bestParsed = ExperimentRecordSchema.safeParse(bestRaw)
      if (bestParsed.success) {
        best = {
          run_id: bestParsed.data.run_id,
          model: bestParsed.data.model,
          metrics: {
            mse: bestParsed.data.metrics?.mse ?? null,
            mae: bestParsed.data.metrics?.mae ?? null,
            wape: bestParsed.data.metrics?.wape ?? null,
            mase: bestParsed.data.metrics?.mase ?? null,
          },
        }
      }
    } catch {
      // No best result has been recorded yet.
    }

    const historyRaw = await readFile(path.join(forecastRoot, "history.jsonl"), "utf-8").catch(() => "")
    const experiments: Array<{
      run_id: string
      attempt_index?: number
      model: string
      status: string
      failure_reason?: string
      decision?: string
    }> = []
    let latestFeedback: ForecastContext["expert_feedback"] = null

    for (const line of historyRaw.split("\n").filter(Boolean)) {
      try {
        const parsed = JSON.parse(line)
        const feedback = ExpertFeedbackRecordSchema.safeParse(parsed)
        if (feedback.success) {
          latestFeedback = { text: feedback.data.text, timestamp: feedback.data.timestamp }
          continue
        }

        const reflection = ReflectionRecordSchema.safeParse(parsed)
        if (reflection.success) {
          continue
        }

        const experiment = ExperimentRecordSchema.safeParse(parsed)
        if (experiment.success) {
          experiments.push({
            run_id: experiment.data.run_id,
            attempt_index: experiment.data.attempt_index,
            model: experiment.data.model,
            status: experiment.data.status,
            failure_reason: experiment.data.failure_reason,
            decision: experiment.data.decision,
          })
        }
      } catch {
        // Skip invalid lines so prompt metadata remains resilient to partial history corruption.
      }
    }

    const uniqueExperiments = deduplicateExperimentsByRunId(experiments)
    const recentFailures = uniqueExperiments
      .filter((record) => record.status !== "success" || record.decision === "discard")
      .slice(-5)
      .map((record) => ({
        run_id: record.run_id,
        model: record.model,
        failure_reason: record.failure_reason?.trim() || record.status,
        decision: record.decision,
      }))

    return {
      current_best: best,
      recent_failures: recentFailures,
      budget: {
        remaining: Math.max(0, thresholds.max_experiments - uniqueExperiments.length),
        total: thresholds.max_experiments,
        consecutive_no_improvement: noImprovement,
        consecutive_crashes: crashes,
      },
      expert_feedback: latestFeedback,
    }
  } catch {
    return null
  }
}

async function assertRawForecastSearchCanStart(
  directory: string,
  spec: Record<string, unknown>,
  runId: string,
): Promise<void> {
  if (spec.phase !== "forecast" || spec.eval_split !== "val") return

  const forecastRoot = path.join(directory, ".forecast")
  let thresholds = BudgetThresholds.parse({})
  try {
    const stateRaw = await readFile(path.join(forecastRoot, "STATE.md"), "utf-8")
    const state = ForecastPhaseState.parse(
      normalizeForecastStateFrontmatter(matter(stateRaw).data as Record<string, unknown>),
    )
    thresholds = state.budget_thresholds ?? thresholds
  } catch {
    // Use defaults when the phase state is temporarily unavailable.
  }

  const historyRaw = await readFile(path.join(forecastRoot, "history.jsonl"), "utf-8").catch(() => "")
  const attempts = historyRaw
    .split("\n")
    .filter(Boolean)
    .flatMap((line) => {
      try {
        const parsed = ExperimentRecordSchema.safeParse(JSON.parse(line))
        return parsed.success ? [parsed.data] : []
      } catch {
        return []
      }
    })
    .filter((record) => (record.spec.phase ?? "forecast") === "forecast" && (record.spec.eval_split ?? "val") === "val")
  const records = deduplicateExperimentsByRunId(attempts)
  const isRetry = attempts.some((record) => record.run_id === runId)

  if (!isRetry && records.length >= thresholds.max_experiments) {
    throw new Error(
      `Forecast experiment budget exhausted: ${records.length}/${thresholds.max_experiments} unique validation runs`,
    )
  }

  let trailingCrashes = 0
  for (const record of attempts.toReversed()) {
    if (record.status !== "crash") break
    trailingCrashes += 1
  }
  if (trailingCrashes >= thresholds.crash_threshold) {
    throw new Error(
      `Forecast crash threshold reached: ${trailingCrashes}/${thresholds.crash_threshold} consecutive crashes`,
    )
  }
}

function buildCompactSummary(
  result: RunnerOutputValue,
  model: string,
  elapsed: number,
  cached: true,
): CachedCompactSummaryValue
function buildCompactSummary(
  result: RunnerOutputValue,
  model: string,
  elapsed: number,
  cached: false,
): FreshCompactSummaryValue
function buildCompactSummary(
  result: RunnerOutputValue,
  model: string,
  elapsed: number,
  cached: boolean,
): CompactSummaryValue {
  if (result.status === "error") {
    return {
      model,
      status: "crash",
      run_id: result.run_id,
      metrics: null,
      training_duration_s: elapsed,
      cached,
    }
  }

  return {
    model,
    status: cached ? "cached" : "success",
    run_id: result.run_id,
    metrics: result.metrics ?? null,
    training_duration_s: elapsed,
    cached,
  }
}

function isCacheDiagnosticEvent(event: StderrEventValue): boolean {
  return (
    event.type === "cache_config" ||
    event.type === "cache_hit" ||
    event.type === "cache_miss" ||
    event.type === "cache_warning" ||
    event.type === "dependency_missing"
  )
}

export function summarizeCacheDiagnostics(events: StderrEventValue[], logPath?: string): CacheSummaryValue | undefined {
  if (!events.some(isCacheDiagnosticEvent)) {
    return undefined
  }

  let cacheConfig: StderrEventValue | undefined
  const warnings: CacheSummaryValue["warnings"] = []
  const dependenciesMissing: CacheSummaryValue["dependencies_missing"] = []
  let hits = 0
  let misses = 0

  for (const event of events) {
    if (event.type === "cache_config") {
      cacheConfig = event
      for (const warning of event.warnings ?? []) {
        warnings.push({ code: warning.code, message: warning.message })
      }
    } else if (event.type === "cache_hit") {
      hits += 1
    } else if (event.type === "cache_miss") {
      misses += 1
    } else if (event.type === "cache_warning") {
      warnings.push({
        code: event.code ?? "cache_warning",
        message: event.message ?? "Cache warning emitted without message.",
      })
    } else if (event.type === "dependency_missing") {
      dependenciesMissing.push({
        module: event.module ?? "unknown",
        package: event.package ?? event.module ?? "unknown",
        guidance: event.guidance ?? "Install the missing Python dependency and retry the run.",
      })
    }
  }

  return CacheSummary.parse({
    source: cacheConfig?.source,
    endpoint: cacheConfig?.endpoint,
    cache_root: cacheConfig?.cache_root,
    hits,
    misses,
    warning_count: warnings.length,
    dependency_missing_count: dependenciesMissing.length,
    warnings,
    dependencies_missing: dependenciesMissing,
    log_path: logPath,
  })
}

async function appendExperimentRecord(input: {
  directory: string
  runId: string
  model: string
  spec: Record<string, unknown>
  status: "success" | "crash" | "timeout" | "ipc_error"
  metrics: CompactSummaryValue["metrics"]
  elapsedSeconds: number
  failureReason?: string
  windowFeatures?: z.infer<typeof WindowFeatures>
}) {
  const historyPath = path.join(input.directory, ".forecast", "history.jsonl")
  const historyRaw = await readFile(historyPath, "utf-8").catch(() => "")
  const priorAttemptIndex = historyRaw
    .split("\n")
    .filter(Boolean)
    .flatMap((line) => {
      try {
        const parsed = ExperimentRecordSchema.safeParse(JSON.parse(line))
        return parsed.success && parsed.data.run_id === input.runId ? [experimentAttemptIndex(parsed.data)] : []
      } catch {
        return []
      }
    })
    .reduce((maximum, value) => Math.max(maximum, value), 0)
  const record = ExperimentRecordSchema.parse({
    run_id: input.runId,
    attempt_index: priorAttemptIndex + 1,
    model: input.model,
    status: input.status,
    metrics: input.metrics,
    spec: input.spec,
    window_features: input.windowFeatures,
    failure_reason: input.failureReason,
    elapsed_seconds: input.elapsedSeconds,
    timestamp: new Date().toISOString(),
  })
  await appendFile(historyPath, `${JSON.stringify(record)}\n`, "utf-8")
}

function extractWindowFeatures(
  spec: Record<string, unknown>,
  model: string,
): z.infer<typeof WindowFeatures> | undefined {
  const inferredFamily = inferForecastModelFamily(model)
  const raw = spec.window_features
  if (!raw || typeof raw !== "object") {
    return inferredFamily ? { model_family: inferredFamily } : undefined
  }
  const parsed = WindowFeatures.safeParse(raw)
  if (!parsed.success) return inferredFamily ? { model_family: inferredFamily } : undefined
  return {
    ...parsed.data,
    model_family: parsed.data.model_family ?? inferredFamily,
  }
}

async function hasCompletePredictionArtifacts(
  runDir: string,
  runId: string,
  spec: Record<string, unknown>,
  reuseSource?: ReuseSource | null,
): Promise<boolean> {
  const required = ["spec.json", "eval.json", "predictions.csv", "actual.npy", "pred.npy", "train_scale.json"]
  const checks = await Promise.all(required.map((file) => Filesystem.exists(path.join(runDir, file))))
  if (!checks.every(Boolean)) return false

  try {
    const storedSpec = await Filesystem.readJson<Record<string, unknown>>(path.join(runDir, "spec.json"))
    const requiresExplicitLayout =
      Number(spec.input_contract_version ?? 1) >= 3 || isStatisticalForecastModel(String(spec.model ?? ""))
    if (requiresExplicitLayout) {
      const header =
        (await readFile(path.join(runDir, "predictions.csv"), "utf-8")).split(/\r?\n/, 1)[0]?.split(",") ?? []
      if (
        !header.includes("prediction_layout") ||
        !header.includes("window_index") ||
        !header.includes("horizon_index")
      ) {
        return false
      }
    }
    const cachedEval = EvalResult.parse(await Filesystem.readJson<EvalResultValue>(path.join(runDir, "eval.json")))
    const artifact = await readBoundModelArtifactIdentity(runDir, String(spec.model ?? ""), cachedEval)
    return (
      computeSpecHash(storedSpec) === computeSpecHash(spec) &&
      cachedEval.run_id === runId &&
      cachedEval.spec_hash === runId &&
      (!reuseSource ||
        (cachedEval.artifact_binding?.model_artifact.source_run_id === reuseSource.runId &&
          artifact.kind === reuseSource.artifact.kind &&
          artifact.sha256 === reuseSource.artifact.sha256))
    )
  } catch {
    return false
  }
}

function runIdIsSafe(runId: string): boolean {
  return /^[A-Za-z0-9_-]+$/.test(runId)
}

async function resolveTestReuseSource(
  spec: Record<string, unknown>,
  reuseModelFromRunId: string | undefined,
): Promise<ReuseSource | null> {
  const isPostForecastTest = spec.phase === "post-forecast" && spec.eval_split === "test"
  if (!isPostForecastTest) {
    if (reuseModelFromRunId) {
      throw new Error("reuse_model_from_run_id is only valid for post-forecast test inference")
    }
    return null
  }
  if (!reuseModelFromRunId) {
    throw new Error(
      "Post-forecast test inference requires reuse_model_from_run_id; retraining after validation selection is forbidden",
    )
  }
  if (!runIdIsSafe(reuseModelFromRunId)) {
    throw new Error("reuse_model_from_run_id contains unsafe characters")
  }

  const runDir = path.join(Instance.directory, ".forecast", "runs", reuseModelFromRunId)
  const sourceSpecPath = path.join(runDir, "spec.json")
  const sourceEvalPath = path.join(runDir, "eval.json")
  if (!(await Filesystem.exists(sourceSpecPath)) || !(await Filesystem.exists(sourceEvalPath))) {
    throw new Error("Reusable validation run is missing spec.json or eval.json")
  }

  const sourceSpec = await Filesystem.readJson<Record<string, unknown>>(sourceSpecPath)
  const sourceEval = EvalResult.parse(await Filesystem.readJson<EvalResultValue>(sourceEvalPath))
  const sourceSpecHash = computeSpecHash(sourceSpec)
  let normalizedSourceSpec
  try {
    normalizedSourceSpec = normalizeValidationExecutionContract(sourceSpec)
  } catch {
    throw new Error("Reusable model source is not a self-consistent forecast-phase validation run")
  }
  if (
    sourceEval.eval_split !== "val" ||
    sourceEval.run_id !== reuseModelFromRunId ||
    (sourceEval.spec_hash !== reuseModelFromRunId && sourceEval.spec_hash !== sourceSpecHash)
  ) {
    throw new Error("Reusable model source is not a self-consistent forecast-phase validation run")
  }

  if (!validationSpecMatchesPostForecastTestSpec(normalizedSourceSpec, spec)) {
    throw new Error(
      "Post-forecast test spec must exactly match the reusable validation spec except for phase and eval_split",
    )
  }

  const artifact = await readBoundModelArtifactIdentity(runDir, String(normalizedSourceSpec.model ?? ""), sourceEval)
  return { runId: reuseModelFromRunId, runDir, artifact }
}

export const GenerateModelTool = Tool.define<typeof parameters, GenerateModelMetadata>("generate_model", {
  description: DESCRIPTION,
  parameters,
  async execute(params, ctx) {
    const taskPath = path.join(Instance.directory, ".forecast", "task.json")
    if (!(await Filesystem.exists(taskPath))) {
      throw new TaskNotFoundError({ expected_path: taskPath })
    }
    await assertFrozenTaskIntegrity(path.dirname(taskPath))

    const taskRaw = await Filesystem.readJson<Record<string, unknown>>(taskPath)
    const task = TaskConfig.parse(taskRaw)
    const merged: Record<string, unknown> = {
      dataset_path: task.dataset_path,
      target_col: task.target_col,
      time_col: task.time_col,
      pred_len: task.pred_len,
      forecast_gap: task.forecast_gap,
      seq_len: task.seq_len,
      label_len: task.label_len,
      metrics: task.eval_metrics,
      phase: "forecast",
      eval_split: "val",
      ...params.spec,
      adjustment_fit_holdout_ratio: task.adjustment_fit_holdout_ratio,
    }
    delete merged.window_stride
    if (Object.hasOwn(taskRaw, "window_stride")) merged.window_stride = task.window_stride
    delete merged.forecast_gap
    if (Object.hasOwn(taskRaw, "forecast_gap")) merged.forecast_gap = task.forecast_gap
    merged.train_ratio = task.train_ratio
    merged.val_ratio = task.val_ratio
    merged.test_ratio = task.test_ratio
    delete merged.split_mode
    delete merged.train_end_timestamp
    delete merged.validation_end_timestamp
    if (task.split_mode) merged.split_mode = task.split_mode
    if (task.train_end_timestamp) merged.train_end_timestamp = task.train_end_timestamp
    if (task.validation_end_timestamp) merged.validation_end_timestamp = task.validation_end_timestamp
    if (merged.phase === "forecast" && merged.eval_split === "val") {
      merged.input_contract_version = isStatisticalForecastModel(String(merged.model ?? "")) ? 4 : 3
    }
    const normalized = normalizeGenerateModelExecutionContract(merged)

    // Extract window_features from the merged spec before sending to the Python runner.
    // window_features is agent metadata — it must not be passed to the runner or
    // included in the spec hash (it is stored in the history record separately).
    const model = String(normalized.model ?? "unknown")
    const windowFeatures = extractWindowFeatures(normalized, model)
    const { window_features: _extractedWf, ...spec } = normalized
    const reuseSource = await resolveTestReuseSource(spec, params.reuse_model_from_run_id)

    const runId = computeSpecHash(spec)
    const runsDir = path.join(Instance.directory, ".forecast", "runs")
    const runDir = path.join(runsDir, runId)
    const relativeTrainLogPath = path.join(".forecast", "runs", runId, "train.log")
    const evalPath = path.join(runDir, "eval.json")

    if (await hasCompletePredictionArtifacts(runDir, runId, spec, reuseSource)) {
      const cached = EvalResult.parse(await Filesystem.readJson<EvalResultValue>(evalPath))
      const summary = buildCompactSummary(
        { status: "cached", run_id: runId, out_dir: runDir, metrics: cached.metrics },
        model,
        0,
        true,
      )
      const forecastCtx = await readForecastContext(Instance.directory)
      return {
        title: `Cached: ${model}`,
        metadata: {
          runId,
          cached: true,
          status: "cached",
          reuseModelFromRunId: reuseSource?.runId,
          forecast_context: forecastCtx,
        },
        output: JSON.stringify(summary),
      }
    }

    await assertRawForecastSearchCanStart(Instance.directory, spec, runId)

    const specPath = path.join(runDir, "spec.json")
    await Filesystem.write(specPath, JSON.stringify(spec, null, 2))

    const pythonDir = await resolveForecastPythonProject(Instance.directory)
    const forecastDir = path.join(Instance.directory, ".forecast")
    const modelEnvironment = resolveForecastModelEnvironment({ directory: Instance.directory, env: process.env })
    const runnerCommand = [
      "uv",
      "run",
      "--project",
      pythonDir,
      ...(isFoundationModelName(model) ? ["--extra", "heavy-models"] : []),
      "python",
      "-m",
      "adacast_ml.runner",
      "--spec",
      specPath,
      "--run-id",
      runId,
      "--out-dir",
      runsDir,
      "--evaluator-hash-store",
      path.join(forecastDir, "evaluator.sha256"),
      ...(reuseSource ? ["--reuse-model-from", reuseSource.runDir] : []),
    ]
    const startTime = Date.now()
    const proc = Process.spawn(runnerCommand, {
      cwd: Instance.directory,
      env: modelEnvironment.env,
      stdout: "pipe",
      stderr: "pipe",
      abort: ctx.abort,
      kill: "SIGTERM",
      timeout: SIGKILL_GRACE_MS,
    })

    ctx.metadata({
      title: reuseSource ? `Evaluating ${model}` : `Training ${model}`,
      metadata: { runId, status: "running", reuseModelFromRunId: reuseSource?.runId },
    })

    const stderrLines: string[] = []
    const stderrEvents: StderrEventValue[] = []
    const stderrInterface = proc.stderr ? createInterface({ input: proc.stderr }) : null
    stderrInterface?.on(
      "line",
      Instance.bind((line: string) => {
        stderrLines.push(line)
        try {
          const event = StderrEvent.parse(JSON.parse(line))
          stderrEvents.push(event)
          ctx.metadata({
            title: `Training ${model}`,
            metadata: { runId, status: "running", progress: event },
          })
        } catch {
          // Ignore non-JSON stderr lines and anything that is not a progress event.
        }
      }),
    )

    const stdoutChunks: Buffer[] = []
    proc.stdout?.on(
      "data",
      Instance.bind((chunk: Buffer | string) => {
        stdoutChunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
      }),
    )

    let timedOut = false
    let sigkillTimer: ReturnType<typeof setTimeout> | undefined
    const timeoutTimer = setTimeout(() => {
      timedOut = true
      proc.kill("SIGTERM")
      sigkillTimer = setTimeout(() => {
        try {
          proc.kill("SIGKILL")
        } catch {
          // Process already exited.
        }
      }, SIGKILL_GRACE_MS)
    }, DEFAULT_TIMEOUT_MS)

    const exitCode = await proc.exited
      .catch(() => 1)
      .finally(() => {
        clearTimeout(timeoutTimer)
        if (sigkillTimer) clearTimeout(sigkillTimer)
        stderrInterface?.close()
      })
    const elapsed = (Date.now() - startTime) / 1000

    if (timedOut) {
      const cacheSummary = summarizeCacheDiagnostics(stderrEvents, relativeTrainLogPath)
      const summary = CompactSummary.parse({
        model,
        status: "timeout",
        run_id: runId,
        metrics: null,
        training_duration_s: elapsed,
        cached: false,
        cache_summary: cacheSummary,
      })
      await appendExperimentRecord({
        directory: Instance.directory,
        runId,
        model,
        spec,
        status: "timeout",
        metrics: summary.metrics,
        elapsedSeconds: elapsed,
        failureReason: "timeout",
        windowFeatures,
      })
      const forecastCtx = await readForecastContext(Instance.directory)
      return {
        title: `Timeout: ${model}`,
        metadata: { runId, status: "timeout", cache_summary: cacheSummary, forecast_context: forecastCtx },
        output: JSON.stringify(summary),
      }
    }

    const stdoutText = Buffer.concat(stdoutChunks).toString("utf-8").trim()
    let result: RunnerOutputValue
    try {
      result = RunnerOutput.parse(JSON.parse(stdoutText))
    } catch {
      const ipcError = new IPCError({
        reason: exitCode === 0 ? "runner_stdout_invalid" : "runner_exited_without_valid_json",
        stderr_tail: stderrLines.slice(-10).join("\n") || undefined,
      })
      const cacheSummary = summarizeCacheDiagnostics(stderrEvents, relativeTrainLogPath)
      const summary = CompactSummary.parse({
        model,
        status: "ipc_error",
        run_id: runId,
        metrics: null,
        training_duration_s: elapsed,
        cached: false,
        cache_summary: cacheSummary,
      })
      await appendExperimentRecord({
        directory: Instance.directory,
        runId,
        model,
        spec,
        status: "ipc_error",
        metrics: summary.metrics,
        elapsedSeconds: elapsed,
        failureReason: ipcError.data.reason,
        windowFeatures,
      })
      const forecastCtx = await readForecastContext(Instance.directory)
      return {
        title: `IPC Error: ${model}`,
        metadata: {
          runId,
          status: "ipc_error",
          reason: ipcError.data.reason,
          cache_summary: cacheSummary,
          forecast_context: forecastCtx,
        },
        output: JSON.stringify(summary),
      }
    }

    if (!/^[A-Fa-f0-9]{64}$/.test(result.run_id)) {
      const ipcError = new IPCError({
        reason: "runner_returned_invalid_run_id",
        stderr_tail: result.run_id,
      })
      const cacheSummary = summarizeCacheDiagnostics(stderrEvents, relativeTrainLogPath)
      const summary = CompactSummary.parse({
        model,
        status: "ipc_error",
        run_id: runId,
        metrics: null,
        training_duration_s: elapsed,
        cached: false,
        cache_summary: cacheSummary,
      })
      await appendExperimentRecord({
        directory: Instance.directory,
        runId,
        model,
        spec,
        status: "ipc_error",
        metrics: summary.metrics,
        elapsedSeconds: elapsed,
        failureReason: ipcError.data.reason,
        windowFeatures,
      })
      const forecastCtx = await readForecastContext(Instance.directory)
      return {
        title: `IPC Error: ${model}`,
        metadata: {
          runId,
          status: "ipc_error",
          reason: ipcError.data.reason,
          cache_summary: cacheSummary,
          forecast_context: forecastCtx,
        },
        output: JSON.stringify(summary),
      }
    }

    const actualRunId = result.run_id
    const actualTrainLogPath = path.join(".forecast", "runs", actualRunId, "train.log")
    const cacheSummary = summarizeCacheDiagnostics(stderrEvents, actualTrainLogPath)
    const actualRunDir = path.join(runsDir, actualRunId)
    if (
      result.status !== "error" &&
      !(await hasCompletePredictionArtifacts(actualRunDir, actualRunId, spec, reuseSource))
    ) {
      const failureReason = "runner_completed_without_self-consistent_prediction_artifacts"
      const summary = CompactSummary.parse({
        model,
        status: "ipc_error",
        run_id: actualRunId,
        metrics: null,
        training_duration_s: elapsed,
        cached: false,
        cache_summary: cacheSummary,
      })
      await appendExperimentRecord({
        directory: Instance.directory,
        runId: actualRunId,
        model,
        spec,
        status: "ipc_error",
        metrics: summary.metrics,
        elapsedSeconds: elapsed,
        failureReason,
        windowFeatures,
      })
      const forecastCtx = await readForecastContext(Instance.directory)
      return {
        title: `IPC Error: ${model}`,
        metadata: {
          runId: actualRunId,
          status: "ipc_error",
          reason: failureReason,
          exitCode,
          cache_summary: cacheSummary,
          forecast_context: forecastCtx,
        },
        output: JSON.stringify(summary),
      }
    }

    const runnerReturnedCached = result.status === "cached"
    let summary: CompactSummaryValue
    if (runnerReturnedCached) {
      summary = CompactSummary.parse({
        ...buildCompactSummary(result, model, elapsed, true),
        cache_summary: cacheSummary,
      })
    } else {
      const freshSummary = CompactSummary.parse({
        ...buildCompactSummary(result, model, elapsed, false),
        cache_summary: cacheSummary,
      }) as FreshCompactSummaryValue
      summary = freshSummary
      await appendExperimentRecord({
        directory: Instance.directory,
        runId: actualRunId,
        model,
        spec,
        status: freshSummary.status,
        metrics: freshSummary.metrics,
        elapsedSeconds: elapsed,
        failureReason: result.status === "error" ? (result.error ?? result.error_type ?? "runner_error") : undefined,
        windowFeatures,
      })
    }
    const forecastCtx = await readForecastContext(Instance.directory)
    return {
      title: runnerReturnedCached ? `Cached: ${model}` : model,
      metadata: {
        runId: actualRunId,
        status: summary.status,
        cached: runnerReturnedCached,
        reuseModelFromRunId: reuseSource?.runId,
        exitCode,
        cache_summary: cacheSummary,
        forecast_context: forecastCtx,
      },
      output: JSON.stringify(summary),
    }
  },
})
