import { access, readFile } from "fs/promises"
import matter from "gray-matter"
import path from "path"
import { BestModelManifest, ExperimentRecord, ForecastPhaseState, TaskConfig } from "./types"
import type {
  BestModelManifest as BestModelManifestValue,
  ExperimentRecord as ExperimentRecordValue,
  ForecastPhaseState as ForecastPhaseStateValue,
  TaskConfig as TaskConfigValue,
} from "./types"
import { normalizeForecastStateFrontmatter } from "./forecast-progress"
import { deduplicateExperimentsByRunId } from "./forecast-history"

export type ForecastWorkspaceStatus = {
  exists: boolean
  path: string
}

export type ForecastWorkflowStatus = {
  phase: ForecastPhaseStateValue["phase"] | "unknown"
  auto_mode: {
    enabled: boolean | "unknown"
    active_agent: ForecastPhaseStateValue["auto_mode"]["active_agent"] | "unknown"
    state: ForecastPhaseStateValue["auto_mode"]["state"] | "unknown"
  }
  stop_reason: ForecastPhaseStateValue["stop_reason"] | "unknown"
}

export type ForecastTaskStatus = {
  dataset: string | null
  target: string | null
  horizon: number | null
  forecast_gap: number | null
  model_horizon: number | null
  model_families: string[]
}

export type ForecastExperimentsStatus = {
  completed: number
  success: number
  failed: number
  crash: number
  timeout: number
  ipc_error: number
  budget: {
    used: number
    max: number | null
    remaining: number | null
  }
}

export type ForecastBestStatus = {
  ready: boolean
  source: "best.json" | "history" | "manifest" | null
  model: string | null
  run_id: string | null
  mse: number | null
  mae: number | null
}

export type ForecastOutputsStatus = {
  best_model_package: "ready" | "pending" | "invalid"
  final_report: "ready" | "pending"
  visualizations: {
    window_level: "ready" | "pending"
    trajectory: "ready" | "pending"
  }
}

export type ForecastAdjustmentStatus = {
  enabled: boolean | "unknown"
  mode: ForecastPhaseStateValue["adjustment"]["mode"] | "unknown"
  strategy: ForecastPhaseStateValue["adjustment"]["strategy"] | "unknown"
  user_decision: ForecastPhaseStateValue["adjustment"]["user_decision"] | "unknown"
  user_decision_actor: ForecastPhaseStateValue["adjustment"]["user_decision_actor"] | "unknown"
  user_decision_reason: string | null | "unknown"
  source_model_set_locked: boolean | "unknown"
  fit_runs_ready: boolean | "unknown"
  active_skill_version: string | null | "unknown"
  forecaster_locked: boolean | "unknown"
  no_effective_skill: boolean | "unknown"
  test_application_decision: ForecastPhaseStateValue["adjustment"]["test_application_decision"] | "unknown"
  test_application_decision_actor: ForecastPhaseStateValue["adjustment"]["test_application_decision_actor"] | "unknown"
  test_application_decision_reason: string | null | "unknown"
  test_application_run_id: string | null | "unknown"
  test_adjustment_locked: boolean | "unknown"
  test_collection_count: number | "unknown"
  test_eligible_count: number | "unknown"
  test_application_records: ForecastPhaseStateValue["adjustment"]["test_application_records"] | "unknown"
}

export type ForecastStatus = {
  workspace: ForecastWorkspaceStatus
  workflow: ForecastWorkflowStatus
  task: ForecastTaskStatus
  experiments: ForecastExperimentsStatus
  validation_best: ForecastBestStatus
  test_best: ForecastBestStatus
  adjustment: ForecastAdjustmentStatus
  outputs: ForecastOutputsStatus
  warnings: string[]
  next_step: string
}

type HistoryParseResult = {
  experiments: ExperimentRecordValue[]
  warnings: string[]
}

const FINAL_REPORT_PLACEHOLDER = "_Pending: Critic agent will populate this report._"

async function exists(file: string): Promise<boolean> {
  return access(file)
    .then(() => true)
    .catch(() => false)
}

async function readOptionalText(file: string): Promise<string | null> {
  return readFile(file, "utf-8").catch(() => null)
}

function emptyBest(): ForecastBestStatus {
  return {
    ready: false,
    source: null,
    model: null,
    run_id: null,
    mse: null,
    mae: null,
  }
}

function bestFromExperiment(
  experiment: ExperimentRecordValue | null,
  source: ForecastBestStatus["source"],
): ForecastBestStatus {
  if (!experiment) return emptyBest()

  return {
    ready: true,
    source,
    model: experiment.model,
    run_id: experiment.run_id,
    mse: experiment.metrics?.mse ?? null,
    mae: experiment.metrics?.mae ?? null,
  }
}

function bestFromManifest(manifest: BestModelManifestValue | null): ForecastBestStatus {
  if (!manifest) return emptyBest()

  return {
    ready: true,
    source: "manifest",
    model: manifest.model,
    run_id: manifest.test_run_id,
    mse: typeof manifest.test_metrics.mse === "number" ? manifest.test_metrics.mse : null,
    mae: typeof manifest.test_metrics.mae === "number" ? manifest.test_metrics.mae : null,
  }
}

function chooseBestByMse(experiments: ExperimentRecordValue[]): ExperimentRecordValue | null {
  return experiments.reduce<ExperimentRecordValue | null>((best, experiment) => {
    if (experiment.status !== "success") return best
    const mse = experiment.metrics?.mse
    if (typeof mse !== "number" || !Number.isFinite(mse)) return best
    if (!best) return experiment
    const bestMse = best.metrics?.mse
    if (typeof bestMse !== "number" || mse < bestMse) return experiment
    return best
  }, null)
}

async function readState(forecastRoot: string, warnings: string[]): Promise<ForecastPhaseStateValue | null> {
  const raw = await readOptionalText(path.join(forecastRoot, "STATE.md"))
  if (!raw) return null

  try {
    const parsed = matter(raw)
    const data = normalizeForecastStateFrontmatter(parsed.data as Record<string, unknown>)
    const state = ForecastPhaseState.safeParse(data)
    if (state.success) return state.data
    warnings.push(`Invalid .forecast/STATE.md frontmatter: ${state.error.issues[0]?.message ?? "schema mismatch"}`)
  } catch (error) {
    warnings.push(`Could not read .forecast/STATE.md: ${error instanceof Error ? error.message : String(error)}`)
  }

  return null
}

async function readTask(forecastRoot: string, warnings: string[]): Promise<TaskConfigValue | null> {
  const raw = await readOptionalText(path.join(forecastRoot, "task.json"))
  if (!raw) return null

  try {
    const parsed = TaskConfig.safeParse(JSON.parse(raw))
    if (parsed.success) return parsed.data
    warnings.push(`Invalid .forecast/task.json: ${parsed.error.issues[0]?.message ?? "schema mismatch"}`)
  } catch (error) {
    warnings.push(`Could not parse .forecast/task.json: ${error instanceof Error ? error.message : String(error)}`)
  }

  return null
}

async function readBest(forecastRoot: string, warnings: string[]): Promise<ExperimentRecordValue | null> {
  const raw = await readOptionalText(path.join(forecastRoot, "best.json"))
  if (!raw) return null

  try {
    const parsed = ExperimentRecord.safeParse(JSON.parse(raw))
    if (parsed.success) return parsed.data
    warnings.push(`Invalid .forecast/best.json: ${parsed.error.issues[0]?.message ?? "schema mismatch"}`)
  } catch (error) {
    warnings.push(`Could not parse .forecast/best.json: ${error instanceof Error ? error.message : String(error)}`)
  }

  return null
}

async function readManifest(forecastRoot: string, warnings: string[]): Promise<BestModelManifestValue | null> {
  const raw = await readOptionalText(path.join(forecastRoot, "best-model", "manifest.json"))
  if (!raw) return null

  try {
    const parsed = BestModelManifest.safeParse(JSON.parse(raw))
    if (parsed.success) return parsed.data
    warnings.push(`Invalid .forecast/best-model/manifest.json: ${parsed.error.issues[0]?.message ?? "schema mismatch"}`)
  } catch (error) {
    warnings.push(
      `Could not parse .forecast/best-model/manifest.json: ${error instanceof Error ? error.message : String(error)}`,
    )
  }

  return null
}

async function readHistory(forecastRoot: string): Promise<HistoryParseResult> {
  const raw = await readOptionalText(path.join(forecastRoot, "history.jsonl"))
  const result: HistoryParseResult = { experiments: [], warnings: [] }
  if (!raw) return result

  raw.split("\n").forEach((line, index) => {
    if (!line.trim()) return

    try {
      const parsed = JSON.parse(line)
      const experiment = ExperimentRecord.safeParse(parsed)
      if (experiment.success) {
        result.experiments.push(experiment.data)
        return
      }

      if (!parsed || typeof parsed !== "object" || !("type" in parsed)) {
        result.warnings.push(`Ignored invalid ExperimentRecord in .forecast/history.jsonl line ${index + 1}`)
      }
    } catch (error) {
      result.warnings.push(
        `Could not parse .forecast/history.jsonl line ${index + 1}: ${
          error instanceof Error ? error.message : String(error)
        }`,
      )
    }
  })

  result.experiments = deduplicateExperimentsByRunId(result.experiments)
  return result
}

async function reportReady(forecastRoot: string): Promise<boolean> {
  const raw = await readOptionalText(path.join(forecastRoot, "reports", "final-report.md"))
  return raw !== null && raw.trim() !== "" && !raw.includes(FINAL_REPORT_PLACEHOLDER)
}

function deriveNextStep(input: {
  workspaceExists: boolean
  phase: ForecastWorkflowStatus["phase"]
  hasBestModel: boolean
  hasFinalReport: boolean
}): string {
  if (!input.workspaceExists) return "Start the forecast workflow."

  if (input.phase === "init") return "Define the forecast task."
  if (input.phase === "pre-forecast") return "Complete the pre-forecast report."
  if (input.phase === "skills-review") return "Confirm skills and enter forecast."
  if (input.phase === "forecast") return "Continue experiments or let Forecaster wrap up."
  if (input.phase === "post-forecast" && !input.hasBestModel) return "Let Critic complete Distillation."
  if (input.hasBestModel && !input.hasFinalReport) return "Let Critic write the final report."
  if (input.hasBestModel && input.hasFinalReport) return "Review the final report and best-model package."

  return "Read .forecast/STATE.md and continue the next legal forecast phase."
}

export async function readForecastStatus(directory: string = process.cwd()): Promise<ForecastStatus> {
  const forecastRoot = path.join(directory, ".forecast")
  const workspaceExists = await exists(forecastRoot)
  const warnings: string[] = []

  if (!workspaceExists) {
    return {
      workspace: { exists: false, path: forecastRoot },
      workflow: {
        phase: "unknown",
        auto_mode: { enabled: "unknown", active_agent: "unknown", state: "unknown" },
        stop_reason: "unknown",
      },
      task: {
        dataset: null,
        target: null,
        horizon: null,
        forecast_gap: null,
        model_horizon: null,
        model_families: [],
      },
      experiments: {
        completed: 0,
        success: 0,
        failed: 0,
        crash: 0,
        timeout: 0,
        ipc_error: 0,
        budget: { used: 0, max: null, remaining: null },
      },
      validation_best: emptyBest(),
      test_best: emptyBest(),
      adjustment: {
        enabled: "unknown",
        mode: "unknown",
        strategy: "unknown",
        user_decision: "unknown",
        user_decision_actor: "unknown",
        user_decision_reason: "unknown",
        source_model_set_locked: "unknown",
        fit_runs_ready: "unknown",
        active_skill_version: "unknown",
        forecaster_locked: "unknown",
        no_effective_skill: "unknown",
        test_application_decision: "unknown",
        test_application_decision_actor: "unknown",
        test_application_decision_reason: "unknown",
        test_application_run_id: "unknown",
        test_adjustment_locked: "unknown",
        test_collection_count: "unknown",
        test_eligible_count: "unknown",
        test_application_records: "unknown",
      },
      outputs: {
        best_model_package: "pending",
        final_report: "pending",
        visualizations: { window_level: "pending", trajectory: "pending" },
      },
      warnings,
      next_step: deriveNextStep({
        workspaceExists: false,
        phase: "unknown",
        hasBestModel: false,
        hasFinalReport: false,
      }),
    }
  }

  const [state, task, best, history, manifest, hasFinalReport, hasWindowViz, hasTrajectoryViz, hasManifestFile] =
    await Promise.all([
      readState(forecastRoot, warnings),
      readTask(forecastRoot, warnings),
      readBest(forecastRoot, warnings),
      readHistory(forecastRoot),
      readManifest(forecastRoot, warnings),
      reportReady(forecastRoot),
      exists(path.join(forecastRoot, "viz", "window-level.png")),
      exists(path.join(forecastRoot, "viz", "trajectory.png")),
      exists(path.join(forecastRoot, "best-model", "manifest.json")),
    ])

  warnings.push(...history.warnings)

  const success = history.experiments.filter((experiment) => experiment.status === "success").length
  const crash = history.experiments.filter((experiment) => experiment.status === "crash").length
  const timeout = history.experiments.filter((experiment) => experiment.status === "timeout").length
  const ipcError = history.experiments.filter((experiment) => experiment.status === "ipc_error").length
  const maxBudget = state?.budget_thresholds?.max_experiments ?? null
  const budgetUsed = history.experiments.length
  const packageState = manifest ? "ready" : hasManifestFile ? "invalid" : "pending"
  const phase = state?.phase ?? "unknown"

  return {
    workspace: { exists: true, path: forecastRoot },
    workflow: {
      phase,
      auto_mode: {
        enabled: state?.auto_mode.enabled ?? "unknown",
        active_agent: state?.auto_mode.active_agent ?? (state ? null : "unknown"),
        state: state?.auto_mode.state ?? "unknown",
      },
      stop_reason: state?.stop_reason ?? (state ? null : "unknown"),
    },
    task: {
      dataset: task?.dataset_path ?? null,
      target: task?.target_col ?? null,
      horizon: task?.pred_len ?? null,
      forecast_gap: task?.forecast_gap ?? null,
      model_horizon: task ? task.forecast_gap + task.pred_len : null,
      model_families: task?.model_families ?? [],
    },
    experiments: {
      completed: history.experiments.length,
      success,
      failed: crash + timeout + ipcError,
      crash,
      timeout,
      ipc_error: ipcError,
      budget: {
        used: budgetUsed,
        max: maxBudget,
        remaining: maxBudget === null ? null : Math.max(maxBudget - budgetUsed, 0),
      },
    },
    validation_best: bestFromExperiment(best ?? chooseBestByMse(history.experiments), best ? "best.json" : "history"),
    test_best: bestFromManifest(manifest),
    adjustment: {
      enabled: state?.adjustment.enabled ?? "unknown",
      mode: state?.adjustment.mode ?? "unknown",
      strategy: state?.adjustment.strategy ?? (state ? null : "unknown"),
      user_decision: state?.adjustment.user_decision ?? "unknown",
      user_decision_actor: state?.adjustment.user_decision_actor ?? (state ? null : "unknown"),
      user_decision_reason: state?.adjustment.user_decision_reason ?? (state ? null : "unknown"),
      source_model_set_locked: state?.adjustment.source_model_set_locked ?? "unknown",
      fit_runs_ready: state?.adjustment.fit_runs_ready ?? "unknown",
      active_skill_version: state?.adjustment.active_skill_version ?? (state ? null : "unknown"),
      forecaster_locked: state?.adjustment.forecaster_locked ?? "unknown",
      no_effective_skill: state?.adjustment.no_effective_skill ?? "unknown",
      test_application_decision: state?.adjustment.test_application_decision ?? "unknown",
      test_application_decision_actor: state?.adjustment.test_application_decision_actor ?? (state ? null : "unknown"),
      test_application_decision_reason:
        state?.adjustment.test_application_decision_reason ?? (state ? null : "unknown"),
      test_application_run_id: state?.adjustment.test_application_run_id ?? "unknown",
      test_adjustment_locked: state?.adjustment.test_adjustment_locked ?? "unknown",
      test_collection_count: state ? state.adjustment.test_collection_run_ids.length : "unknown",
      test_eligible_count: state ? state.adjustment.test_eligible_run_ids.length : "unknown",
      test_application_records: state?.adjustment.test_application_records ?? "unknown",
    },
    outputs: {
      best_model_package: packageState,
      final_report: hasFinalReport ? "ready" : "pending",
      visualizations: {
        window_level: hasWindowViz ? "ready" : "pending",
        trajectory: hasTrajectoryViz ? "ready" : "pending",
      },
    },
    warnings,
    next_step: deriveNextStep({
      workspaceExists: true,
      phase,
      hasBestModel: packageState === "ready",
      hasFinalReport,
    }),
  }
}

function formatValue(value: string | number | boolean | null): string {
  if (value === null) return "unknown"
  return String(value)
}

function formatMetric(best: ForecastBestStatus): string {
  if (!best.ready) return "pending"
  const metrics = [`MSE ${formatValue(best.mse)}`, `MAE ${formatValue(best.mae)}`]
  return `${formatValue(best.model)} (${formatValue(best.run_id)}), ${metrics.join(", ")}`
}

export function formatForecastStatus(status: ForecastStatus): string {
  const budget =
    status.experiments.budget.max === null
      ? `${status.experiments.budget.used}/unknown`
      : `${status.experiments.budget.used}/${status.experiments.budget.max} (${status.experiments.budget.remaining} remaining)`

  const lines = [
    "Workflow",
    `  phase: ${status.workflow.phase}`,
    `  auto mode: ${formatValue(status.workflow.auto_mode.enabled)}`,
    `  active agent: ${formatValue(status.workflow.auto_mode.active_agent)}`,
    `  stop reason: ${formatValue(status.workflow.stop_reason)}`,
    "",
    "Task",
    `  dataset: ${formatValue(status.task.dataset)}`,
    `  target: ${formatValue(status.task.target)}`,
    `  horizon: ${formatValue(status.task.horizon)}`,
    `  forecast_gap: ${formatValue(status.task.forecast_gap)}`,
    `  model_horizon: ${formatValue(status.task.model_horizon)}`,
    `  model families: ${status.task.model_families.length ? status.task.model_families.join(", ") : "unknown"}`,
    "",
    "Experiments",
    `  completed: ${status.experiments.completed}`,
    `  success: ${status.experiments.success}`,
    `  failed: ${status.experiments.failed}`,
    `  budget: ${budget}`,
    "",
    "Validation Best",
    `  ${formatMetric(status.validation_best)}`,
    "",
    "Generative prediction reasoning",
    `  enabled: ${formatValue(status.adjustment.enabled)}`,
    `  mode: ${status.adjustment.mode}`,
    `  strategy: ${formatValue(status.adjustment.strategy)}`,
    `  start decision: ${formatValue(status.adjustment.user_decision)}`,
    `  start decision actor: ${formatValue(status.adjustment.user_decision_actor)}`,
    `  start decision reason: ${formatValue(status.adjustment.user_decision_reason)}`,
    `  source model set locked: ${formatValue(status.adjustment.source_model_set_locked)}`,
    `  fit runs ready: ${formatValue(status.adjustment.fit_runs_ready)}`,
    `  active skill: ${formatValue(status.adjustment.active_skill_version)}`,
    `  forecaster locked: ${formatValue(status.adjustment.forecaster_locked)}`,
    `  no effective skill: ${formatValue(status.adjustment.no_effective_skill)}`,
    `  test application decision: ${formatValue(status.adjustment.test_application_decision)}`,
    `  test application actor: ${formatValue(status.adjustment.test_application_decision_actor)}`,
    `  test application reason: ${formatValue(status.adjustment.test_application_decision_reason)}`,
    `  test application run: ${formatValue(status.adjustment.test_application_run_id)}`,
    `  test adjustment locked: ${formatValue(status.adjustment.test_adjustment_locked)}`,
    `  test collection models: ${formatValue(status.adjustment.test_collection_count)}`,
    `  reasoning-eligible test models: ${formatValue(status.adjustment.test_eligible_count)}`,
    `  per-model test application records: ${
      Array.isArray(status.adjustment.test_application_records)
        ? status.adjustment.test_application_records
            .map(
              (record) =>
                `${record.model ?? "unknown"}:${record.test_run_id}=${record.decision}/${record.adjustment_evaluated ? "evaluated" : record.adjustment_locked ? "locked" : "not-run"}`,
            )
            .join(", ") || "none"
        : status.adjustment.test_application_records
    }`,
    "",
    "Critic Outputs",
    `  best model package: ${status.outputs.best_model_package}`,
    `  final report: ${status.outputs.final_report}`,
    `  visualizations: window-level ${status.outputs.visualizations.window_level}, trajectory ${status.outputs.visualizations.trajectory}`,
    "",
    "Test Best",
    `  ${formatMetric(status.test_best)}`,
    "",
    "Next Step",
    `  ${status.next_step}`,
  ]

  if (status.warnings.length) {
    lines.push("", "Warnings", ...status.warnings.map((warning) => `  ${warning}`))
  }

  return `${lines.join("\n")}\n`
}
