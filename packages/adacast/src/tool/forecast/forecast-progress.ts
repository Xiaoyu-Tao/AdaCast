import { readdir, readFile } from "fs/promises"
import matter from "gray-matter"
import path from "path"
import { deduplicateExperimentsByRunId } from "./forecast-history"
import { readBoundModelArtifactIdentity } from "./model-artifact"
import {
  BestModelManifest as BestModelManifestSchema,
  BudgetThresholds as BudgetThresholdsSchema,
  EvalResult as EvalResultSchema,
  ExperimentRecord as ExperimentRecordSchema,
  ExpertFeedbackRecord as ExpertFeedbackRecordSchema,
  ForecastPhaseState as ForecastPhaseStateSchema,
  ReflectionRecord as ReflectionRecordSchema,
} from "./types"
import type { ExperimentRecord, ExpertFeedbackRecord, ForecastPhaseState, ReflectionRecord } from "./types"

export type ProgressStatus = "pending" | "active" | "complete" | "warning" | "stopped"
export type ProgressAgent = "planner" | "forecaster" | "critic"
export type ProgressStep = { label: string; status: ProgressStatus; detail?: string }
export type StageRow = {
  id: ProgressAgent
  label: "Planner" | "Forecaster" | "Critic"
  status: ProgressStatus
  active: boolean
  detail: string
  steps: ProgressStep[]
}
export type ForecastLiveToolEvent = {
  tool:
    | "generate_model"
    | "evaluate_experiment"
    | "export_best_model"
    | "forecast_state"
    | "bash"
    | "write"
    | "forecast_adjustment_policy"
    | "forecast_adjustment_decision"
    | "forecast_adjustment_source_models"
    | "forecast_adjustment_fit_runs"
    | "forecast_adjustment_candidate"
    | "forecast_adjustment_apply"
    | "forecast_adjustment_evaluate"
    | "forecast_adjustment_skill"
    | "forecast_prediction_prior"
    | "forecast_window_reflection"
  stage?:
    | "forecast"
    | "forecast-adjustment"
    | "critic-evaluate"
    | "critic-adjustment"
    | "critic-reflection"
    | "critic-viz"
    | "critic-distill"
    | "critic-report"
  status: string
  runId?: string
  progressType?: string
  title?: string
}
export type ForecastToolPartLike = {
  type?: string
  tool?: string
  callID?: string
  state?: {
    status?: string
    title?: string
    input?: Record<string, unknown>
    output?: string
    error?: string
    metadata?: Record<string, unknown>
    time?: { start?: number; end?: number }
  }
  metadata?: Record<string, unknown>
}
export type ForecastProgressSnapshot = {
  modeLabel: "Auto Mode" | "Manual" | "Auto Mode Off"
  activeAgent: ProgressAgent | null
  workflowState: string
  pauseReason: string | null
  summary: string
  rows: StageRow[]
}
export type ForecastProgressInput = {
  state: ForecastPhaseState | null
  history: Array<ExperimentRecord | ReflectionRecord | ExpertFeedbackRecord>
  best: ExperimentRecord | null
  artifacts: {
    hasTask: boolean
    hasPreForecastReport: boolean
    testEvaluationCount: number
    hasWindowViz: boolean
    hasTrajectoryViz: boolean
    hasBestModelExport: boolean
    finalReportState: "missing" | "placeholder" | "written"
    historicalExplanationsState: "missing" | "placeholder" | "written"
  }
  liveEvents?: ForecastLiveToolEvent[]
}

const DEFAULT_EXPERIMENT_BUDGET = 20
const FINAL_REPORT_PLACEHOLDER = "_Pending: Critic agent will populate this report._"
const HISTORICAL_EXPLANATIONS_PLACEHOLDER =
  "_Pending: Critic agent will search historical training data for similar patterns to explain predictions._"

const AGENT_LABELS: Record<ProgressAgent, StageRow["label"]> = {
  planner: "Planner",
  forecaster: "Forecaster",
  critic: "Critic",
}
const FORECAST_TOOL_NAMES = new Set<ForecastLiveToolEvent["tool"]>([
  "generate_model",
  "evaluate_experiment",
  "export_best_model",
  "forecast_state",
  "bash",
  "write",
  "forecast_adjustment_policy",
  "forecast_adjustment_decision",
  "forecast_adjustment_source_models",
  "forecast_adjustment_fit_runs",
  "forecast_adjustment_candidate",
  "forecast_adjustment_apply",
  "forecast_adjustment_evaluate",
  "forecast_adjustment_skill",
  "forecast_prediction_prior",
  "forecast_window_reflection",
])
const CRITIC_LIVE_STAGE_LABELS: Record<
  Extract<
    ForecastLiveToolEvent["stage"],
    "critic-evaluate" | "critic-adjustment" | "critic-reflection" | "critic-viz" | "critic-distill" | "critic-report"
  >,
  string
> = {
  "critic-evaluate": "Evaluating raw test set",
  "critic-adjustment": "Applying generative prediction reasoning",
  "critic-reflection": "Running window reflection",
  "critic-viz": "Generating visualizations",
  "critic-distill": "Distillation",
  "critic-report": "Writing final report",
}

const DEFAULT_ARTIFACTS: ForecastProgressInput["artifacts"] = {
  hasTask: false,
  hasPreForecastReport: false,
  testEvaluationCount: 0,
  hasWindowViz: false,
  hasTrajectoryViz: false,
  hasBestModelExport: false,
  finalReportState: "missing",
  historicalExplanationsState: "missing",
}

function step(label: string, status: ProgressStatus, detail?: string): ProgressStep {
  return detail ? { label, status, detail } : { label, status }
}

function titleCaseAgent(agent: ProgressAgent): StageRow["label"] {
  return AGENT_LABELS[agent]
}

function workflowStateLabel(value: string): string {
  return value.replace(/[-_]+/g, " ").toLowerCase()
}

function phaseActiveAgent(state: ForecastPhaseState): ProgressAgent | null {
  if (state.phase === "forecast") return "forecaster"
  if (state.phase === "post-forecast") return "critic"
  return "planner"
}

function isExperimentRecord(
  entry: ExperimentRecord | ReflectionRecord | ExpertFeedbackRecord,
): entry is ExperimentRecord {
  return !("type" in entry)
}

function firstActiveStep(steps: ProgressStep[]): ProgressStep | undefined {
  return steps.find((item) => item.status === "active" || item.status === "warning" || item.status === "stopped")
}

function isForecastToolName(tool: string | undefined): tool is ForecastLiveToolEvent["tool"] {
  return Boolean(tool && FORECAST_TOOL_NAMES.has(tool as ForecastLiveToolEvent["tool"]))
}

function recordStringValue(record: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = record?.[key]
  return typeof value === "string" ? value : undefined
}

function progressTypeFromMetadata(metadata: Record<string, unknown>): string | undefined {
  const progress = metadata.progress
  if (!progress || typeof progress !== "object") return undefined

  const type = (progress as Record<string, unknown>).type
  return typeof type === "string" ? type : undefined
}

function includesForecastVizReference(value: string | undefined): boolean {
  if (!value) return false
  return value.includes("window_viz") || value.includes("trajectory_viz") || value.includes(".forecast/viz")
}

function liveStageForPart(part: ForecastToolPartLike): ForecastLiveToolEvent["stage"] | undefined {
  if (part.tool === "generate_model") return "forecast"
  if (part.tool === "evaluate_experiment") return "critic-evaluate"
  if (part.tool === "export_best_model") return "critic-distill"
  if (part.tool === "forecast_window_reflection") {
    const subcommand = recordStringValue(part.state?.input, "subcommand")
    return subcommand === "run_test" || subcommand === "resume_test" ? "critic-reflection" : "forecast-adjustment"
  }
  if (
    part.tool === "forecast_adjustment_policy" ||
    part.tool === "forecast_adjustment_decision" ||
    part.tool === "forecast_adjustment_source_models" ||
    part.tool === "forecast_adjustment_fit_runs" ||
    part.tool === "forecast_adjustment_candidate" ||
    part.tool === "forecast_adjustment_apply" ||
    part.tool === "forecast_adjustment_evaluate" ||
    part.tool === "forecast_adjustment_skill" ||
    part.tool === "forecast_prediction_prior"
  ) {
    const subcommand = recordStringValue(part.state?.input, "subcommand")
    if (
      (part.tool === "forecast_adjustment_policy" && subcommand === "confirm_for_critic") ||
      (part.tool === "forecast_adjustment_decision" &&
        (subcommand === "prepare_test_applications" ||
          subcommand === "ask_test_application" ||
          subcommand === "auto_test_application"))
    ) {
      return "critic-adjustment"
    }
    const split = recordStringValue(part.state?.input, "split") ?? recordStringValue(part.state?.metadata, "split")
    return split === "test" ? "critic-adjustment" : "forecast-adjustment"
  }

  if (part.tool === "bash") {
    if (
      includesForecastVizReference(part.state?.title) ||
      includesForecastVizReference(recordStringValue(part.state?.input, "command")) ||
      includesForecastVizReference(part.state?.output)
    ) {
      return "critic-viz"
    }
  }

  if (part.tool === "write") {
    const outputPath = recordStringValue(part.state?.input, "path")
    if (outputPath?.includes(".forecast/reports/final-report.md")) {
      return "critic-report"
    }
  }

  return undefined
}

export function extractForecastLiveEvents(parts: ForecastToolPartLike[]): ForecastLiveToolEvent[] {
  return parts
    .flatMap((part) => {
      if (part.type !== "tool" || !isForecastToolName(part.tool)) return []

      const metadata = part.state?.metadata ?? part.metadata ?? {}
      const status = recordStringValue(metadata, "status") ?? part.state?.status
      if (!status) return []

      const runId = recordStringValue(metadata, "runId") ?? recordStringValue(metadata, "run_id")
      const progressType = progressTypeFromMetadata(metadata)
      const stage = liveStageForPart(part)
      const event: ForecastLiveToolEvent = {
        tool: part.tool,
        status,
        ...(stage ? { stage } : {}),
        ...(runId ? { runId } : {}),
        ...(progressType ? { progressType } : {}),
        ...(part.state?.title ? { title: part.state.title } : {}),
      }

      return [{ event, start: part.state?.time?.start ?? 0 }]
    })
    .sort((a, b) => b.start - a.start)
    .map(({ event }) => event)
}

function appendActivePauseReason(input: {
  detail: string
  active: boolean
  workflowState: string
  pauseReason: string | null
}): string {
  if (!input.active || input.workflowState !== "paused" || !input.pauseReason) return input.detail
  return `${input.detail}, ${input.pauseReason}`
}

function readStopReason(reason: NonNullable<ForecastPhaseState["stop_reason"]>): string {
  return reason.replace(/_/g, " ")
}

function derivePlannerRow(input: {
  state: ForecastPhaseState
  artifacts: ForecastProgressInput["artifacts"]
  activeAgent: ProgressAgent | null
  workflowState: string
  pauseReason: string | null
}): StageRow {
  const { state, artifacts, activeAgent, workflowState, pauseReason } = input
  const clarifyingComplete = Boolean(state.task_frozen_at || artifacts.hasTask)
  const preForecastComplete = Boolean(state.pre_forecast_completed || artifacts.hasPreForecastReport)
  const skillsComplete = state.skills_confirmed

  const steps = [
    step("Clarifying", clarifyingComplete ? "complete" : "active"),
    step("Task frozen", clarifyingComplete ? "complete" : "pending"),
    step("Pre-forecast", preForecastComplete ? "complete" : state.phase === "pre-forecast" ? "active" : "pending"),
    step("Skills confirmed", skillsComplete ? "complete" : state.phase === "skills-review" ? "active" : "pending"),
  ]

  const active = activeAgent === "planner"
  const complete =
    steps.every((item) => item.status === "complete") || state.phase === "forecast" || state.phase === "post-forecast"
  const paused = active && workflowState === "paused" && Boolean(pauseReason)
  const currentStep = firstActiveStep(steps)
  const detail = appendActivePauseReason({
    detail: complete ? "Planner complete" : (currentStep?.label ?? "Waiting to start"),
    active,
    workflowState,
    pauseReason,
  })

  return {
    id: "planner",
    label: "Planner",
    status: paused ? "warning" : active ? "active" : complete ? "complete" : currentStep ? "active" : "pending",
    active,
    detail,
    steps,
  }
}

function deriveForecasterRow(input: {
  state: ForecastPhaseState
  history: Array<ExperimentRecord | ReflectionRecord | ExpertFeedbackRecord>
  best: ExperimentRecord | null
  liveEvents: ForecastLiveToolEvent[] | undefined
  activeAgent: ProgressAgent | null
  workflowState: string
  pauseReason: string | null
}): StageRow {
  const { state, history, best, liveEvents, activeAgent, workflowState, pauseReason } = input
  const completed = deduplicateExperimentsByRunId(history.filter(isExperimentRecord)).length
  const total = state.budget_thresholds?.max_experiments ?? DEFAULT_EXPERIMENT_BUDGET
  const remaining = Math.max(0, total - completed)
  const bestMetric = formatBestMetric(best)
  const runningForecastEvent =
    state.phase === "forecast"
      ? liveEvents?.find(
          (event) => event.tool === "generate_model" && event.stage === "forecast" && event.status === "running",
        )
      : undefined
  const active = activeAgent === "forecaster" || Boolean(runningForecastEvent)
  const paused = active && workflowState === "paused" && Boolean(pauseReason)

  let detail = `${completed}/${total} experiments, ${remaining} remaining`
  if (bestMetric) detail += `, best ${bestMetric}`
  if (state.stop_reason) detail += `, ${readStopReason(state.stop_reason)}`
  if (runningForecastEvent?.progressType) detail += `, running ${runningForecastEvent.progressType}`
  detail = appendActivePauseReason({ detail, active, workflowState, pauseReason })

  let status: ProgressStatus = "pending"
  if (
    state.stop_reason === "budget_exhausted" ||
    state.stop_reason === "crash_threshold" ||
    state.stop_reason === "user_stop"
  ) {
    status = "stopped"
  } else if (state.stop_reason === "no_improvement" || paused) {
    status = "warning"
  } else if (state.phase === "post-forecast") {
    status = "complete"
  } else if (state.phase === "forecast" || active) {
    status = "active"
  }

  return {
    id: "forecaster",
    label: "Forecaster",
    status,
    active,
    detail,
    steps: [],
  }
}

function hasLiveStage(
  events: ForecastLiveToolEvent[] | undefined,
  stage: NonNullable<ForecastLiveToolEvent["stage"]>,
  status?: string,
) {
  return events?.some((event) => event.stage === stage && (!status || event.status === status)) ?? false
}

function activeCriticLiveStage(
  events: ForecastLiveToolEvent[] | undefined,
): keyof typeof CRITIC_LIVE_STAGE_LABELS | undefined {
  return events?.find((event) => {
    if (event.status !== "running") return false
    return (
      event.stage === "critic-evaluate" ||
      event.stage === "critic-adjustment" ||
      event.stage === "critic-reflection" ||
      event.stage === "critic-viz" ||
      event.stage === "critic-distill" ||
      event.stage === "critic-report"
    )
  })?.stage as keyof typeof CRITIC_LIVE_STAGE_LABELS | undefined
}

function deriveCriticRow(input: {
  state: ForecastPhaseState
  artifacts: ForecastProgressInput["artifacts"]
  liveEvents: ForecastLiveToolEvent[] | undefined
  activeAgent: ProgressAgent | null
  workflowState: string
  pauseReason: string | null
}): StageRow {
  const { state, artifacts, liveEvents, activeAgent, workflowState, pauseReason } = input
  const postForecast = state.phase === "post-forecast"
  const hasTestEvaluation = artifacts.testEvaluationCount > 0
  const hasVisualizations = artifacts.hasWindowViz && artifacts.hasTrajectoryViz
  const evaluationActive = postForecast && hasLiveStage(liveEvents, "critic-evaluate", "running")
  const adjustmentExpected =
    state.adjustment.enabled && !state.adjustment.no_effective_skill && Boolean(state.adjustment.active_skill_version)
  const testApplicationRecords = state.adjustment.test_application_records
  const testCollectionPrepared = state.adjustment.test_collection_run_ids.length > 0
  const adjustmentComplete =
    testApplicationRecords.length > 0
      ? testApplicationRecords.every(
          (record) =>
            record.decision !== "pending" &&
            (record.decision === "skip" || (record.adjustment_locked && record.adjustment_evaluated)),
        )
      : testCollectionPrepared && state.adjustment.test_eligible_run_ids.length === 0
        ? true
        : state.adjustment.test_adjustment_locked
  const adjustmentActive = postForecast && hasLiveStage(liveEvents, "critic-adjustment", "running")
  const reflectionExpected = state.reflection.enabled && state.reflection.configured
  const reflectionComplete = state.reflection.test_records.length > 0
  const reflectionActive = postForecast && hasLiveStage(liveEvents, "critic-reflection", "running")
  const visualizationActive = postForecast && hasLiveStage(liveEvents, "critic-viz", "running")
  const distillationActive = postForecast && hasLiveStage(liveEvents, "critic-distill", "running")
  const reportActive = postForecast && hasLiveStage(liveEvents, "critic-report", "running")
  const liveStage = postForecast ? activeCriticLiveStage(liveEvents) : undefined

  const steps = [
    step(
      "Evaluating raw test set",
      hasTestEvaluation ? "complete" : postForecast || evaluationActive ? "active" : "pending",
    ),
    ...(adjustmentExpected
      ? [
          step(
            "Applying generative prediction reasoning",
            adjustmentComplete
              ? "complete"
              : adjustmentActive || (postForecast && hasTestEvaluation)
                ? "active"
                : "pending",
          ),
        ]
      : []),
    ...(reflectionExpected
      ? [
          step(
            "Running window reflection",
            reflectionComplete
              ? "complete"
              : reflectionActive || (postForecast && hasTestEvaluation && (!adjustmentExpected || adjustmentComplete))
                ? "active"
                : "pending",
          ),
        ]
      : []),
    step(
      "Generating visualizations",
      hasVisualizations
        ? "complete"
        : visualizationActive ||
            (postForecast &&
              hasTestEvaluation &&
              (!adjustmentExpected || adjustmentComplete) &&
              (!reflectionExpected || reflectionComplete))
          ? "active"
          : "pending",
    ),
    step(
      "Distillation",
      artifacts.hasBestModelExport
        ? "complete"
        : distillationActive || (postForecast && hasVisualizations)
          ? "active"
          : "pending",
    ),
    step(
      "Writing final report",
      artifacts.finalReportState === "written"
        ? "complete"
        : reportActive || (postForecast && artifacts.hasBestModelExport)
          ? "active"
          : "pending",
    ),
  ]

  const active = activeAgent === "critic" || Boolean(liveStage)
  const paused = active && workflowState === "paused" && Boolean(pauseReason)
  const complete = steps.every((item) => item.status === "complete")
  const currentStep =
    liveStage && CRITIC_LIVE_STAGE_LABELS[liveStage]
      ? steps.find((item) => item.label === CRITIC_LIVE_STAGE_LABELS[liveStage])
      : firstActiveStep(steps)
  const detail = appendActivePauseReason({
    detail: complete ? "Final report written" : (currentStep?.label ?? "Waiting for post forecast"),
    active,
    workflowState,
    pauseReason,
  })

  return {
    id: "critic",
    label: "Critic",
    status: paused ? "warning" : complete ? "complete" : active || currentStep ? "active" : "pending",
    active,
    detail,
    steps,
  }
}

export function defaultForecastProgress(): ForecastProgressSnapshot {
  return {
    modeLabel: "Manual",
    activeAgent: null,
    workflowState: "not-started",
    pauseReason: null,
    summary: "Manual: Planner not started",
    rows: [
      {
        id: "planner",
        label: "Planner",
        status: "active",
        active: true,
        detail: "Clarifying",
        steps: [
          step("Clarifying", "active"),
          step("Task frozen", "pending"),
          step("Pre-forecast", "pending"),
          step("Skills confirmed", "pending"),
        ],
      },
      {
        id: "forecaster",
        label: "Forecaster",
        status: "pending",
        active: false,
        detail: "0/20 experiments, 20 remaining",
        steps: [],
      },
      {
        id: "critic",
        label: "Critic",
        status: "pending",
        active: false,
        detail: "Waiting for post forecast",
        steps: [
          step("Evaluating raw test set", "pending"),
          step("Generating visualizations", "pending"),
          step("Distillation", "pending"),
          step("Writing final report", "pending"),
        ],
      },
    ],
  }
}

export function normalizeForecastStateFrontmatter(data: Record<string, unknown>): Record<string, unknown> {
  const normalizeValue = (value: unknown) => (value instanceof Date ? value.toISOString() : value)
  const autoMode =
    data.auto_mode && typeof data.auto_mode === "object"
      ? (() => {
          const value = data.auto_mode as Record<string, unknown>
          return {
            ...value,
            handoffs: Array.isArray(value.handoffs)
              ? value.handoffs.map((handoff) => {
                  if (!handoff || typeof handoff !== "object") return handoff
                  const record = handoff as Record<string, unknown>
                  return { ...record, at: normalizeValue(record.at) }
                })
              : value.handoffs,
          }
        })()
      : data.auto_mode
  const adjustment =
    data.adjustment && typeof data.adjustment === "object"
      ? (() => {
          const value = data.adjustment as Record<string, unknown>
          return {
            ...value,
            user_decision_at: normalizeValue(value.user_decision_at),
            critic_user_confirmed: normalizeValue(value.critic_user_confirmed),
            test_application_decision_at: normalizeValue(value.test_application_decision_at),
            test_application_records: Array.isArray(value.test_application_records)
              ? value.test_application_records.map((entry) => {
                  if (!entry || typeof entry !== "object") return entry
                  const record = entry as Record<string, unknown>
                  return { ...record, decision_at: normalizeValue(record.decision_at) }
                })
              : value.test_application_records,
          }
        })()
      : data.adjustment
  const predictionPrior =
    data.prediction_prior && typeof data.prediction_prior === "object"
      ? (() => {
          const value = data.prediction_prior as Record<string, unknown>
          return {
            ...value,
            user_decision_at: normalizeValue(value.user_decision_at),
          }
        })()
      : data.prediction_prior
  const reflection =
    data.reflection && typeof data.reflection === "object"
      ? (() => {
          const value = data.reflection as Record<string, unknown>
          const normalizeRecords = (records: unknown) =>
            Array.isArray(records)
              ? records.map((entry) => {
                  if (!entry || typeof entry !== "object") return entry
                  const record = entry as Record<string, unknown>
                  return { ...record, completed_at: normalizeValue(record.completed_at) }
                })
              : records
          return {
            ...value,
            validation_records: normalizeRecords(value.validation_records),
            test_records: normalizeRecords(value.test_records),
          }
        })()
      : data.reflection

  return {
    ...data,
    skills_confirmed_at: normalizeValue(data.skills_confirmed_at),
    skill_memory_published_at: normalizeValue(data.skill_memory_published_at) ?? null,
    skill_memory_manifest_path:
      typeof data.skill_memory_manifest_path === "string" ? data.skill_memory_manifest_path : null,
    skill_memory_published_count:
      typeof data.skill_memory_published_count === "number" ? data.skill_memory_published_count : null,
    skill_memory_publish_error:
      typeof data.skill_memory_publish_error === "string" ? data.skill_memory_publish_error : null,
    pre_forecast_completed_at: normalizeValue(data.pre_forecast_completed_at),
    task_frozen_at: normalizeValue(data.task_frozen_at),
    auto_mode: autoMode,
    adjustment,
    reflection,
    prediction_prior: predictionPrior,
    transitions: Array.isArray(data.transitions)
      ? data.transitions.map((transition) => {
          if (!transition || typeof transition !== "object") return transition
          const record = transition as Record<string, unknown>
          return { ...record, at: normalizeValue(record.at) }
        })
      : data.transitions,
  }
}

export function parseForecastHistory(raw: string): Array<ExperimentRecord | ReflectionRecord | ExpertFeedbackRecord> {
  const records: Array<ExperimentRecord | ReflectionRecord | ExpertFeedbackRecord> = []

  for (const line of raw.split("\n").filter(Boolean)) {
    try {
      const parsed = JSON.parse(line)

      const feedback = ExpertFeedbackRecordSchema.safeParse(parsed)
      if (feedback.success) {
        records.push(feedback.data)
        continue
      }

      const reflection = ReflectionRecordSchema.safeParse(parsed)
      if (reflection.success) {
        records.push(reflection.data)
        continue
      }

      const experiment = ExperimentRecordSchema.safeParse(parsed)
      if (experiment.success) {
        records.push(experiment.data)
      }
    } catch {
      // Partial history should not hide otherwise durable progress.
    }
  }

  return records
}

export function formatBestMetric(best: ExperimentRecord | null): string | null {
  if (!best?.metrics) return null

  for (const key of ["mse", "mae", "wape", "mase"] as const) {
    const value = best.metrics[key]
    if (typeof value === "number" && Number.isFinite(value)) {
      return `${best.model} ${key}=${value.toPrecision(4)}`
    }
  }

  return null
}

export function deriveForecastProgress(input: ForecastProgressInput): ForecastProgressSnapshot {
  if (!input.state) return defaultForecastProgress()

  const state = input.state
  const artifacts = input.artifacts ?? DEFAULT_ARTIFACTS
  const modeLabel: ForecastProgressSnapshot["modeLabel"] = state.auto_mode.enabled ? "Auto Mode" : "Auto Mode Off"
  const activeAgent = state.auto_mode.enabled ? state.auto_mode.active_agent : phaseActiveAgent(state)
  const workflowState = state.auto_mode.enabled ? state.auto_mode.state : state.phase
  const pauseReason = state.auto_mode.enabled ? state.auto_mode.pause_reason : null
  const summary = activeAgent
    ? `${modeLabel}: ${titleCaseAgent(activeAgent)} ${workflowStateLabel(workflowState)}`
    : "Manual: Planner not started"

  return {
    modeLabel,
    activeAgent,
    workflowState,
    pauseReason,
    summary,
    rows: [
      derivePlannerRow({ state, artifacts, activeAgent, workflowState, pauseReason }),
      deriveForecasterRow({
        state,
        history: input.history,
        best: input.best,
        liveEvents: input.liveEvents,
        activeAgent,
        workflowState,
        pauseReason,
      }),
      deriveCriticRow({ state, artifacts, liveEvents: input.liveEvents, activeAgent, workflowState, pauseReason }),
    ],
  }
}

async function readOptionalText(file: string): Promise<string | null> {
  return readFile(file, "utf-8").catch(() => null)
}

async function hasReadableFile(file: string): Promise<boolean> {
  return readOptionalText(file).then((content) => content !== null)
}

async function readReportState(
  file: string,
  placeholder: string,
): Promise<ForecastProgressInput["artifacts"]["finalReportState"]> {
  const raw = await readOptionalText(file)
  if (raw === null || raw.trim() === "") return "missing"
  if (raw.includes(placeholder)) return "placeholder"
  return "written"
}

async function readBest(forecastRoot: string): Promise<ExperimentRecord | null> {
  const raw = await readOptionalText(path.join(forecastRoot, "best.json"))
  if (!raw) return null

  try {
    const parsed = ExperimentRecordSchema.safeParse(JSON.parse(raw))
    return parsed.success ? parsed.data : null
  } catch {
    return null
  }
}

async function readState(forecastRoot: string): Promise<ForecastPhaseState | null> {
  const raw = await readOptionalText(path.join(forecastRoot, "STATE.md"))
  if (!raw) return null

  const parsed = matter(raw)
  const data = normalizeForecastStateFrontmatter(parsed.data as Record<string, unknown>)
  const state = ForecastPhaseStateSchema.safeParse(data)
  return state.success ? state.data : null
}

async function countTestEvaluations(forecastRoot: string): Promise<number> {
  const runsRoot = path.join(forecastRoot, "runs")
  const entries = await readdir(runsRoot, { withFileTypes: true }).catch(() => [])
  let count = 0

  for (const entry of entries) {
    if (!entry.isDirectory()) continue

    const runDir = path.join(runsRoot, entry.name)
    const [specRaw, evalRaw] = await Promise.all([
      readOptionalText(path.join(runDir, "spec.json")),
      readOptionalText(path.join(runDir, "eval.json")),
    ])
    if (!specRaw || evalRaw === null) continue

    try {
      const spec = JSON.parse(specRaw)
      const parsedEval = EvalResultSchema.safeParse(JSON.parse(evalRaw))
      if (
        parsedEval.success &&
        parsedEval.data.run_id === entry.name &&
        parsedEval.data.eval_split === "test" &&
        spec &&
        typeof spec === "object" &&
        (spec as Record<string, unknown>).phase === "post-forecast" &&
        (spec as Record<string, unknown>).eval_split === "test"
      ) {
        await readBoundModelArtifactIdentity(
          runDir,
          String((spec as Record<string, unknown>).model ?? ""),
          parsedEval.data,
        )
        count += 1
      }
    } catch {
      // Ignore corrupt run metadata while preserving progress from other runs.
    }
  }

  return count
}

async function hasValidBestModelExport(forecastRoot: string): Promise<boolean> {
  const raw = await readOptionalText(path.join(forecastRoot, "best-model", "manifest.json"))
  if (!raw) return false

  try {
    return BestModelManifestSchema.safeParse(JSON.parse(raw)).success
  } catch {
    return false
  }
}

export async function readForecastProgress(
  directory: string | undefined,
  liveEvents: ForecastLiveToolEvent[] = [],
): Promise<ForecastProgressSnapshot> {
  if (!directory) return defaultForecastProgress()

  const forecastRoot = path.join(directory, ".forecast")
  const state = await readState(forecastRoot)
  if (!state) return defaultForecastProgress()

  const [
    hasTask,
    historyRaw,
    best,
    hasPreForecastReport,
    testEvaluationCount,
    hasWindowViz,
    hasTrajectoryViz,
    hasBestModelExport,
    finalReportState,
    historicalExplanationsState,
  ] = await Promise.all([
    hasReadableFile(path.join(forecastRoot, "task.json")),
    readOptionalText(path.join(forecastRoot, "history.jsonl")),
    readBest(forecastRoot),
    hasReadableFile(path.join(forecastRoot, "reports", "pre-forecast.md")),
    countTestEvaluations(forecastRoot),
    hasReadableFile(path.join(forecastRoot, "viz", "window-level.png")),
    hasReadableFile(path.join(forecastRoot, "viz", "trajectory.png")),
    hasValidBestModelExport(forecastRoot),
    readReportState(path.join(forecastRoot, "reports", "final-report.md"), FINAL_REPORT_PLACEHOLDER),
    readReportState(
      path.join(forecastRoot, "reports", "historical-explanations.md"),
      HISTORICAL_EXPLANATIONS_PLACEHOLDER,
    ),
  ])

  const thresholds = state.budget_thresholds ? BudgetThresholdsSchema.parse(state.budget_thresholds) : undefined

  return deriveForecastProgress({
    state: thresholds ? { ...state, budget_thresholds: thresholds } : state,
    history: parseForecastHistory(historyRaw ?? ""),
    best,
    artifacts: {
      hasTask,
      hasPreForecastReport,
      testEvaluationCount,
      hasWindowViz,
      hasTrajectoryViz,
      hasBestModelExport,
      finalReportState,
      historicalExplanationsState,
    },
    liveEvents,
  })
}
