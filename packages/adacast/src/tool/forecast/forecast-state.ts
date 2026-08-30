import { appendFile, mkdir, readFile, symlink, writeFile } from "fs/promises"
import matter from "gray-matter"
import path from "path"
import z from "zod"
import { MessageV2 } from "@/session/message-v2"
import { SessionCompaction } from "@/session/compaction"
import { SessionPrompt } from "@/session/prompt"
import { Instance } from "@/project/instance"
import {
  buildForecastAutoModeContinuationPromptFromArtifacts,
  type ForecastHandoffTarget,
} from "@/plugin/forecast-compaction"
import { Filesystem } from "@/util/filesystem"
import { Tool } from "../tool"
import DESCRIPTION from "./forecast-state.txt"
import { PhaseGateError, TaskNotFoundError } from "./errors"
import { deduplicateExperimentsByRunId, experimentAttemptIndex } from "./forecast-history"
import { normalizeForecastStateFrontmatter } from "./forecast-progress"
import {
  validateWindowReflectionPolicyArtifact,
  validateWindowReflectionRecordArtifact,
} from "./forecast-window-reflection-artifact"
import { publishForecastSkillMemory } from "./forecast-skill-memory"
import { inferForecastModelFamily } from "./model-family"
import { resolveForecastPythonProject } from "./python-project"
import {
  ForecastAutoModeAgent,
  ForecastAutoModeHandoff,
  ForecastAutoModeWorkflowState,
  BudgetState,
  BudgetThresholds,
  ExperimentRecord,
  ExpertFeedbackRecord,
  ForecastContext,
  ForecastPhase,
  ForecastPhaseState,
  AdjustmentEvalResult,
  AdjustmentSkillManifest,
  ReflectionRecord,
  StopCheckResult,
} from "./types"
import { hasResolvedUserPriorMirror } from "./forecast-adjustment-acceptance"
import { inspectFrozenTaskIntegrity } from "./forecast-task-integrity"

const parameters = z.object({
  subcommand: z
    .enum([
      "read",
      "append",
      "best",
      "resume",
      "budget",
      "init",
      "transition",
      "phase_gate",
      "confirm_skills",
      "configure_auto_mode",
      "check_reflection",
      "update_stop_counters",
      "record_expert_feedback",
    ])
    .describe(
      "State operation: read (enriched experiment context with D-12 schema), append (add experiment record), best (read/update best result), resume (rebuild from history), budget (read/update budget), init (prepare .forecast), transition (advance forecast phase), phase_gate (assert phase), confirm_skills (freeze reviewed skills), configure_auto_mode (persist durable Auto Mode settings), check_reflection (verify the latest experiment has a paired reflection), update_stop_counters (increment stop counters and check thresholds), record_expert_feedback (record expert guidance and reset no-improvement counter)",
    ),
  record: ExperimentRecord.optional().describe("Experiment record to append or persist as best.json"),
  update_best: z.boolean().optional().describe("If true with 'best', write best.json from the provided record"),
  budget_increment: z
    .object({
      elapsed_seconds: z.number().describe("Seconds to add to budget"),
    })
    .optional()
    .describe("Budget increment payload for the 'budget' subcommand"),
  target_phase: ForecastPhase.optional().describe("Target phase for transition or phase_gate subcommands"),
  stop_counter_input: z
    .object({
      run_id: z
        .string()
        .optional()
        .describe("Run id for the experiment being accounted; defaults to the latest unique experiment"),
      improvement: z.boolean().describe("Whether the last experiment improved the best metric"),
      crashed: z.boolean().describe("Whether the last experiment status was crash (not ipc_error or timeout)"),
    })
    .optional()
    .describe("Input for update_stop_counters subcommand"),
  expert_feedback_text: z
    .string()
    .optional()
    .describe("Free-form expert text to record in history via record_expert_feedback"),
  auto_mode_update: z
    .object({
      enabled: z.boolean().optional().describe("Enable or disable Auto Mode for the current forecast session"),
      active_agent: ForecastAutoModeAgent.nullable()
        .optional()
        .describe("Active pipeline agent while Auto Mode is orchestrating the workflow"),
      state: ForecastAutoModeWorkflowState.optional().describe("Durable Auto Mode workflow state label"),
      pause_reason: z.string().nullable().optional().describe("Reason Auto Mode is paused or waiting for manual input"),
      auto_approve_permissions: z
        .boolean()
        .optional()
        .describe("Whether routine tool permission prompts are auto-approved"),
      auto_approve_skills: z.boolean().optional().describe("Whether routine skills review prompts are auto-approved"),
      handoff: ForecastAutoModeHandoff.optional().describe("Append a durable handoff audit record"),
      reset_handoffs: z
        .boolean()
        .optional()
        .describe("Reset the durable Auto Mode handoff log before applying the update"),
    })
    .optional()
    .describe("Payload for configure_auto_mode"),
})

type ExperimentRecordValue = z.infer<typeof ExperimentRecord>
type BudgetStateValue = z.infer<typeof BudgetState>
type ForecastPhaseStateValue = z.infer<typeof ForecastPhaseState>
type ForecastAutoModeAgentValue = z.infer<typeof ForecastAutoModeAgent>
type ReflectionRecordValue = z.infer<typeof ReflectionRecord>
type ExpertFeedbackRecordValue = z.infer<typeof ExpertFeedbackRecord>
type HistoryEntryValue = ExperimentRecordValue | ReflectionRecordValue | ExpertFeedbackRecordValue
type AutoModeUpdateValue = NonNullable<z.infer<typeof parameters>["auto_mode_update"]>

function defaultAutoModeState(): ForecastPhaseStateValue["auto_mode"] {
  return {
    enabled: false,
    active_agent: null,
    state: "waiting",
    pause_reason: null,
    auto_approve_permissions: false,
    auto_approve_skills: false,
    handoffs: [],
  }
}

function mergeAutoModeState(
  current: ForecastPhaseStateValue["auto_mode"] | undefined,
  update: AutoModeUpdateValue,
): ForecastPhaseStateValue["auto_mode"] {
  const previous = current ?? defaultAutoModeState()
  const handoffs = update.reset_handoffs ? [] : [...previous.handoffs]

  if (update.handoff) {
    handoffs.push(update.handoff)
  }

  const next: ForecastPhaseStateValue["auto_mode"] = {
    enabled: update.enabled ?? previous.enabled,
    active_agent: update.active_agent !== undefined ? update.active_agent : previous.active_agent,
    state: update.state ?? previous.state,
    pause_reason: update.pause_reason !== undefined ? update.pause_reason : previous.pause_reason,
    auto_approve_permissions: update.auto_approve_permissions ?? previous.auto_approve_permissions,
    auto_approve_skills: update.auto_approve_skills ?? previous.auto_approve_skills,
    handoffs,
  }

  if (update.enabled === false && update.active_agent === undefined) {
    next.active_agent = null
  }

  if (update.enabled === false && update.state === undefined) {
    next.state = "waiting"
  }

  if (update.state && update.state !== "paused" && update.pause_reason === undefined) {
    next.pause_reason = null
  }

  return next
}

function forecastDir(dir: string): string {
  return path.join(dir, ".forecast")
}

async function ensureForecastDir(dir: string): Promise<string> {
  const result = forecastDir(dir)
  await mkdir(result, { recursive: true })
  return result
}

async function ensureTaskDefined(dir: string): Promise<string> {
  const taskPath = path.join(forecastDir(dir), "task.json")
  if (!(await Filesystem.exists(taskPath))) {
    throw new TaskNotFoundError({ expected_path: taskPath })
  }
  return taskPath
}

async function readHistory(dir: string): Promise<ExperimentRecordValue[]> {
  const entries = await readHistoryEntries(dir)
  return deduplicateExperimentsByRunId(entries.filter(isExperimentRecord))
}

function parseHistoryEntry(line: string): HistoryEntryValue | null {
  try {
    const parsed = JSON.parse(line)

    const feedback = ExpertFeedbackRecord.safeParse(parsed)
    if (feedback.success) {
      return feedback.data
    }

    const reflection = ReflectionRecord.safeParse(parsed)
    if (reflection.success) {
      return reflection.data
    }

    const experiment = ExperimentRecord.safeParse(parsed)
    if (experiment.success) {
      return experiment.data
    }
  } catch {
    // Ignore invalid JSON lines so state recovery can keep going.
  }

  return null
}

function isExperimentRecord(entry: HistoryEntryValue): entry is ExperimentRecordValue {
  return !("type" in entry)
}

function isExpertFeedbackRecord(entry: HistoryEntryValue): entry is ExpertFeedbackRecordValue {
  return "type" in entry && entry.type === "expert_feedback"
}

async function readHistoryEntries(dir: string): Promise<HistoryEntryValue[]> {
  const historyPath = path.join(forecastDir(dir), "history.jsonl")
  const raw = await readFile(historyPath, "utf-8").catch(() => "")
  const records: HistoryEntryValue[] = []

  for (const line of raw.split("\n").filter(Boolean)) {
    const entry = parseHistoryEntry(line)
    if (entry) {
      records.push(entry)
    }
  }

  return records
}

function getLastReflectionStatus(entries: HistoryEntryValue[]): {
  lastExperimentRunId: string | null
  hasReflection: boolean
} {
  let lastExperimentRunId: string | null = null
  let hasReflection = false

  for (const entry of entries) {
    if ("type" in entry) {
      if (entry.type === "reflection") {
        if (entry.run_id === lastExperimentRunId) {
          hasReflection = true
        }
      }
      continue
    }

    lastExperimentRunId = entry.run_id
    hasReflection = false
  }

  return { lastExperimentRunId, hasReflection }
}

function rebuildState(records: ExperimentRecordValue[]) {
  records = deduplicateExperimentsByRunId(records)
  const completedRunIds = [...new Set(records.map((record) => record.run_id))]
  const bestResult =
    records.reduce<ExperimentRecordValue | null>((best, record) => {
      if (record.status !== "success" || record.metrics?.mse == null) return best
      if (!best || best.metrics?.mse == null || record.metrics.mse < best.metrics.mse) {
        return record
      }
      return best
    }, null) ?? null

  // Compute per-family best results to support condition-based model selection.
  // Each model is assigned to a family via its window_features.model_family field
  // (set by the Forecaster agent when proposing experiments).
  const perFamilyBest: Record<string, { run_id: string; model: string; metrics: ExperimentRecordValue["metrics"] }> = {}
  for (const record of records) {
    if (record.status !== "success" || record.metrics?.mse == null) continue
    const family = record.window_features?.model_family ?? inferForecastModelFamily(record.model) ?? "unclassified"
    const current = perFamilyBest[family]
    if (!current || (current.metrics?.mse != null && record.metrics.mse < current.metrics.mse)) {
      perFamilyBest[family] = {
        run_id: record.run_id,
        model: record.model,
        metrics: record.metrics,
      }
    }
  }

  const budget = {
    experiment_count: records.length,
    total_elapsed_seconds: records.reduce((sum, record) => sum + record.elapsed_seconds, 0),
    last_updated: records.at(-1)?.timestamp ?? bestResult?.timestamp ?? new Date(0).toISOString(),
  } satisfies BudgetStateValue

  return {
    completedRunIds,
    bestResult,
    perFamilyBest,
    budget,
    experimentCount: records.length,
  }
}

async function readBest(forecastRoot: string): Promise<ExperimentRecordValue | null> {
  const bestPath = path.join(forecastRoot, "best.json")
  try {
    return ExperimentRecord.parse(await Filesystem.readJson(bestPath))
  } catch {
    return null
  }
}

async function readBudget(forecastRoot: string): Promise<BudgetStateValue | null> {
  const budgetPath = path.join(forecastRoot, "budget.json")
  try {
    return BudgetState.parse(await Filesystem.readJson(budgetPath))
  } catch {
    return null
  }
}

function defaultPhaseState(): ForecastPhaseStateValue {
  return {
    phase: "init",
    skills_confirmed: false,
    skills_confirmed_at: null,
    skill_memory_published_at: null,
    skill_memory_manifest_path: null,
    skill_memory_published_count: null,
    skill_memory_publish_error: null,
    pre_forecast_completed: false,
    pre_forecast_completed_at: null,
    task_frozen_at: null,
    transitions: [],
    auto_mode: defaultAutoModeState(),
    consecutive_no_improvement: 0,
    consecutive_crashes: 0,
    stop_counter_last_run_id: null,
    stop_counter_last_attempt_index: null,
    stop_reason: null,
    budget_initialized: false,
    adjustment: {
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
    },
    reflection: {
      enabled: true,
      configured: false,
      policy_path: null,
      policy_sha256: null,
      controller_version: "1",
      max_reflection_rounds: 3,
      max_format_retries: 2,
      validation_records: [],
      test_records: [],
    },
    prediction_prior: {
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
    },
  }
}

async function readPhaseState(forecastRoot: string): Promise<ForecastPhaseStateValue> {
  const statePath = path.join(forecastRoot, "STATE.md")
  try {
    const raw = await readFile(statePath, "utf-8")
    const parsed = matter(raw)
    return ForecastPhaseState.parse(normalizeForecastStateFrontmatter(parsed.data as Record<string, unknown>))
  } catch {
    return defaultPhaseState()
  }
}

async function writePhaseState(forecastRoot: string, state: ForecastPhaseStateValue, logEntry?: string): Promise<void> {
  const statePath = path.join(forecastRoot, "STATE.md")
  let body = ""
  try {
    const existing = matter(await readFile(statePath, "utf-8"))
    body = existing.content
  } catch {
    body = "\n# Forecast State Log\n"
  }
  if (logEntry) {
    body += `\n${logEntry}\n`
  }
  const output = matter.stringify(body, state)
  await writeFile(statePath, output, "utf-8")
}

const VALID_TRANSITIONS: Record<string, string[]> = {
  init: ["pre-forecast"],
  "pre-forecast": ["skills-review"],
  "skills-review": ["forecast"],
  forecast: ["post-forecast"],
  "post-forecast": [],
}

const REQUIRED_PRE_FORECAST_SECTIONS = [
  "Task Summary",
  "Data Characteristics",
  "Domain Context",
  "Risk Flags",
  "Modeling Direction Recommendations",
] as const

async function validateTransition(
  forecastRoot: string,
  current: string,
  target: string,
): Promise<{ ok: true } | { ok: false; missing: string[] }> {
  const allowed = VALID_TRANSITIONS[current]
  if (!allowed || !allowed.includes(target)) {
    return { ok: false, missing: [`transition ${current} -> ${target} is not allowed`] }
  }

  const missing: string[] = []
  const rawTask = await Filesystem.readJson<Record<string, unknown>>(path.join(forecastRoot, "task.json")).catch(
    () => null,
  )
  const taskIntegrity = await inspectFrozenTaskIntegrity(forecastRoot)
  if (taskIntegrity.status === "modified") {
    return {
      ok: false,
      missing: [
        "frozen task.json changed after forecast_task; restore the exact frozen task or start a new task instead of editing managed artifacts",
      ],
    }
  }
  if (taskIntegrity.status === "invalid_lock") {
    return { ok: false, missing: [`frozen task integrity metadata is invalid: ${taskIntegrity.reason}`] }
  }
  const reflectionRequired = rawTask?.reflection_enabled === true

  switch (target) {
    case "pre-forecast": {
      const taskPath = path.join(forecastRoot, "task.json")
      if (!(await Filesystem.exists(taskPath))) {
        missing.push("task.json must exist before entering pre-forecast")
      }
      const state = await readPhaseState(forecastRoot)
      if (state.prediction_prior.user_decision === "pending") {
        missing.push(
          "ask the user whether to submit a prediction-prior dataset with forecast_prediction_prior ask_start before entering pre-forecast",
        )
      }
      break
    }

    case "skills-review": {
      const reportPath = path.join(forecastRoot, "reports", "pre-forecast.md")
      if (!(await Filesystem.exists(reportPath))) {
        missing.push("reports/pre-forecast.md must exist before entering skills-review")
        break
      }
      const report = await readFile(reportPath, "utf-8")
      const absentSections = REQUIRED_PRE_FORECAST_SECTIONS.filter(
        (heading) => !extractMarkdownSection(report, heading),
      )
      if (absentSections.length > 0) {
        missing.push(
          `reports/pre-forecast.md must contain sections: ${REQUIRED_PRE_FORECAST_SECTIONS.join(", ")}. Missing: ${absentSections.join(", ")}`,
        )
      }
      break
    }

    case "forecast": {
      const state = await readPhaseState(forecastRoot)
      if (!state.skills_confirmed) {
        missing.push("skills must be confirmed (skills_confirmed: true) before entering forecast")
      }
      break
    }

    case "post-forecast": {
      const state = await readPhaseState(forecastRoot)
      if (state.adjustment.user_decision === "pending") {
        missing.push(
          state.auto_mode.enabled
            ? "record the Auto Mode LLM generative prediction reasoning decision (disabled, lightweight prediction reasoning, or deep prediction reasoning) with forecast_adjustment_decision auto_start before entering post-forecast"
            : "ask the user to choose whether to disable generative prediction reasoning, use lightweight prediction reasoning, or use deep prediction reasoning with forecast_adjustment_decision ask_start before entering post-forecast",
        )
      }
      if (state.adjustment.enabled) {
        const hasLockedSkill = Boolean(state.adjustment.forecaster_locked && state.adjustment.active_skill_version)
        const hasNoEffectiveSkillRecord = Boolean(
          state.adjustment.forecaster_locked && state.adjustment.no_effective_skill,
        )
        if (!hasLockedSkill && !hasNoEffectiveSkillRecord) {
          missing.push(
            "adjustment is enabled; lock an active adjustment skill or record no-effective-skill before entering post-forecast",
          )
        }
        if (
          typeof rawTask?.adjustment_skill_priors === "string" &&
          rawTask.adjustment_skill_priors.trim() &&
          state.adjustment.source_model_set_locked &&
          state.adjustment.fit_runs_ready &&
          !(await hasResolvedUserPriorMirror(forecastRoot))
        ) {
          missing.push(
            "task contains adjustment_skill_priors; independently inspect, write, validate, and accept or reject at least one user_prior_mirror candidate before entering post-forecast",
          )
        }
      }
      if (reflectionRequired) {
        const policy = await validateWindowReflectionPolicyArtifact(forecastRoot, state.reflection.policy_path, {
          enabled: state.reflection.enabled,
          controllerVersion: state.reflection.controller_version,
          maxReflectionRounds: state.reflection.max_reflection_rounds,
          maxFormatRetries: state.reflection.max_format_retries,
          policySha256: state.reflection.policy_sha256,
        })
        if (!state.reflection.enabled || !state.reflection.configured || !policy) {
          missing.push(
            "reflection_enabled is true; freeze the independent window-reflection policy before entering post-forecast",
          )
        } else {
          const expectedValidationRunIds = state.prediction_prior.enabled
            ? [state.prediction_prior.validation_run_id].filter((value): value is string => Boolean(value))
            : Object.values(rebuildState(await readHistory(Instance.directory)).perFamilyBest).map(
                (record) => record.run_id,
              )
          const expectedCandidates = new Map(expectedValidationRunIds.map((runId) => [runId, null as string | null]))
          if (state.adjustment.enabled && state.adjustment.active_skill_version) {
            const activeManifestPath = path.join(forecastRoot, "adjustments", "skills", "active", "manifest.json")
            const activeManifest = AdjustmentSkillManifest.safeParse(
              await Filesystem.readJson<unknown>(activeManifestPath).catch(() => null),
            )
            if (
              !activeManifest.success ||
              activeManifest.data.version !== state.adjustment.active_skill_version ||
              activeManifest.data.status !== "locked"
            ) {
              missing.push(
                "reflection validation requires the locked active adjustment manifest to match adjustment state",
              )
            } else {
              for (const resultPath of activeManifest.data.validation_results) {
                const resolvedResultPath = path.resolve(Instance.directory, resultPath)
                const resultRelative = path.relative(
                  path.resolve(forecastRoot, "adjustments", "validation"),
                  resolvedResultPath,
                )
                if (resultRelative.startsWith("..") || path.isAbsolute(resultRelative)) continue
                const result = AdjustmentEvalResult.safeParse(
                  await Filesystem.readJson<unknown>(resolvedResultPath).catch(() => null),
                )
                if (
                  result.success &&
                  result.data.split === "val" &&
                  result.data.candidate_id === activeManifest.data.candidate_id &&
                  result.data.acceptance_decision?.accepted === true &&
                  expectedCandidates.has(result.data.run_id)
                ) {
                  expectedCandidates.set(result.data.run_id, result.data.candidate_id)
                }
              }
            }
          }
          const baselineKey = (runId: string, candidateId: string | null) => `${runId}\u0000${candidateId ?? "raw"}`
          const completedValidationBaselines = new Set<string>()
          for (const record of state.reflection.validation_records) {
            if (["pending", "failed"].includes(record.status)) continue
            const artifact = await validateWindowReflectionRecordArtifact(forecastRoot, record, "val")
            if (!artifact?.auditPassed) continue
            completedValidationBaselines.add(baselineKey(record.validation_run_id, record.adjustment_candidate_id))
          }
          const absent = [...expectedCandidates].filter(
            ([runId, candidateId]) => !completedValidationBaselines.has(baselineKey(runId, candidateId)),
          )
          if (absent.length > 0) {
            missing.push(
              `reflection_enabled is true; complete independent validation-back reflection checks for final validation baseline(s): ${absent
                .map(([runId, candidateId]) => `${runId}/${candidateId ?? "raw"}`)
                .join(", ")}`,
            )
          }
        }
      }
      break
    }
  }

  return missing.length > 0 ? { ok: false, missing } : { ok: true }
}

function extractMarkdownSection(markdown: string, heading: string): string {
  const headingLine = `## ${heading}`
  const startIndex = markdown.indexOf(headingLine)
  if (startIndex === -1) {
    return ""
  }

  const sectionStart = markdown.indexOf("\n", startIndex)
  if (sectionStart === -1) {
    return ""
  }

  const remainder = markdown.slice(sectionStart + 1)
  const nextHeadingOffset = remainder.search(/^## /m)
  const section = nextHeadingOffset === -1 ? remainder : remainder.slice(0, nextHeadingOffset)
  return section.trim()
}

function parseCastThresholds(
  castText: string,
): Partial<{ max_experiments: number; no_improvement_rounds: number; crash_threshold: number }> {
  const result: Record<string, number> = {}
  const pattern = /^(max_experiments|no_improvement_rounds|crash_threshold):\s*(\d+)/gim
  for (const match of castText.matchAll(pattern)) {
    result[match[1].toLowerCase()] = parseInt(match[2], 10)
  }
  return result
}

async function lazyInitBudget(
  forecastRoot: string,
  directory: string,
  currentState: ForecastPhaseStateValue,
): Promise<ForecastPhaseStateValue> {
  if (currentState.budget_initialized) return currentState

  let overrides: Partial<z.infer<typeof BudgetThresholds>> = {}
  try {
    const [castPath] = await Filesystem.findUp("CAST.md", directory, directory)
    if (castPath) {
      const castText = await readFile(castPath, "utf-8")
      overrides = parseCastThresholds(castText)
    }
  } catch {
    // CAST.md is optional — use defaults.
  }

  const thresholds = BudgetThresholds.parse({
    max_experiments: overrides.max_experiments ?? 20,
    no_improvement_rounds: overrides.no_improvement_rounds ?? 3,
    crash_threshold: overrides.crash_threshold ?? 3,
  })

  const updated: ForecastPhaseStateValue = {
    ...currentState,
    budget_initialized: true,
    budget_thresholds: thresholds,
  }
  await writePhaseState(
    forecastRoot,
    updated,
    `## Budget Initialized\n\nThresholds frozen: max_experiments=${thresholds.max_experiments}, no_improvement_rounds=${thresholds.no_improvement_rounds}, crash_threshold=${thresholds.crash_threshold}.`,
  )
  return updated
}

async function generateForecastContext(forecastRoot: string, directory: string): Promise<void> {
  const contextPath = path.join(forecastRoot, "CONTEXT.md")
  const [castPath] = await Filesystem.findUp("CAST.md", directory, directory)

  let castConstraints = ""
  if (castPath) {
    castConstraints = (await readFile(castPath, "utf-8")).trim()
  }

  let preForecastSummary = ""
  let modelingDirections = ""
  const preForecastPath = path.join(forecastRoot, "reports", "pre-forecast.md")
  if (await Filesystem.exists(preForecastPath)) {
    const report = await readFile(preForecastPath, "utf-8")
    const summarySections = [
      extractMarkdownSection(report, "Data Characteristics"),
      (() => {
        const riskFlags = extractMarkdownSection(report, "Risk Flags")
        return riskFlags ? `### Risk Flags\n${riskFlags}` : ""
      })(),
    ].filter(Boolean)

    preForecastSummary = summarySections.join("\n\n")
    modelingDirections = extractMarkdownSection(report, "Modeling Direction Recommendations")
  }

  const now = new Date().toISOString()
  const context = ForecastContext.parse({
    generated_at: now,
    constraints: castConstraints,
    pre_forecast_summary: preForecastSummary,
    modeling_directions: modelingDirections,
  })

  const lines = ["# Forecast Context", "", `_Generated: ${context.generated_at}_`, "", "## Constraints", ""]
  if (context.constraints) {
    lines.push(context.constraints, "")
  } else {
    lines.push("No CAST.md constraints defined. All model families and approaches are permitted.", "")
  }

  if (context.pre_forecast_summary) {
    lines.push("## Pre-Forecast Summary", "", context.pre_forecast_summary, "")
  }

  if (context.modeling_directions) {
    lines.push("## Modeling Directions", "", context.modeling_directions, "")
  }

  await writeFile(contextPath, `${lines.join("\n")}\n`, "utf-8")
}

function findLastUserMessage(messages: Tool.Context["messages"]): MessageV2.User | undefined {
  for (let index = messages.length - 1; index >= 0; index--) {
    const info = messages[index]?.info
    if (info?.role === "user") {
      return info
    }
  }
}

async function resolveCompactionSource(ctx: Tool.Context): Promise<MessageV2.User | undefined> {
  const userFromContext = findLastUserMessage(ctx.messages)
  if (userFromContext) return userFromContext

  try {
    for await (const message of MessageV2.stream(ctx.sessionID)) {
      if (message.info.role === "user") {
        return message.info
      }
    }
  } catch {
    // Tool tests can run without a persisted session backing the synthetic context.
  }
}

async function queuePostForecastCompaction(ctx: Tool.Context): Promise<boolean> {
  const source = await resolveCompactionSource(ctx)
  if (!source) return false

  await SessionCompaction.create({
    sessionID: ctx.sessionID,
    agent: source.agent || ctx.agent,
    model: source.model,
    auto: true,
  })
  return true
}

async function queueAutoModeContinuation(input: {
  ctx: Tool.Context
  targetAgent: ForecastHandoffTarget
  prompt: string
}): Promise<boolean> {
  const source = await resolveCompactionSource(input.ctx)
  if (!source) {
    throw new Error("No source user message found for Auto Mode continuation")
  }

  await SessionPrompt.prompt({
    sessionID: input.ctx.sessionID,
    agent: input.targetAgent,
    model: source.model,
    variant: source.variant,
    noReply: true,
    parts: [
      {
        type: "text",
        synthetic: true,
        text: input.prompt,
      },
    ],
  })

  return true
}

function autoModeBoundaryForTransition(targetPhase: ForecastPhaseStateValue["phase"]): {
  from: ForecastAutoModeAgentValue
  to: ForecastHandoffTarget
  reason: string
} | null {
  if (targetPhase === "forecast") {
    return {
      from: "planner",
      to: "forecaster",
      reason: "Planner completed task freeze and skills review; Forecaster can start the experiment loop.",
    }
  }

  if (targetPhase === "post-forecast") {
    return {
      from: "forecaster",
      to: "critic",
      reason: "Forecaster completed the experiment loop; Critic can produce the final report.",
    }
  }

  return null
}

function applyAutoModeBoundary(
  state: ForecastPhaseStateValue,
  boundary: NonNullable<ReturnType<typeof autoModeBoundaryForTransition>>,
  at: string,
): ForecastPhaseStateValue {
  return {
    ...state,
    auto_mode: mergeAutoModeState(state.auto_mode, {
      active_agent: boundary.to,
      state: "running",
      pause_reason: null,
      handoff: {
        from: boundary.from,
        to: boundary.to,
        at,
        reason: boundary.reason,
      },
    }),
  }
}

async function pauseAutoMode(root: string, state: ForecastPhaseStateValue, reason: string): Promise<void> {
  const updatedState: ForecastPhaseStateValue = {
    ...state,
    auto_mode: mergeAutoModeState(state.auto_mode, {
      state: "paused",
      pause_reason: reason,
    }),
  }

  await writePhaseState(root, updatedState, `## Auto Mode Paused\n\n${reason}`)
}

function pauseReasonForStop(input: {
  stopReason: "budget_exhausted" | "crash_threshold" | "no_improvement" | null
  hitlTriggered: boolean
  noImprovementCount: number
  crashCount: number
  experimentCount: number
  thresholds: z.infer<typeof BudgetThresholds>
}): string | null {
  if (input.stopReason === "budget_exhausted") {
    return `budget_exhausted after ${input.experimentCount}/${input.thresholds.max_experiments} experiments; manual review is required before Auto Mode can continue.`
  }

  if (input.stopReason === "crash_threshold") {
    return `crash_threshold after ${input.crashCount} consecutive crashes; manual recovery is required before Auto Mode can continue.`
  }

  return null
}

export const ForecastStateTool = Tool.define("forecast_state", {
  description: DESCRIPTION,
  parameters,
  async execute(params, ctx) {
    const directory = Instance.directory
    const root = await ensureForecastDir(directory)

    let result: unknown

    switch (params.subcommand) {
      case "read": {
        await ensureTaskDefined(directory)
        const entries = await readHistoryEntries(directory)
        const records = deduplicateExperimentsByRunId(entries.filter(isExperimentRecord))
        const rebuilt = rebuildState(records)
        const best = (await readBest(root)) ?? rebuilt.bestResult
        let phaseState = await readPhaseState(root)

        if (phaseState.phase === "forecast" && !phaseState.budget_initialized) {
          phaseState = await lazyInitBudget(root, directory, phaseState)
        }

        const thresholds = phaseState.budget_thresholds ?? BudgetThresholds.parse({})
        const recentFailures = records
          .filter((record) => record.status !== "success" || record.decision === "discard")
          .slice(-5)
          .map((record) => ({
            run_id: record.run_id,
            model: record.model,
            failure_reason: record.failure_reason ?? record.status,
            decision: record.decision,
          }))
        const feedbackEntries = entries.filter(isExpertFeedbackRecord)
        const latestFeedback = feedbackEntries.at(-1) ?? null

        const pythonDir = await resolveForecastPythonProject(directory).catch(() => null)

        result = {
          current_best: best
            ? {
                run_id: best.run_id,
                model: best.model,
                metrics: best.metrics ?? {
                  mse: null,
                  mae: null,
                  wape: null,
                  mase: null,
                },
              }
            : null,
          per_family_best: rebuilt.perFamilyBest,
          recent_failures: recentFailures,
          budget: {
            remaining: Math.max(0, thresholds.max_experiments - rebuilt.experimentCount),
            total: thresholds.max_experiments,
            consecutive_no_improvement: phaseState.consecutive_no_improvement ?? 0,
            consecutive_crashes: phaseState.consecutive_crashes ?? 0,
          },
          expert_feedback: latestFeedback
            ? {
                text: latestFeedback.text,
                timestamp: latestFeedback.timestamp,
              }
            : null,
          auto_mode: phaseState.auto_mode,
          adjustment: phaseState.adjustment,
          reflection: phaseState.reflection,
          prediction_prior: phaseState.prediction_prior,
          python_dir: pythonDir,
        }
        break
      }

      case "append": {
        await ensureTaskDefined(directory)
        if (!params.record) {
          throw new Error("forecast_state append requires record")
        }
        const existingRecords = await readHistory(directory)
        if (existingRecords.some((record) => record.run_id === params.record?.run_id)) {
          result = {
            appended: false,
            run_id: params.record.run_id,
            reason: "run_id_already_recorded",
          }
          break
        }
        const historyPath = path.join(root, "history.jsonl")
        await appendFile(historyPath, `${JSON.stringify(params.record)}\n`, "utf-8")
        result = {
          appended: true,
          run_id: params.record.run_id,
        }
        break
      }

      case "best": {
        await ensureTaskDefined(directory)
        const bestPath = path.join(root, "best.json")
        if (params.update_best) {
          if (!params.record) {
            throw new Error("forecast_state best requires record when update_best is true")
          }
          await Filesystem.writeJson(bestPath, params.record)
          result = {
            updated: true,
            best: params.record,
          }
          break
        }

        result = (await readBest(root)) ?? { message: "No best result yet" }
        break
      }

      case "resume": {
        await ensureTaskDefined(directory)
        const records = await readHistory(directory)
        const rebuilt = rebuildState(records)
        result = {
          completedRunIds: rebuilt.completedRunIds,
          bestResult: (await readBest(root)) ?? rebuilt.bestResult,
          budget: (await readBudget(root)) ?? rebuilt.budget,
          experimentCount: rebuilt.experimentCount,
        }
        break
      }

      case "budget": {
        await ensureTaskDefined(directory)
        const budgetPath = path.join(root, "budget.json")
        const current =
          (await readBudget(root)) ??
          BudgetState.parse({
            experiment_count: 0,
            total_elapsed_seconds: 0,
            last_updated: new Date().toISOString(),
          })

        if (params.budget_increment) {
          const updated = BudgetState.parse({
            experiment_count: current.experiment_count + 1,
            total_elapsed_seconds: current.total_elapsed_seconds + params.budget_increment.elapsed_seconds,
            last_updated: new Date().toISOString(),
          })
          await Filesystem.writeJson(budgetPath, updated)
          result = updated
          break
        }

        result = current
        break
      }

      case "init": {
        const dirs = ["skills", "runs", "reports", "viz", "adjustments", "reflection"]
        for (const dir of dirs) {
          await mkdir(path.join(root, dir), { recursive: true })
        }

        let pythonProjectLinked = false
        const workspacePython = path.join(directory, "python")
        if (
          !(await Filesystem.exists(path.join(workspacePython, "pyproject.toml"))) &&
          !(await Filesystem.exists(workspacePython))
        ) {
          const pythonProject = await resolveForecastPythonProject(directory)
          await symlink(pythonProject, workspacePython, process.platform === "win32" ? "junction" : "dir")
          pythonProjectLinked = true
        }

        const statePath = path.join(root, "STATE.md")
        if (!(await Filesystem.exists(statePath))) {
          const initialState = defaultPhaseState()
          const task = await Filesystem.readJson<Record<string, unknown>>(path.join(root, "task.json")).catch(
            () => null,
          )
          if (typeof task?.reflection_enabled === "boolean") {
            initialState.reflection.enabled = task.reflection_enabled
          }
          await writePhaseState(root, initialState, "## Initialized\n\nForecast session initialized.")
        }

        const historyPath = path.join(root, "history.jsonl")
        if (!(await Filesystem.exists(historyPath))) {
          await writeFile(historyPath, "", "utf-8")
        }

        const bestPath = path.join(root, "best.json")
        if (!(await Filesystem.exists(bestPath))) {
          await writeFile(bestPath, "null\n", "utf-8")
        }

        const configPath = path.join(directory, "adacast.json")
        let config: Record<string, unknown> = {}
        let configUpdated = false
        try {
          const raw = await readFile(configPath, "utf-8")
          config = JSON.parse(raw) as Record<string, unknown>
        } catch {
          // No existing config — create one.
        }

        const skills = (config.skills ?? {}) as Record<string, unknown>
        const existingPaths = Array.isArray(skills.paths)
          ? skills.paths.filter((value): value is string => typeof value === "string")
          : []

        if (!existingPaths.includes(".forecast/skills")) {
          skills.paths = [...existingPaths, ".forecast/skills"]
          config.skills = skills
          await writeFile(configPath, JSON.stringify(config, null, 2) + "\n", "utf-8")
          configUpdated = true
        }

        result = {
          initialized: true,
          directories: dirs,
          files: ["STATE.md", "history.jsonl", "best.json"],
          state_path: statePath,
          config_updated: configUpdated,
          python_project_linked: pythonProjectLinked,
        }
        break
      }

      case "transition": {
        if (!params.target_phase) {
          throw new Error("forecast_state transition requires target_phase")
        }

        const currentState = await readPhaseState(root)
        const validation = await validateTransition(root, currentState.phase, params.target_phase)
        if (!validation.ok) {
          throw new PhaseGateError({
            current_phase: currentState.phase,
            required_phase: params.target_phase,
            missing_preconditions: validation.missing,
          })
        }

        const now = new Date().toISOString()
        let newState: ForecastPhaseStateValue = {
          ...currentState,
          phase: params.target_phase,
          transitions: [...currentState.transitions, { from: currentState.phase, to: params.target_phase, at: now }],
        }
        const autoModeBoundary = autoModeBoundaryForTransition(params.target_phase)

        if (params.target_phase === "skills-review") {
          newState.pre_forecast_completed = true
          newState.pre_forecast_completed_at = now
        }

        if (params.target_phase === "skills-review") {
          await generateForecastContext(root, directory)
        }

        if (params.target_phase === "pre-forecast") {
          try {
            const taskRaw = await readFile(path.join(root, "task.json"), "utf-8")
            const task = JSON.parse(taskRaw) as { frozen_at?: unknown }
            if (typeof task.frozen_at === "string") {
              newState.task_frozen_at = task.frozen_at
            }
          } catch {
            // task.json timestamp extraction is best-effort only.
          }
        }

        let skillMemoryPublish: Awaited<ReturnType<typeof publishForecastSkillMemory>> | null = null
        let skillMemoryPublishError: string | null = null
        if (params.target_phase === "post-forecast") {
          try {
            skillMemoryPublish = await publishForecastSkillMemory({
              directory,
              reason: "forecast_state transition to post-forecast",
            })
            newState = {
              ...newState,
              skill_memory_published_at: skillMemoryPublish.published_at,
              skill_memory_manifest_path: skillMemoryPublish.manifest_path,
              skill_memory_published_count: skillMemoryPublish.count,
              skill_memory_publish_error: null,
            }
          } catch (error) {
            skillMemoryPublishError = error instanceof Error ? error.message : String(error)
            newState = {
              ...newState,
              skill_memory_published_at: null,
              skill_memory_manifest_path: null,
              skill_memory_published_count: 0,
              skill_memory_publish_error: skillMemoryPublishError,
            }
          }
        }

        if (currentState.auto_mode.enabled && autoModeBoundary) {
          newState = applyAutoModeBoundary(newState, autoModeBoundary, now)
        }

        const logEntryParts = [`Transitioned at ${now}.`]
        if (skillMemoryPublish) {
          logEntryParts.push(
            `Forecast skill memory publish count=${skillMemoryPublish.count}, manifest=${skillMemoryPublish.manifest_path ?? "none"}.`,
          )
        }
        if (skillMemoryPublishError) {
          logEntryParts.push(`Forecast skill memory publish failed: ${skillMemoryPublishError}`)
        }
        const logEntry = `## Transition: ${currentState.phase} -> ${params.target_phase}\n\n${logEntryParts.join("\n")}`
        await writePhaseState(root, newState, logEntry)

        let compactionQueued = false
        let continuationQueued = false
        if (params.target_phase === "post-forecast") {
          await mkdir(path.join(root, "viz"), { recursive: true })
          await mkdir(path.join(root, "reports"), { recursive: true })

          const reportPath = path.join(root, "reports", "final-report.md")
          if (!(await Filesystem.exists(reportPath))) {
            await writeFile(
              reportPath,
              "# Final Forecast Report\n\n_Pending: Critic agent will populate this report._\n",
              "utf-8",
            )
          }

          const explanationsPath = path.join(root, "reports", "historical-explanations.md")
          if (!(await Filesystem.exists(explanationsPath))) {
            await writeFile(
              explanationsPath,
              "# Historical Sample Explanations\n\n_Pending: Critic agent will search historical training data for similar patterns to explain predictions._\n",
              "utf-8",
            )
          }

          if (!currentState.auto_mode.enabled) {
            // Preserve the manual handoff path: compaction keeps the Critic's
            // context artifact-based even when Auto Mode is not driving agents.
            compactionQueued = await queuePostForecastCompaction(ctx)
          }
        }

        if (currentState.auto_mode.enabled && autoModeBoundary) {
          try {
            const handoffPrompt = await buildForecastAutoModeContinuationPromptFromArtifacts({
              forecastRoot: root,
              targetAgent: autoModeBoundary.to,
            })
            continuationQueued = await queueAutoModeContinuation({
              ctx,
              targetAgent: autoModeBoundary.to,
              prompt: handoffPrompt,
            })
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error)
            const pausedState = await readPhaseState(root)
            await pauseAutoMode(
              root,
              pausedState,
              `Auto Mode ${autoModeBoundary.from}->${autoModeBoundary.to} handoff failed: ${message}`,
            )
          }
        }

        result = {
          ok: true,
          phase: params.target_phase,
          previous: currentState.phase,
          ...(params.target_phase === "post-forecast" ? { compaction_queued: compactionQueued } : {}),
          ...(currentState.auto_mode.enabled && autoModeBoundary
            ? { continuation_queued: continuationQueued, next_agent: autoModeBoundary.to }
            : {}),
        }
        break
      }

      case "phase_gate": {
        if (!params.target_phase) {
          throw new Error("forecast_state phase_gate requires target_phase")
        }

        const currentState = await readPhaseState(root)
        if (currentState.phase !== params.target_phase) {
          throw new PhaseGateError({
            current_phase: currentState.phase,
            required_phase: params.target_phase,
            missing_preconditions: [`current phase is "${currentState.phase}", expected "${params.target_phase}"`],
          })
        }

        result = {
          ok: true,
          phase: currentState.phase,
        }
        break
      }

      case "confirm_skills": {
        const currentState = await readPhaseState(root)
        if (currentState.phase !== "skills-review") {
          throw new PhaseGateError({
            current_phase: currentState.phase,
            required_phase: "skills-review",
            missing_preconditions: ["skills can only be confirmed during skills-review phase"],
          })
        }

        const now = new Date().toISOString()
        const newState: ForecastPhaseStateValue = {
          ...currentState,
          skills_confirmed: true,
          skills_confirmed_at: now,
          skill_memory_published_at: null,
          skill_memory_manifest_path: null,
          skill_memory_published_count: null,
          skill_memory_publish_error: null,
        }
        const logEntry = `## Skills Confirmed\n\nUser confirmed the skill set at ${now}.`
        await writePhaseState(root, newState, logEntry)

        result = {
          ok: true,
          skills_confirmed: true,
          confirmed_at: now,
        }
        break
      }

      case "configure_auto_mode": {
        if (!params.auto_mode_update) {
          throw new Error("forecast_state configure_auto_mode requires auto_mode_update")
        }

        const currentState = await readPhaseState(root)
        const updatedAutoMode = mergeAutoModeState(currentState.auto_mode, params.auto_mode_update)
        const updatedState: ForecastPhaseStateValue = {
          ...currentState,
          auto_mode: updatedAutoMode,
        }

        await writePhaseState(
          root,
          updatedState,
          `## Auto Mode Updated\n\nEnabled=${updatedAutoMode.enabled}, state=${updatedAutoMode.state}, active_agent=${updatedAutoMode.active_agent ?? "none"}.`,
        )

        result = {
          ok: true,
          auto_mode: updatedAutoMode,
        }
        break
      }

      case "check_reflection": {
        await ensureTaskDefined(directory)
        const history = await readHistoryEntries(directory)
        const { lastExperimentRunId, hasReflection } = getLastReflectionStatus(history)

        if (lastExperimentRunId && !hasReflection) {
          throw new PhaseGateError({
            current_phase: "forecast",
            required_phase: "forecast",
            missing_preconditions: [
              `last experiment (run_id: ${lastExperimentRunId}) has no paired ReflectionRecord — call forecast_reflect before proposing the next experiment`,
            ],
          })
        }

        result = {
          ok: true,
          has_reflection: hasReflection,
          last_experiment_run_id: lastExperimentRunId,
        }
        break
      }

      case "update_stop_counters": {
        await ensureTaskDefined(directory)
        if (!params.stop_counter_input) {
          throw new Error("forecast_state update_stop_counters requires stop_counter_input")
        }
        const { improvement, crashed, run_id: requestedRunId } = params.stop_counter_input
        let phaseState = await readPhaseState(root)

        if (!phaseState.budget_initialized) {
          phaseState = await lazyInitBudget(root, directory, phaseState)
        }
        const thresholds = phaseState.budget_thresholds ?? BudgetThresholds.parse({})

        const entries = await readHistoryEntries(directory)
        const attempts = entries.filter(isExperimentRecord)
        const records = deduplicateExperimentsByRunId(attempts)
        const experimentCount = records.length
        const latestAttempt = attempts.at(-1) ?? null
        const latestRunId = latestAttempt?.run_id ?? null
        const latestAttemptIndex = latestAttempt ? experimentAttemptIndex(latestAttempt) : null
        if (requestedRunId && latestRunId && requestedRunId !== latestRunId) {
          throw new Error(
            `forecast_state update_stop_counters expected latest run_id ${latestRunId}, received ${requestedRunId}`,
          )
        }

        let noImprovementCount = phaseState.consecutive_no_improvement ?? 0
        let crashCount = phaseState.consecutive_crashes ?? 0
        let lastAccountedIndex = -1
        if (phaseState.stop_counter_last_run_id) {
          const accountedAttemptIndex = phaseState.stop_counter_last_attempt_index ?? 1
          for (let index = attempts.length - 1; index >= 0; index -= 1) {
            const record = attempts[index]
            if (
              record.run_id === phaseState.stop_counter_last_run_id &&
              experimentAttemptIndex(record) === accountedAttemptIndex
            ) {
              lastAccountedIndex = index
              break
            }
          }
        }
        const pendingRecords = lastAccountedIndex >= 0 ? attempts.slice(lastAccountedIndex + 1) : attempts
        const pendingStartIndex = attempts.length - pendingRecords.length
        let bestMseBeforePending: number | null = null

        for (const record of attempts.slice(0, pendingStartIndex)) {
          if (record.status !== "success" || record.metrics?.mse == null) continue
          bestMseBeforePending =
            bestMseBeforePending == null ? record.metrics.mse : Math.min(bestMseBeforePending, record.metrics.mse)
        }

        const applyOutcome = (didImprove: boolean, didCrash: boolean) => {
          if (didImprove) {
            noImprovementCount = 0
            crashCount = 0
          } else if (didCrash) {
            crashCount += 1
          } else {
            noImprovementCount += 1
            crashCount = 0
          }
        }

        if (pendingRecords.length === 0 && records.length === 0) {
          // Backward-compatible fallback for callers that update counters before
          // persisting an ExperimentRecord.
          applyOutcome(improvement, crashed)
        } else {
          for (const [index, record] of pendingRecords.entries()) {
            const isLatestPending = index === pendingRecords.length - 1
            const inferredImprovement =
              record.status === "success" &&
              record.metrics?.mse != null &&
              (bestMseBeforePending == null || record.metrics.mse < bestMseBeforePending)
            applyOutcome(isLatestPending ? improvement : inferredImprovement, record.status === "crash")

            if (record.status === "success" && record.metrics?.mse != null) {
              bestMseBeforePending =
                bestMseBeforePending == null ? record.metrics.mse : Math.min(bestMseBeforePending, record.metrics.mse)
            }
          }
        }

        let stopReason: "no_improvement" | "budget_exhausted" | "crash_threshold" | null = null
        let hitlTriggered = false

        if (experimentCount >= thresholds.max_experiments) {
          stopReason = "budget_exhausted"
        } else if (crashCount >= thresholds.crash_threshold) {
          stopReason = "crash_threshold"
        } else if (noImprovementCount >= thresholds.no_improvement_rounds) {
          hitlTriggered = true
        }

        const autoModePauseReason = phaseState.auto_mode.enabled
          ? pauseReasonForStop({
              stopReason,
              hitlTriggered,
              noImprovementCount,
              crashCount,
              experimentCount,
              thresholds,
            })
          : null

        const updatedState: ForecastPhaseStateValue = {
          ...phaseState,
          consecutive_no_improvement: noImprovementCount,
          consecutive_crashes: crashCount,
          stop_counter_last_run_id: latestRunId ?? phaseState.stop_counter_last_run_id ?? null,
          stop_counter_last_attempt_index: latestAttemptIndex ?? phaseState.stop_counter_last_attempt_index ?? null,
          ...(autoModePauseReason
            ? {
                auto_mode: mergeAutoModeState(phaseState.auto_mode, {
                  active_agent: phaseState.auto_mode.active_agent ?? "forecaster",
                  state: "paused",
                  pause_reason: autoModePauseReason,
                }),
              }
            : {}),
          ...(stopReason ? { stop_reason: stopReason } : {}),
        }

        const logParts = [`no_improvement=${noImprovementCount}`, `crashes=${crashCount}`]
        if (stopReason) logParts.push(`STOP: ${stopReason}`)
        if (hitlTriggered) logParts.push("HITL triggered")
        if (phaseState.auto_mode.enabled && hitlTriggered && !autoModePauseReason) {
          logParts.push("Auto Mode automatic no-improvement stop; continue to reflection and adjustment decision")
        }
        if (autoModePauseReason) logParts.push(`Auto Mode paused: ${autoModePauseReason}`)
        await writePhaseState(root, updatedState, `## Stop Counter Update\n\n${logParts.join(", ")}`)

        result = StopCheckResult.parse({
          stop: stopReason !== null,
          reason: stopReason ?? "continue",
          hitl_triggered: hitlTriggered,
          counters: {
            consecutive_no_improvement: noImprovementCount,
            consecutive_crashes: crashCount,
            experiments_remaining: Math.max(0, thresholds.max_experiments - experimentCount),
          },
        })
        break
      }

      case "record_expert_feedback": {
        await ensureTaskDefined(directory)
        if (!params.expert_feedback_text) {
          throw new Error("forecast_state record_expert_feedback requires expert_feedback_text")
        }

        const now = new Date().toISOString()
        const feedbackRecord: ExpertFeedbackRecordValue = {
          type: "expert_feedback",
          text: params.expert_feedback_text,
          timestamp: now,
        }

        const historyPath = path.join(root, "history.jsonl")
        await appendFile(historyPath, `${JSON.stringify(feedbackRecord)}\n`, "utf-8")

        const phaseState = await readPhaseState(root)
        const latestAttempt = (await readHistoryEntries(directory)).filter(isExperimentRecord).at(-1) ?? null
        const updatedState: ForecastPhaseStateValue = {
          ...phaseState,
          consecutive_no_improvement: 0,
          stop_counter_last_run_id: latestAttempt?.run_id ?? null,
          stop_counter_last_attempt_index: latestAttempt ? experimentAttemptIndex(latestAttempt) : null,
        }
        await writePhaseState(
          root,
          updatedState,
          `## Expert Feedback Recorded\n\nReset consecutive_no_improvement to 0 at ${now}.`,
        )

        result = {
          recorded: true,
          timestamp: now,
          counter_reset: true,
          consecutive_no_improvement: 0,
        }
        break
      }
    }

    return {
      title: `Forecast state: ${params.subcommand}`,
      metadata: { subcommand: params.subcommand },
      output: JSON.stringify(result),
    }
  },
})
