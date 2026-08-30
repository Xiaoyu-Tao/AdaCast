import type { Plugin as PluginInstance } from "@adacast/plugin"
import { readFile } from "fs/promises"
import matter from "gray-matter"
import path from "path"
import { ExperimentRecord, ExpertFeedbackRecord, ForecastPhaseState, ReflectionRecord } from "@/tool/forecast/types"

export type ForecastHistorySummary = {
  experimentCount: number
  reflectionCount: number
  expertFeedbackCount: number
  lastExperiment: {
    run_id: string
    model: string
    status: string
  } | null
  lastReflection: {
    run_id: string
    decision: string
    effective: boolean
  } | null
}

export type ForecastHandoffTarget = "forecaster" | "critic"

export function summarizeForecastHistory(raw: string): ForecastHistorySummary {
  const summary: ForecastHistorySummary = {
    experimentCount: 0,
    reflectionCount: 0,
    expertFeedbackCount: 0,
    lastExperiment: null,
    lastReflection: null,
  }
  const experimentRunIds = new Set<string>()

  for (const line of raw.split("\n").filter(Boolean)) {
    try {
      const parsed = JSON.parse(line)

      const feedback = ExpertFeedbackRecord.safeParse(parsed)
      if (feedback.success) {
        summary.expertFeedbackCount += 1
        continue
      }

      const reflection = ReflectionRecord.safeParse(parsed)
      if (reflection.success) {
        summary.reflectionCount += 1
        summary.lastReflection = {
          run_id: reflection.data.run_id,
          decision: reflection.data.decision,
          effective: reflection.data.effective,
        }
        continue
      }

      const experiment = ExperimentRecord.safeParse(parsed)
      if (experiment.success) {
        if (!experimentRunIds.has(experiment.data.run_id)) {
          experimentRunIds.add(experiment.data.run_id)
          summary.experimentCount += 1
        }
        summary.lastExperiment = {
          run_id: experiment.data.run_id,
          model: experiment.data.model,
          status: experiment.data.status,
        }
      }
    } catch {
      // Ignore invalid JSON lines so compaction can still proceed.
    }
  }

  return summary
}

export function buildForecastCompactionPrompt(input: {
  targetAgent?: ForecastHandoffTarget
  stateText: string
  contextText?: string
  taskText?: string
  bestText?: string
  history: ForecastHistorySummary
}): string {
  const { stateText, contextText, taskText, bestText, history } = input
  const targetAgent = input.targetAgent ?? "critic"
  const targetName = targetAgent === "forecaster" ? "Forecaster" : "Critic"
  const sourceName = targetAgent === "forecaster" ? "Planner" : "Forecaster"
  const artifactList =
    targetAgent === "forecaster"
      ? "`.forecast/STATE.md`, `.forecast/CONTEXT.md`, `.forecast/task.json`, or `.forecast/history.jsonl`"
      : "`.forecast/STATE.md`, `.forecast/best.json`, or `.forecast/history.jsonl`"
  const outcome =
    targetAgent === "forecaster"
      ? "the frozen task, durable constraints, modeling directions, and the remaining Forecaster experiment work"
      : "forecast outcomes, reflections, and the remaining Critic work"

  const lines = [
    `Create the compacted handoff prompt for the ${targetName} agent using only the \`.forecast/\` artifacts below.`,
    `Do not mention or rely on prior ${sourceName} conversation history, tool traces, or unrelated repository files.`,
    `The resulting compacted context should describe ${outcome} only.`,
    "",
    "## Artifact Rules",
    `- Only summarize information present in ${artifactList}.`,
    `- Treat these artifacts as the entire durable context for the ${targetName} handoff.`,
    "- Keep the handoff concise and operational.",
    "",
    "## .forecast/STATE.md",
    "```md",
    stateText.trim() || "_Empty state file._",
    "```",
    "",
  ]

  if (targetAgent === "forecaster") {
    lines.push(
      "## .forecast/CONTEXT.md",
      "```md",
      contextText?.trim() || "_No durable forecast context generated yet._",
      "```",
      "",
      "## .forecast/task.json",
      "```json",
      taskText?.trim() || "null",
      "```",
      "",
    )
  }

  if (targetAgent === "critic") {
    lines.push("## .forecast/best.json", "```json", bestText?.trim() || "null", "```", "")
  }

  lines.push(
    "## History Summary",
    `- Experiments recorded: ${history.experimentCount}`,
    `- Reflections recorded: ${history.reflectionCount}`,
    `- Expert feedback entries: ${history.expertFeedbackCount}`,
    history.lastExperiment
      ? `- Last experiment: ${history.lastExperiment.model} (${history.lastExperiment.run_id}) status=${history.lastExperiment.status}`
      : "- Last experiment: none recorded",
    history.lastReflection
      ? `- Last reflection: ${history.lastReflection.run_id} decision=${history.lastReflection.decision} effective=${history.lastReflection.effective}`
      : "- Last reflection: none recorded",
    "",
    `Produce the compacted continuation prompt that the ${targetName} should receive next.`,
  )

  return lines.join("\n")
}

export function buildForecastAutoModeContinuationPrompt(input: {
  targetAgent: ForecastHandoffTarget
  stateText: string
  contextText?: string
  taskText?: string
  bestText?: string
  history: ForecastHistorySummary
}): string {
  const { stateText, contextText, taskText, bestText, history } = input
  const targetName = input.targetAgent === "forecaster" ? "Forecaster" : "Critic"
  const sourceName = input.targetAgent === "forecaster" ? "Planner" : "Forecaster"

  const lines = [
    `Auto Mode continuation for the ${targetName} agent.`,
    input.targetAgent === "forecaster"
      ? "Auto Mode is enabled. This is the automatic Planner -> Forecaster continuation."
      : "Auto Mode is enabled. This is the automatic Forecaster -> Critic continuation.",
    "",
    `You are receiving the ${sourceName} -> ${targetName} handoff for an active Auto Mode forecast workflow.`,
    "Continue from the durable `.forecast/` artifacts below. Do not rely on prior conversation history, tool traces, or unrelated repository files.",
    "Do not call the question tool or wait for user input for Auto Mode workflow decisions. Post-adjustment choices are made autonomously by the active LLM through the auto_* forecast_adjustment_decision subcommands.",
    "Only stop for a hard blocker, denied permission, unrecoverable runtime/tool failure, missing critical artifact, safety restriction, or an explicitly out-of-scope user verification checkpoint.",
    "",
    "## Auto Mode Contract",
  ]

  if (input.targetAgent === "forecaster") {
    lines.push(
      "- Start Forecaster work now by reading `forecast_state`, loading frozen Skills, and running the experiment loop.",
      "- Do not ask the user to confirm routine Forecaster steps, including handoff receipt, skill review status, experiment proposals, or no-improvement HITL/expert-feedback checkpoints.",
      "- Do not ask for handoff, skill-review, experiment-start, HITL, or expert-feedback confirmation.",
      "- Before entering `post-forecast`, call `forecast_adjustment_decision auto_start` unless `.forecast/STATE.md` already records `adjustment.user_decision` as `disabled`, `enabled_a`, or `enabled_b`. Choose disabled, lightweight prediction reasoning, or deep prediction reasoning from task context, leakage-safe pre-analysis, current and per-family aggregate held-out validation performance, experiment stability, supplied exogenous context, and expected adjustment complexity; include a concrete rationale. Do not infer held-out validation/test target behavior from later split timestamps.",
      "- If lightweight prediction reasoning exhausts its candidate budget, call `forecast_adjustment_decision auto_escalation`: switch to deep prediction reasoning only when validation-front diagnostics contain credible untested blind-safe structure that warrants broader analysis; otherwise stop adjustment.",
      "- Design concrete adjustment rules only from canonical validation-front fit diagnostics. Pre-analysis, held-out validation artifacts, test artifacts, and domain intuition are not rule evidence; held-out validation aggregate metrics may only accept or reject a frozen candidate.",
      '- If `forecast_state update_stop_counters` returns `hitl_triggered: true` while Auto Mode is enabled, do not request expert input. Treat it as an automatic no-improvement stop: call `forecast_reflect` with `decision: "stop"`, then handle `forecast_adjustment_decision auto_start` before calling `forecast_state transition` with `target_phase: "post-forecast"`.',
      "- Keep using `forecast_reflect` after every experiment round and preserve all normal hard-stop handling for `budget_exhausted` and `crash_threshold`.",
      "",
      "## .forecast/STATE.md",
      "```md",
      stateText.trim() || "_Empty state file._",
      "```",
      "",
      "## .forecast/CONTEXT.md",
      "```md",
      contextText?.trim() || "_No durable forecast context generated yet._",
      "```",
      "",
      "## .forecast/task.json",
      "```json",
      taskText?.trim() || "null",
      "```",
      "",
    )
  } else {
    lines.push(
      "- Start Critic work now by reading `forecast_state` and the `.forecast/` artifacts, then execute test-set evaluation, visualizations, historical pattern search, and final report writing.",
      "- Do not ask the user to confirm routine Critic steps, including test-set evaluation, visualization generation, historical pattern search, or final report writing.",
      "- Do not ask for handoff, report-generation, visualization, or test-evaluation confirmation.",
      "- If a locked active adjustment skill exists after raw test-best selection, call `forecast_adjustment_decision prepare_test_applications` with every successful run id in the current Critic raw test collection. Then call `forecast_adjustment_decision auto_test_application` without a run id so every exact validation-matched model defaults to apply, including eligible models that are not the raw test-best. Apply and evaluate each apply record independently. Never apply a skill to a non-matching model spec.",
      "- If an artifact needed for a Critic step is missing, inspect the durable forecast files first and proceed when recoverable.",
      "",
      "## .forecast/STATE.md",
      "```md",
      stateText.trim() || "_Empty state file._",
      "```",
      "",
      "## .forecast/task.json",
      "```json",
      taskText?.trim() || "null",
      "```",
      "",
      "## .forecast/best.json",
      "```json",
      bestText?.trim() || "null",
      "```",
      "",
    )
  }

  lines.push(
    "## History Summary",
    `- Experiments recorded: ${history.experimentCount}`,
    `- Reflections recorded: ${history.reflectionCount}`,
    `- Expert feedback entries: ${history.expertFeedbackCount}`,
    history.lastExperiment
      ? `- Last experiment: ${history.lastExperiment.model} (${history.lastExperiment.run_id}) status=${history.lastExperiment.status}`
      : "- Last experiment: none recorded",
    history.lastReflection
      ? `- Last reflection: ${history.lastReflection.run_id} decision=${history.lastReflection.decision} effective=${history.lastReflection.effective}`
      : "- Last reflection: none recorded",
    "",
    `Continue as the ${targetName} agent now.`,
  )

  return lines.join("\n")
}

export async function buildForecastHandoffPromptFromArtifacts(input: {
  forecastRoot: string
  targetAgent: ForecastHandoffTarget
}): Promise<string> {
  const [stateText, contextText, taskText, bestText, historyRaw] = await Promise.all([
    readFile(path.join(input.forecastRoot, "STATE.md"), "utf-8").catch(() => ""),
    readFile(path.join(input.forecastRoot, "CONTEXT.md"), "utf-8").catch(() => ""),
    readFile(path.join(input.forecastRoot, "task.json"), "utf-8").catch(() => "null\n"),
    readFile(path.join(input.forecastRoot, "best.json"), "utf-8").catch(() => "null\n"),
    readFile(path.join(input.forecastRoot, "history.jsonl"), "utf-8").catch(() => ""),
  ])

  return buildForecastCompactionPrompt({
    targetAgent: input.targetAgent,
    stateText,
    contextText,
    taskText,
    bestText,
    history: summarizeForecastHistory(historyRaw),
  })
}

export async function buildForecastAutoModeContinuationPromptFromArtifacts(input: {
  forecastRoot: string
  targetAgent: ForecastHandoffTarget
}): Promise<string> {
  const [stateText, contextText, taskText, bestText, historyRaw] = await Promise.all([
    readFile(path.join(input.forecastRoot, "STATE.md"), "utf-8").catch(() => ""),
    readFile(path.join(input.forecastRoot, "CONTEXT.md"), "utf-8").catch(() => ""),
    readFile(path.join(input.forecastRoot, "task.json"), "utf-8").catch(() => "null\n"),
    readFile(path.join(input.forecastRoot, "best.json"), "utf-8").catch(() => "null\n"),
    readFile(path.join(input.forecastRoot, "history.jsonl"), "utf-8").catch(() => ""),
  ])

  return buildForecastAutoModeContinuationPrompt({
    targetAgent: input.targetAgent,
    stateText,
    contextText,
    taskText,
    bestText,
    history: summarizeForecastHistory(historyRaw),
  })
}

function normalizePhaseStateFrontmatter(data: Record<string, unknown>): Record<string, unknown> {
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

  return {
    ...data,
    skills_confirmed_at: normalizeValue(data.skills_confirmed_at),
    skill_memory_published_at: normalizeValue(data.skill_memory_published_at) ?? null,
    skill_memory_manifest_path: typeof data.skill_memory_manifest_path === "string" ? data.skill_memory_manifest_path : null,
    skill_memory_published_count:
      typeof data.skill_memory_published_count === "number" ? data.skill_memory_published_count : null,
    skill_memory_publish_error: typeof data.skill_memory_publish_error === "string" ? data.skill_memory_publish_error : null,
    pre_forecast_completed_at: normalizeValue(data.pre_forecast_completed_at),
    task_frozen_at: normalizeValue(data.task_frozen_at),
    auto_mode: autoMode,
    transitions: Array.isArray(data.transitions)
      ? data.transitions.map((transition) => {
          if (!transition || typeof transition !== "object") return transition
          const record = transition as Record<string, unknown>
          return { ...record, at: normalizeValue(record.at) }
        })
      : data.transitions,
  }
}

function resolveForecastCompactionTarget(
  state: ReturnType<typeof ForecastPhaseState.parse>,
): ForecastHandoffTarget | null {
  if (state.auto_mode.enabled) {
    if (state.auto_mode.active_agent === "forecaster" || state.auto_mode.active_agent === "critic") {
      return state.auto_mode.active_agent
    }

    const lastHandoff = state.auto_mode.handoffs.at(-1)
    if (lastHandoff?.to === "forecaster" || lastHandoff?.to === "critic") {
      return lastHandoff.to
    }
  }

  if (state.phase === "post-forecast") {
    return "critic"
  }

  return null
}

export const ForecastCompactionPlugin: PluginInstance = async (input) => {
  return {
    "experimental.session.compacting": async (_hookInput, output) => {
      const forecastRoot = path.join(input.directory, ".forecast")
      const statePath = path.join(forecastRoot, "STATE.md")
      const stateRaw = await readFile(statePath, "utf-8").catch(() => null)
      if (!stateRaw) return

      let state
      try {
        state = ForecastPhaseState.parse(normalizePhaseStateFrontmatter(matter(stateRaw).data))
      } catch {
        return
      }

      const targetAgent = resolveForecastCompactionTarget(state)
      if (!targetAgent) return

      output.prompt = await buildForecastHandoffPromptFromArtifacts({
        forecastRoot,
        targetAgent,
      })
    },
  }
}
