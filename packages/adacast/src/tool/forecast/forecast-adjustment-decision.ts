import path from "path"
import z from "zod"
import { Instance } from "@/project/instance"
import { Question } from "@/question"
import { Filesystem } from "@/util/filesystem"
import { Tool } from "../tool"
import { validationPostForecastTestSpecDifferences } from "./execution-contract"
import DESCRIPTION from "./forecast-adjustment-decision.txt"
import {
  AdjustmentEvalResult,
  AdjustmentPolicy,
  AdjustmentSkillManifest,
  EvalResult,
  parseStoredAdjustmentPolicy,
} from "./types"
import { readBoundModelArtifactIdentity } from "./model-artifact"
import {
  adjustmentRoot,
  ensureAdjustmentDirs,
  forecastRoot,
  forecastPath,
  readJsonFile,
  readPhaseState,
  runIdIsSafe,
  writeJsonFile,
  writePhaseState,
} from "./forecast-adjustment-common"
import {
  assertNoRetainableValidationCandidate,
  assertUserPriorMirrorResolved,
} from "./forecast-adjustment-acceptance"
import { assertFrozenTaskIntegrity } from "./forecast-task-integrity"

const parameters = z.object({
  subcommand: z.enum([
    "read",
    "ask_start",
    "auto_start",
    "ask_escalation",
    "auto_escalation",
    "prepare_test_applications",
    "ask_test_application",
    "auto_test_application",
  ]),
  reason: z
    .string()
    .optional()
    .describe("Decision rationale; required for Auto Mode start/escalation decisions and shown for manual escalation"),
  auto_start_choice: z
    .enum(["disabled", "enabled_a", "enabled_b"])
    .optional()
    .describe("Auto Mode LLM choice for auto_start"),
  auto_escalation_choice: z
    .enum(["upgrade_b", "stop"])
    .optional()
    .describe("Auto Mode LLM choice after lightweight prediction reasoning fails validation"),
  auto_test_application_choice: z
    .enum(["apply", "skip"])
    .optional()
    .describe("Deprecated compatibility input; Auto Mode now defaults to apply for every eligible test run"),
  test_run_ids: z
    .array(z.string())
    .min(1)
    .optional()
    .describe("Successful post-forecast test run ids in the current Critic test collection"),
  raw_test_run_id: z
    .string()
    .optional()
    .describe("One eligible post-forecast test run id; omit to process every pending eligible run"),
})

const START_OPTION_DISABLED = "不开启"
const START_OPTION_A = "轻量预测推理"
const START_OPTION_B = "深度预测推理"
const ESCALATE_OPTION_B = "切换至深度预测推理"
const ESCALATE_OPTION_STOP = "结束生成式预测推理"
const TEST_OPTION_APPLY = "应用生成式预测推理"
const TEST_OPTION_SKIP = "保持原始结果"

function policyPath(): string {
  return path.join(adjustmentRoot(), "policy.json")
}

async function readPolicy(): Promise<z.infer<typeof AdjustmentPolicy>> {
  try {
    return parseStoredAdjustmentPolicy(await readJsonFile(policyPath()))
  } catch {
    return AdjustmentPolicy.parse({})
  }
}

function decisionLocked(state: Awaited<ReturnType<typeof readPhaseState>>): boolean {
  return Boolean(
    state.adjustment.forecaster_locked &&
      (state.adjustment.active_skill_version || state.adjustment.no_effective_skill),
  )
}

function assertForecastDecisionAllowed(operation: string, state: Awaited<ReturnType<typeof readPhaseState>>): void {
  if (state.phase !== "forecast") {
    throw new Error(`forecast_adjustment_decision ${operation} is only allowed during the forecast phase`)
  }
  const startDecision = operation === "ask_start" || operation === "auto_start"
  if (startDecision && state.adjustment.user_decision !== "pending") {
    throw new Error(`forecast_adjustment_decision ${operation} is only allowed before the start decision is recorded`)
  }
  if (state.adjustment.source_model_set_locked && startDecision) {
    throw new Error(`forecast_adjustment_decision ${operation} must happen before freezing adjustment source models`)
  }
  if (decisionLocked(state)) {
    throw new Error(`forecast_adjustment_decision ${operation} is not allowed after the adjustment decision is locked`)
  }
}

function assertManualDecisionMode(operation: string, state: Awaited<ReturnType<typeof readPhaseState>>): void {
  if (state.auto_mode.enabled) {
    throw new Error(
      `forecast_adjustment_decision ${operation} is unavailable while Auto Mode is enabled; use the matching auto_* subcommand`,
    )
  }
}

function assertAutoDecisionMode(operation: string, state: Awaited<ReturnType<typeof readPhaseState>>): void {
  if (!state.auto_mode.enabled) {
    throw new Error(`forecast_adjustment_decision ${operation} is only allowed while Auto Mode is enabled`)
  }
}

function requireAutoReason(reason: string | undefined, operation: string): string {
  const value = reason?.trim()
  if (!value || value.length < 12) {
    throw new Error(
      `forecast_adjustment_decision ${operation} requires a concrete LLM rationale of at least 12 characters`,
    )
  }
  return value
}

function policyForStartDecision(
  choice: "disabled" | "enabled_a" | "enabled_b",
  current: z.infer<typeof AdjustmentPolicy>,
  now: string,
): z.infer<typeof AdjustmentPolicy> {
  const { configured_at: _configuredAt, locked_at: _lockedAt, ...basePolicy } = current
  if (choice === "disabled") {
    return AdjustmentPolicy.parse({
      ...basePolicy,
      enabled: false,
      mode: "off",
      strategy: null,
      configured_at: now,
      auto_confirm_if_validation_passes: false,
    })
  }

  const strategy = choice === "enabled_a" ? "a_lightweight" : "b_comprehensive"
  return AdjustmentPolicy.parse({
    ...basePolicy,
    enabled: true,
    mode: "rule_skill",
    strategy,
    max_candidate_iterations:
      choice === "enabled_a" ? 2 : current.strategy === "a_lightweight" ? current.max_candidate_iterations + 5 : 5,
    auto_confirm_if_validation_passes: false,
    configured_at: now,
  })
}

async function writeStartDecision(
  choice: "disabled" | "enabled_a" | "enabled_b",
  now: string,
  logLabel: string,
  options: {
    resetAdjustmentArtifacts?: boolean
    actor: "user" | "llm_auto"
    reason: string
  },
) {
  const currentPolicy = await readPolicy()
  const policy = policyForStartDecision(choice, currentPolicy, now)
  await writeJsonFile(policyPath(), policy)

  const state = await readPhaseState()
  const resetArtifacts = options.resetAdjustmentArtifacts ?? true
  await writePhaseState(
    {
      ...state,
      adjustment: {
        ...state.adjustment,
        enabled: policy.enabled,
        mode: policy.mode,
        strategy: policy.strategy,
        user_decision: choice,
        user_decision_at: now,
        user_decision_actor: options.actor,
        user_decision_reason: options.reason,
        policy_path: forecastPath("adjustments", "policy.json"),
        ...(resetArtifacts
          ? {
              source_model_set_locked: false,
              source_model_set_path: null,
              fit_run_manifest_path: null,
              fit_runs_ready: false,
              active_skill_version: null,
              active_skill_path: null,
            }
          : {}),
        forecaster_locked: false,
        no_effective_skill: false,
        no_effective_skill_reason: null,
        test_collection_run_ids: [],
        test_eligible_run_ids: [],
        test_application_records: [],
        critic_user_confirmed: null,
        test_application_decision: choice === "disabled" ? "skip" : "pending",
        test_application_decision_at: choice === "disabled" ? now : null,
        test_application_decision_actor: choice === "disabled" ? options.actor : null,
        test_application_decision_reason:
          choice === "disabled" ? `Post-adjustment was disabled at the start decision: ${options.reason}` : null,
        test_application_run_id: null,
        test_adjustment_locked: false,
        test_adjustment_run_id: null,
        test_adjustment_candidate_id: null,
      },
    },
    `## Adjustment User Decision\n\n${logLabel} at ${now}.`,
  )

  return { decision: choice, decision_actor: options.actor, decision_reason: options.reason, policy }
}

async function askSingle(ctx: Parameters<typeof Question.ask>[0]): Promise<string> {
  const answers = await Question.ask(ctx)
  const answer = answers[0]?.[0]
  if (!answer) throw new Error("Question returned no answer")
  return answer
}

function assertOneOf(answer: string, allowed: readonly string[], question: string): void {
  if (!allowed.includes(answer)) {
    throw new Error(`${question} returned unexpected answer "${answer}"`)
  }
}

function formatMetric(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? Number(value).toPrecision(5) : "N/A"
}

function formatValidationSummary(result: z.infer<typeof AdjustmentEvalResult>): string {
  const raw = result.raw_metrics
  const adjusted = result.adjusted_metrics
  const delta = result.delta_vs_raw
  return [
    `MSE: raw ${formatMetric(raw?.mse)} -> adjusted ${formatMetric(adjusted.mse)} (delta ${formatMetric(delta.mse)})`,
    `MAE: raw ${formatMetric(raw?.mae)} -> adjusted ${formatMetric(adjusted.mae)} (delta ${formatMetric(delta.mae)})`,
    `WAPE: raw ${formatMetric(raw?.wape)} -> adjusted ${formatMetric(adjusted.wape)} (delta ${formatMetric(delta.wape)})`,
  ].join("\n")
}

async function readActiveValidationResult(): Promise<{
  manifest: z.infer<typeof AdjustmentSkillManifest>
  validations: Array<z.infer<typeof AdjustmentEvalResult>>
}> {
  const manifest = AdjustmentSkillManifest.parse(
    await readJsonFile(path.join(adjustmentRoot(), "skills", "active", "manifest.json")),
  )
  if (manifest.status !== "locked") {
    throw new Error(`Active adjustment skill must be locked before test decision; current status is ${manifest.status}`)
  }
  if (manifest.validation_results.length === 0) {
    throw new Error("Active adjustment skill manifest has no validation result")
  }
  const validations = []
  for (const validationPath of manifest.validation_results) {
    const absoluteValidationPath = path.isAbsolute(validationPath)
      ? validationPath
      : path.join(Instance.directory, validationPath)
    const validation = AdjustmentEvalResult.parse(await readJsonFile(absoluteValidationPath))
    if (validation.split !== "val" || validation.candidate_id !== manifest.candidate_id) {
      throw new Error("Active adjustment skill validation result does not match the locked candidate")
    }
    if (!validation.acceptance_decision?.accepted) {
      throw new Error("Active adjustment skill validation result is not accepted")
    }
    validations.push(validation)
  }
  return { manifest, validations }
}

async function assertRawTestBestRunReady(runId: string): Promise<{
  evalResult: z.infer<typeof EvalResult>
  spec: Record<string, unknown>
}> {
  if (!runIdIsSafe(runId)) throw new Error(`Invalid raw_test_run_id "${runId}"`)

  const runRoot = path.join(forecastRoot(), "runs", runId)
  const spec = z.record(z.string(), z.unknown()).parse(await readJsonFile(path.join(runRoot, "spec.json")))
  if (spec.phase !== "post-forecast" || spec.eval_split !== "test") {
    throw new Error(
      "forecast_adjustment_decision test application decision requires raw_test_run_id to reference a post-forecast test run",
    )
  }
  const predictionsPath = path.join(runRoot, "predictions.csv")
  if (!(await Filesystem.exists(predictionsPath))) {
    throw new Error(
      `forecast_adjustment_decision test application decision requires raw test predictions.csv: ${predictionsPath}`,
    )
  }

  const evalResult = EvalResult.parse(await readJsonFile(path.join(runRoot, "eval.json")))
  if (evalResult.run_id !== runId || evalResult.eval_split !== "test") {
    throw new Error("forecast_adjustment_decision test application decision requires matching raw test eval.json")
  }
  await readBoundModelArtifactIdentity(runRoot, String(spec.model ?? ""), evalResult)
  if (typeof evalResult.metrics.mse !== "number" || !Number.isFinite(evalResult.metrics.mse)) {
    throw new Error("forecast_adjustment_decision test application decision requires finite raw test metrics")
  }
  return { evalResult, spec }
}

async function inspectValidationResultForRawTestRun(
  validations: Array<z.infer<typeof AdjustmentEvalResult>>,
  rawTestSpec: Record<string, unknown>,
): Promise<{
  matchedValidation: z.infer<typeof AdjustmentEvalResult> | null
  closestDifferences: string[] | null
}> {
  let closestDifferences: string[] | null = null
  for (const validation of validations) {
    if (!runIdIsSafe(validation.run_id)) continue
    const validationSpec = z
      .record(z.string(), z.unknown())
      .parse(await readJsonFile(path.join(forecastRoot(), "runs", validation.run_id, "spec.json")))
    const differences = validationPostForecastTestSpecDifferences(validationSpec, rawTestSpec)
    if (differences?.length === 0) return { matchedValidation: validation, closestDifferences: [] }
    if (differences && (closestDifferences === null || differences.length < closestDifferences.length)) {
      closestDifferences = differences
    }
  }
  return { matchedValidation: null, closestDifferences }
}

async function requireValidationResultForRawTestRun(
  validations: Array<z.infer<typeof AdjustmentEvalResult>>,
  rawTestSpec: Record<string, unknown>,
): Promise<z.infer<typeof AdjustmentEvalResult>> {
  const inspected = await inspectValidationResultForRawTestRun(validations, rawTestSpec)
  if (inspected.matchedValidation) return inspected.matchedValidation
  const detail = inspected.closestDifferences?.length
    ? ` Closest validation contract differs in: ${inspected.closestDifferences.join(", ")}.`
    : ""
  throw new Error(
    "Locked adjustment skill was not validation-accepted on this test model spec; only matching models in the current test collection may be adjusted." +
      detail,
  )
}

function sameRunSet(left: string[], right: string[]): boolean {
  const sortedLeft = [...left].sort()
  const sortedRight = [...right].sort()
  return sortedLeft.length === sortedRight.length && sortedLeft.every((value, index) => value === sortedRight[index])
}

async function prepareTestApplications(testRunIds: string[]) {
  await assertFrozenTaskIntegrity(forecastRoot())
  const state = await readPhaseState()
  if (state.phase !== "post-forecast") {
    throw new Error("forecast_adjustment_decision prepare_test_applications is only allowed during post-forecast")
  }
  if (
    !state.adjustment.forecaster_locked ||
    !state.adjustment.active_skill_version ||
    state.adjustment.no_effective_skill
  ) {
    throw new Error("forecast_adjustment_decision prepare_test_applications requires a locked active adjustment skill")
  }

  const uniqueRunIds = [...new Set(testRunIds)]
  if (uniqueRunIds.length !== testRunIds.length) {
    throw new Error("forecast_adjustment_decision prepare_test_applications requires unique test_run_ids")
  }
  for (const runId of uniqueRunIds) {
    if (!runIdIsSafe(runId)) throw new Error(`Invalid test run id "${runId}"`)
  }

  const priorRecords = state.adjustment.test_application_records
  const legacyDecisionStarted =
    priorRecords.length === 0 &&
    (state.adjustment.test_application_decision !== "pending" || state.adjustment.test_adjustment_locked)
  if (legacyDecisionStarted && state.adjustment.test_collection_run_ids.length === 0) {
    throw new Error(
      "Cannot prepare a multi-model Critic test collection after a legacy single-run test adjustment decision has started",
    )
  }
  const hasStartedDecisions = priorRecords.some(
    (record) => record.decision !== "pending" || record.adjustment_locked || record.adjustment_evaluated,
  )
  if (
    state.adjustment.test_collection_run_ids.length > 0 &&
    !sameRunSet(state.adjustment.test_collection_run_ids, uniqueRunIds) &&
    hasStartedDecisions
  ) {
    throw new Error("Cannot replace the registered Critic test collection after test adjustment decisions have started")
  }

  const { manifest, validations } = await readActiveValidationResult()
  const eligible = []
  const ineligible = []

  for (const runId of uniqueRunIds) {
    const rawTest = await assertRawTestBestRunReady(runId)
    const inspectedValidation = await inspectValidationResultForRawTestRun(validations, rawTest.spec)
    const matchedValidation = inspectedValidation.matchedValidation
    const model = typeof rawTest.spec.model === "string" ? rawTest.spec.model : null
    if (!matchedValidation) {
      const detail = inspectedValidation.closestDifferences?.length
        ? ` Closest validation contract differs in: ${inspectedValidation.closestDifferences.join(", ")}.`
        : ""
      ineligible.push({
        test_run_id: runId,
        model,
        raw_metrics: rawTest.evalResult.metrics,
        reason: "No accepted validation result from the locked skill matches this test model spec." + detail,
      })
      continue
    }
    eligible.push({
      test_run_id: runId,
      validation_run_id: matchedValidation.run_id,
      model,
      raw_metrics: rawTest.evalResult.metrics,
      validation_summary: matchedValidation,
    })
  }

  const previousByRun = new Map(priorRecords.map((record) => [record.test_run_id, record]))
  const records = eligible.map((candidate) => {
    const previous = previousByRun.get(candidate.test_run_id)
    if (previous && previous.validation_run_id === candidate.validation_run_id) return previous
    return {
      test_run_id: candidate.test_run_id,
      validation_run_id: candidate.validation_run_id,
      model: candidate.model,
      decision: "pending" as const,
      decision_at: null,
      decision_actor: null,
      decision_reason: null,
      adjustment_locked: false,
      adjustment_candidate_id: null,
      adjustment_evaluated: false,
      adjusted_eval_path: null,
    }
  })
  const now = new Date().toISOString()
  const noEligibleReason =
    eligible.length === 0
      ? "No test model in the registered Critic collection matches the locked skill validation evidence."
      : null
  const pendingRecord = records.find((record) => record.decision === "pending")
  const latestDecidedRecord = [...records].reverse().find((record) => record.decision !== "pending")
  const aggregateDecision =
    eligible.length === 0
      ? ("skip" as const)
      : pendingRecord
        ? ("pending" as const)
        : (latestDecidedRecord?.decision ?? "pending")
  const latestApply = [...records].reverse().find((record) => record.decision === "apply")
  await writePhaseState(
    {
      ...state,
      adjustment: {
        ...state.adjustment,
        test_collection_run_ids: uniqueRunIds,
        test_eligible_run_ids: eligible.map((candidate) => candidate.test_run_id),
        test_application_records: records,
        critic_user_confirmed: latestApply?.decision_at ?? null,
        test_application_decision: aggregateDecision,
        test_application_decision_at:
          eligible.length === 0
            ? now
            : aggregateDecision === "pending"
              ? null
              : (latestDecidedRecord?.decision_at ?? null),
        test_application_decision_actor:
          aggregateDecision === "pending" ? null : (latestDecidedRecord?.decision_actor ?? null),
        test_application_decision_reason:
          eligible.length === 0
            ? noEligibleReason
            : aggregateDecision === "pending"
              ? null
              : (latestDecidedRecord?.decision_reason ?? null),
        test_application_run_id: aggregateDecision === "pending" ? null : (latestDecidedRecord?.test_run_id ?? null),
        test_adjustment_locked: records.some((record) => record.adjustment_locked),
        test_adjustment_run_id:
          records.find((record) => record.adjustment_locked)?.test_run_id ?? state.adjustment.test_adjustment_run_id,
        test_adjustment_candidate_id:
          records.find((record) => record.adjustment_locked)?.adjustment_candidate_id ??
          state.adjustment.test_adjustment_candidate_id,
      },
    },
    `## Adjustment Test Collection Prepared\n\nRegistered ${uniqueRunIds.length} successful test runs; ${eligible.length} match locked skill ${manifest.version}.`,
  )

  return {
    active_skill_version: manifest.version,
    candidate_id: manifest.candidate_id,
    test_collection_run_ids: uniqueRunIds,
    eligible,
    ineligible,
  }
}

async function prepareTestApplicationDecision(rawTestRunId: string | undefined) {
  await assertFrozenTaskIntegrity(forecastRoot())
  const state = await readPhaseState()
  if (state.phase !== "post-forecast") {
    throw new Error("forecast_adjustment_decision test application decisions are only allowed during post-forecast")
  }
  if (
    !state.adjustment.forecaster_locked ||
    !state.adjustment.active_skill_version ||
    state.adjustment.no_effective_skill
  ) {
    throw new Error("forecast_adjustment_decision test application decision requires a locked active adjustment skill")
  }
  if (!rawTestRunId) {
    throw new Error("forecast_adjustment_decision test application decision requires raw_test_run_id")
  }
  const existingRecord = state.adjustment.test_application_records.find((record) => record.test_run_id === rawTestRunId)
  if (state.adjustment.test_eligible_run_ids.length > 0 && !existingRecord) {
    throw new Error(
      "forecast_adjustment_decision test application decision requires a run from the prepared eligible test set",
    )
  }
  if (existingRecord?.adjustment_locked) {
    throw new Error(
      "forecast_adjustment_decision test application decision is too late after this run's values are locked",
    )
  }
  if (existingRecord && existingRecord.decision !== "pending") {
    throw new Error("forecast_adjustment_decision test application decision is only allowed before this run is decided")
  }
  if (
    !existingRecord &&
    state.adjustment.test_application_records.length === 0 &&
    state.adjustment.test_application_run_id === rawTestRunId &&
    state.adjustment.test_application_decision !== "pending"
  ) {
    throw new Error("forecast_adjustment_decision test application decision is only allowed before it is recorded")
  }

  const rawTest = await assertRawTestBestRunReady(rawTestRunId)
  const { manifest, validations } = await readActiveValidationResult()
  const matchedValidation = await requireValidationResultForRawTestRun(validations, rawTest.spec)
  if (existingRecord && existingRecord.validation_run_id !== matchedValidation.run_id) {
    throw new Error("Prepared test application record no longer matches the locked skill validation evidence")
  }
  return { state, rawTest, manifest, matchedValidation }
}

async function writeTestApplicationDecision(input: {
  state: Awaited<ReturnType<typeof readPhaseState>>
  apply: boolean
  actor: "user" | "llm_auto" | "auto_policy"
  reason: string
  now: string
  rawTestRunId: string
  manifest: z.infer<typeof AdjustmentSkillManifest>
  matchedValidation: z.infer<typeof AdjustmentEvalResult>
  model: string | null
}) {
  const existingRecords = input.state.adjustment.test_application_records
  const nextRecord = {
    test_run_id: input.rawTestRunId,
    validation_run_id: input.matchedValidation.run_id,
    model: input.model,
    decision: input.apply ? ("apply" as const) : ("skip" as const),
    decision_at: input.now,
    decision_actor: input.actor,
    decision_reason: input.reason,
    adjustment_locked: false,
    adjustment_candidate_id: null,
    adjustment_evaluated: false,
    adjusted_eval_path: null,
  }
  const recordIndex = existingRecords.findIndex((record) => record.test_run_id === input.rawTestRunId)
  const records =
    recordIndex >= 0
      ? existingRecords.map((record, index) => (index === recordIndex ? { ...record, ...nextRecord } : record))
      : [...existingRecords, nextRecord]
  const latestApply = [...records].reverse().find((record) => record.decision === "apply")
  await writePhaseState(
    {
      ...input.state,
      adjustment: {
        ...input.state.adjustment,
        test_application_records: records,
        test_eligible_run_ids: [...new Set([...input.state.adjustment.test_eligible_run_ids, input.rawTestRunId])],
        critic_user_confirmed: latestApply?.decision_at ?? null,
        test_application_decision: input.apply ? "apply" : "skip",
        test_application_decision_at: input.now,
        test_application_decision_actor: input.actor,
        test_application_decision_reason: input.reason,
        test_application_run_id: input.rawTestRunId,
        test_adjustment_locked: records.some((record) => record.adjustment_locked),
        test_adjustment_run_id:
          records.find((record) => record.adjustment_locked)?.test_run_id ??
          input.state.adjustment.test_adjustment_run_id,
        test_adjustment_candidate_id:
          records.find((record) => record.adjustment_locked)?.adjustment_candidate_id ??
          input.state.adjustment.test_adjustment_candidate_id,
      },
    },
    `## Adjustment Test Application Decision\n\n${
      input.actor === "user" ? "User" : input.actor === "auto_policy" ? "Auto Mode policy" : "Auto Mode LLM"
    } selected ${input.apply ? "apply" : "skip"} for locked skill ${input.manifest.version} at ${input.now}. Reason: ${
      input.reason
    }`,
  )

  return {
    decision: input.apply ? "apply" : "skip",
    decision_actor: input.actor,
    decision_reason: input.reason,
    raw_test_run_id: input.rawTestRunId,
    matched_validation_run_id: input.matchedValidation.run_id,
    active_skill_version: input.manifest.version,
    candidate_id: input.manifest.candidate_id,
    validation_summary: input.matchedValidation,
    adjustment: (await readPhaseState()).adjustment,
  }
}

async function writeStoppedAfterA(input: {
  state: Awaited<ReturnType<typeof readPhaseState>>
  actor: "user" | "llm_auto"
  reason: string
  now: string
}) {
  await assertNoRetainableValidationCandidate("forecast_adjustment_decision stop_adjustment")
  await assertUserPriorMirrorResolved("forecast_adjustment_decision stop_adjustment")
  await writePhaseState(
    {
      ...input.state,
      adjustment: {
        ...input.state.adjustment,
        enabled: true,
        mode: "rule_skill",
        forecaster_locked: true,
        no_effective_skill: true,
        no_effective_skill_reason: input.reason,
        active_skill_version: null,
        active_skill_path: null,
        test_collection_run_ids: [],
        test_eligible_run_ids: [],
        test_application_records: [],
        critic_user_confirmed: null,
        test_application_decision: "skip",
        test_application_decision_at: input.now,
        test_application_decision_actor: input.actor,
        test_application_decision_reason: input.reason,
        test_application_run_id: null,
        test_adjustment_locked: false,
        test_adjustment_run_id: null,
        test_adjustment_candidate_id: null,
      },
    },
    `## Adjustment Escalation Declined\n\n${input.actor === "user" ? "User" : "Auto Mode LLM"} stopped adjustment. ${input.reason}`,
  )
  return {
    decision: "stop_adjustment",
    decision_actor: input.actor,
    decision_reason: input.reason,
    adjustment: (await readPhaseState()).adjustment,
  }
}

export const ForecastAdjustmentDecisionTool = Tool.define("forecast_adjustment_decision", {
  description: DESCRIPTION,
  parameters,
  async execute(params, ctx) {
    await ensureAdjustmentDirs()
    const now = new Date().toISOString()
    let state = await readPhaseState()
    let result: unknown

    switch (params.subcommand) {
      case "read": {
        result = {
          adjustment: state.adjustment,
          policy: await readPolicy(),
        }
        break
      }

      case "ask_start": {
        assertForecastDecisionAllowed("ask_start", state)
        assertManualDecisionMode("ask_start", state)
        const answer = await askSingle({
          sessionID: ctx.sessionID,
          tool: ctx.callID ? { messageID: ctx.messageID, callID: ctx.callID } : undefined,
          questions: [
            {
              header: "生成式预测推理",
              question:
                "模型搜索与验证已经完成。是否启用生成式预测推理？系统将基于训练侧证据生成候选推理方案，并通过验证集决定是否采纳。",
              custom: false,
              options: [
                {
                  label: START_OPTION_DISABLED,
                  description: "保留原始预测结果，直接进入最终测试评估。",
                },
                {
                  label: START_OPTION_A,
                  description: "采用精简的候选迭代与推理规则，优先处理明确、稳定的系统性偏差。",
                },
                {
                  label: START_OPTION_B,
                  description: "综合多模型与训练侧诊断，进行更充分的候选迭代与规则分析。",
                },
              ],
            },
          ],
        })
        assertOneOf(answer, [START_OPTION_DISABLED, START_OPTION_A, START_OPTION_B], "Adjustment start decision")
        const choice = answer === START_OPTION_A ? "enabled_a" : answer === START_OPTION_B ? "enabled_b" : "disabled"
        result = await writeStartDecision(choice, now, `User selected ${answer}`, {
          actor: "user",
          reason: `User selected ${answer} in the generative prediction reasoning start decision window.`,
        })
        break
      }

      case "auto_start": {
        assertForecastDecisionAllowed("auto_start", state)
        assertAutoDecisionMode("auto_start", state)
        if (!params.auto_start_choice) {
          throw new Error("forecast_adjustment_decision auto_start requires auto_start_choice")
        }
        const reason = requireAutoReason(params.reason, "auto_start")
        result = await writeStartDecision(
          params.auto_start_choice,
          now,
          `Auto Mode LLM selected ${params.auto_start_choice}`,
          {
            actor: "llm_auto",
            reason,
          },
        )
        break
      }

      case "ask_escalation": {
        assertForecastDecisionAllowed("ask_escalation", state)
        assertManualDecisionMode("ask_escalation", state)
        if (state.adjustment.user_decision !== "enabled_a" || state.adjustment.strategy !== "a_lightweight") {
          throw new Error(
            "forecast_adjustment_decision ask_escalation requires an active lightweight prediction reasoning decision",
          )
        }
        const answer = await askSingle({
          sessionID: ctx.sessionID,
          tool: ctx.callID ? { messageID: ctx.messageID, callID: ctx.callID } : undefined,
          questions: [
            {
              header: "生成式预测推理",
              question: [
                "轻量预测推理尚未产生通过验证集验收的有效生成式预测推理方案。",
                params.reason ? `原因：${params.reason}` : "",
                "是否切换至深度预测推理继续分析？",
              ]
                .filter(Boolean)
                .join("\n"),
              custom: false,
              options: [
                {
                  label: ESCALATE_OPTION_B,
                  description: "进一步综合训练侧诊断与多模型证据，继续生成并验证候选方案。",
                },
                {
                  label: ESCALATE_OPTION_STOP,
                  description: "结束生成式预测推理流程，保留原始预测结果进入最终测试。",
                },
              ],
            },
          ],
        })

        assertOneOf(answer, [ESCALATE_OPTION_B, ESCALATE_OPTION_STOP], "Adjustment escalation decision")
        if (answer === ESCALATE_OPTION_B) {
          result = await writeStartDecision(
            "enabled_b",
            now,
            "User switched from lightweight prediction reasoning to deep prediction reasoning",
            {
              actor: "user",
              reason: params.reason
                ? `User switched to deep prediction reasoning after lightweight validation failure: ${params.reason}`
                : "User switched to deep prediction reasoning after lightweight prediction reasoning did not pass validation.",
              resetAdjustmentArtifacts: false,
            },
          )
          break
        }

        state = await readPhaseState()
        result = await writeStoppedAfterA({
          state,
          actor: "user",
          reason:
            params.reason ??
            "User ended generative prediction reasoning after lightweight prediction reasoning did not pass validation.",
          now,
        })
        break
      }

      case "auto_escalation": {
        assertForecastDecisionAllowed("auto_escalation", state)
        assertAutoDecisionMode("auto_escalation", state)
        if (state.adjustment.user_decision !== "enabled_a" || state.adjustment.strategy !== "a_lightweight") {
          throw new Error(
            "forecast_adjustment_decision auto_escalation requires an active lightweight prediction reasoning decision",
          )
        }
        if (!params.auto_escalation_choice) {
          throw new Error("forecast_adjustment_decision auto_escalation requires auto_escalation_choice")
        }
        const reason = requireAutoReason(params.reason, "auto_escalation")
        if (params.auto_escalation_choice === "upgrade_b") {
          result = await writeStartDecision(
            "enabled_b",
            now,
            "Auto Mode LLM switched from lightweight prediction reasoning to deep prediction reasoning",
            {
              actor: "llm_auto",
              reason,
              resetAdjustmentArtifacts: false,
            },
          )
          break
        }
        result = await writeStoppedAfterA({ state, actor: "llm_auto", reason, now })
        break
      }

      case "prepare_test_applications": {
        if (!params.test_run_ids) {
          throw new Error("forecast_adjustment_decision prepare_test_applications requires test_run_ids")
        }
        result = await prepareTestApplications(params.test_run_ids)
        break
      }

      case "ask_test_application": {
        state = await readPhaseState()
        assertManualDecisionMode("ask_test_application", state)
        if (state.phase !== "post-forecast") {
          throw new Error(
            "forecast_adjustment_decision test application decisions are only allowed during post-forecast",
          )
        }
        const runIds = params.raw_test_run_id
          ? [params.raw_test_run_id]
          : state.adjustment.test_application_records
              .filter((record) => record.decision === "pending")
              .map((record) => record.test_run_id)
        if (runIds.length === 0) {
          throw new Error(
            "forecast_adjustment_decision ask_test_application found no pending eligible test runs; call prepare_test_applications with the current Critic test collection first",
          )
        }
        const decisions = []
        for (const runId of runIds) {
          const {
            state: preparedState,
            rawTest,
            manifest,
            matchedValidation,
          } = await prepareTestApplicationDecision(runId)
          const model = typeof rawTest.spec.model === "string" ? rawTest.spec.model : null
          const answer = await askSingle({
            sessionID: ctx.sessionID,
            tool: ctx.callID ? { messageID: ctx.messageID, callID: ctx.callID } : undefined,
            questions: [
              {
                header: "测试阶段生成式预测推理",
                question: [
                  `测试模型 ${model ?? "unknown"}（run: ${runId}）具有严格匹配且已通过验收的验证集生成式预测推理证据：`,
                  formatValidationSummary(matchedValidation),
                  `Validated source run: ${matchedValidation.run_id}.`,
                  `Raw test metrics: MSE ${formatMetric(rawTest.evalResult.metrics.mse)}, MAE ${formatMetric(rawTest.evalResult.metrics.mae)}.`,
                  "是否在该测试模型上应用已锁定的生成式预测推理方案？该决定仅影响当前模型；无论它是否为原始测试最优模型，系统都只使用与该模型严格匹配的验证证据。",
                ].join("\n"),
                custom: false,
                options: [
                  {
                    label: TEST_OPTION_APPLY,
                    description: "应用已锁定方案，并同时保留原始指标与生成式预测推理指标。",
                  },
                  {
                    label: TEST_OPTION_SKIP,
                    description: "不应用生成式预测推理，仅保留该模型的原始测试指标。",
                  },
                ],
              },
            ],
          })

          assertOneOf(answer, [TEST_OPTION_APPLY, TEST_OPTION_SKIP], "Adjustment test application decision")
          decisions.push(
            await writeTestApplicationDecision({
              state: preparedState,
              apply: answer === TEST_OPTION_APPLY,
              actor: "user",
              reason: `User selected ${answer} for eligible test run ${runId} in the test-stage adjustment decision window.`,
              now: new Date().toISOString(),
              rawTestRunId: runId,
              manifest,
              matchedValidation,
              model,
            }),
          )
        }
        result = params.raw_test_run_id ? decisions[0] : { decisions }
        break
      }

      case "auto_test_application": {
        state = await readPhaseState()
        assertAutoDecisionMode("auto_test_application", state)
        if (state.phase !== "post-forecast") {
          throw new Error(
            "forecast_adjustment_decision test application decisions are only allowed during post-forecast",
          )
        }
        if (params.auto_test_application_choice === "skip") {
          throw new Error(
            "Auto Mode defaults to applying the locked skill to every eligible test model; skip is only available through the Manual mode user decision",
          )
        }
        const runIds = params.raw_test_run_id
          ? [params.raw_test_run_id]
          : state.adjustment.test_application_records
              .filter((record) => record.decision === "pending")
              .map((record) => record.test_run_id)
        if (runIds.length === 0) {
          throw new Error(
            "forecast_adjustment_decision auto_test_application found no pending eligible test runs; call prepare_test_applications with the current Critic test collection first",
          )
        }
        const decisions = []
        for (const runId of runIds) {
          const {
            state: preparedState,
            rawTest,
            manifest,
            matchedValidation,
          } = await prepareTestApplicationDecision(runId)
          const model = typeof rawTest.spec.model === "string" ? rawTest.spec.model : null
          decisions.push(
            await writeTestApplicationDecision({
              state: preparedState,
              apply: true,
              actor: "auto_policy",
              reason: `Auto Mode policy defaulted to apply because test run ${runId} has an exact accepted validation counterpart for locked skill ${manifest.version}.`,
              now: new Date().toISOString(),
              rawTestRunId: runId,
              manifest,
              matchedValidation,
              model,
            }),
          )
        }
        result = params.raw_test_run_id ? decisions[0] : { decisions }
        break
      }
    }

    return {
      title: `Forecast adjustment decision: ${params.subcommand}`,
      metadata: { subcommand: params.subcommand },
      output: JSON.stringify(result),
    }
  },
})
