import { afterEach, describe, expect, test } from "bun:test"
import { mkdir, readFile, symlink, writeFile } from "fs/promises"
import path from "path"
import { Instance } from "@/project/instance"
import { MessageID, SessionID } from "@/session/schema"
import { tmpdir } from "../../../test/fixture/fixture"
import { readPhaseState, resolveForecastPythonTimeoutMs, writePhaseState } from "./forecast-adjustment-common"
import {
  validateWindowReflectionPolicyArtifact,
  validateWindowReflectionRecordArtifact,
} from "./forecast-window-reflection-artifact"
import {
  buildProviderContentRiskCompactContext,
  buildProviderContentRiskSafeContext,
  buildContextualCriticPredictionState,
  buildEvidenceGapAssessmentInput,
  bindCriticStateToBlindContext,
  classifyWindowReflectionLlmError,
  createConcurrencyLimiter,
  createWindowLlmAuditWriter,
  deterministicEvidenceTypeFromNeed,
  EVIDENCE_GAP_SYSTEM,
  EVIDENCE_GAP_MAX_RETRIES,
  evidenceGapStructuredOutputFormat,
  EVIDENCE_SEEKING_SYSTEM,
  finalizeEvidenceSeekingResponse,
  UNIFIED_REASONING_SYSTEM,
  ForecastWindowReflectionTool,
  isWindowReflectionContentRisk,
  isReusableEvidenceLoopCorrection,
  isReusableWindowCorrection,
  normalizeProposalLayout,
  parseWindowReflectionText,
  parseEvidenceGapText,
  parseEvidenceGapStructuredOutput,
  parseEvidenceSeekingText,
  providerStructuredResponseTexts,
  recoverEvidenceGapSubmission,
  selectEvidenceTypeByMode,
  buildEvidenceSeekingInput,
  buildUnifiedEvidenceReasoningInput,
  unusedEvidenceCandidates,
  resetReflectionRunDirectory,
  requireCompleteLookbackTargetWindow,
  retryEvidenceGapAssessment,
  restoreLegacyPreSafetyCheckpoint,
  resolveWindowReflectionTestApiConcurrency,
  shouldReopenAbnormalEvidenceLoopCheckpoint,
  SKILL_GUIDED_TEST_SYSTEM,
  withWindowReflectionTimeout,
  WINDOW_REFLECTION_PROVIDER_RETRY_LIMIT,
  WINDOW_REFLECTION_PREDICTION_OUTPUT_TOKEN_LIMIT,
  WINDOW_REFLECTION_REASONING_TOKEN_GUIDANCE,
  WINDOW_REFLECTION_TEST_API_CONCURRENCY,
  windowReflectionPredictionOutputTokenLimit,
  windowOutcomeRequiresRerun,
  windowReflectionProviderRetryDelayMs,
} from "./forecast-window-reflection"
import { computeFileSha256 } from "./model-artifact"
import { ForecastPhaseState, ForecastReflectionValidationRecord, ForecastWindowReflectionPolicy } from "./types"

const ctx = {
  sessionID: SessionID.make("ses_window-reflection-test"),
  messageID: MessageID.make("msg_window-reflection-test"),
  callID: "window-reflection-test-call",
  agent: "window-reflection-test-agent",
  abort: AbortSignal.any([]),
  messages: [],
  metadata: () => {},
  ask: async () => {},
}

const baseTask = {
  dataset_path: "dataset.csv",
  target_col: "value",
  time_col: "timestamp",
  pred_len: 24,
  forecast_gap: 0,
  seq_len: 48,
  label_len: 24,
  window_stride: 1,
  train_ratio: 0.7,
  val_ratio: 0.2,
  test_ratio: 0.1,
  adjustment_fit_holdout_ratio: 0.5,
  eval_metrics: ["mse", "mae"],
  model_families: ["DLinear"],
  reflection_enabled: true,
  frozen_at: "2026-07-15T00:00:00.000Z",
}

async function writeTask(directory: string, overrides: Record<string, unknown> = {}) {
  const taskPath = path.join(directory, ".forecast", "task.json")
  await mkdir(path.dirname(taskPath), { recursive: true })
  await writeFile(taskPath, `${JSON.stringify({ ...baseTask, ...overrides })}\n`, "utf-8")
}

async function writeState(phase: "pre-forecast" | "skills-review" | "forecast" | "post-forecast") {
  await writePhaseState(
    ForecastPhaseState.parse({
      phase,
      skills_confirmed: false,
      skills_confirmed_at: null,
      pre_forecast_completed: false,
      pre_forecast_completed_at: null,
      task_frozen_at: "2026-07-15T00:00:00.000Z",
      transitions: [],
    }),
  )
}

afterEach(async () => {
  await Instance.disposeAll()
})

describe("forecast_window_reflection", () => {
  test("persists every visible LLM interaction and continues audit numbering after resume", async () => {
    await using tmp = await tmpdir()
    const correctionsDir = path.join(tmp.path, "corrections")
    const firstWriter = await createWindowLlmAuditWriter(correctionsDir, 7)
    await firstWriter({
      call_type: "evidence_seeking",
      system_prompt: "seek-system",
      request: { needed_evidence: "current context" },
      status: "success",
      child_session_id: "session-1",
      raw_response: '{"selected_evidence_candidate_ids":["context:current-window"]}',
      parsed_response: { selected_evidence_candidate_ids: ["context:current-window"] },
      error: null,
      provider_response_parts: [
        { type: "reasoning", text: "The current-window candidate directly addresses the missing context." },
        { type: "text", text: '{"selected_evidence_candidate_ids":["context:current-window"]}' },
      ],
      rationale: "The current-window candidate directly addresses the missing context.",
    })

    const resumedWriter = await createWindowLlmAuditWriter(correctionsDir, 7)
    await resumedWriter({
      call_type: "unified_reasoning",
      system_prompt: "reason-system",
      request: { state_sha256: "a".repeat(64) },
      status: "error",
      child_session_id: "session-2",
      raw_response: "invalid response",
      parsed_response: null,
      error: "format failure",
    })

    const auditDir = path.join(correctionsDir, "llm-calls", "window-000007")
    const first = JSON.parse(await readFile(path.join(auditDir, "call-000001.json"), "utf-8"))
    const firstReasoning = JSON.parse(
      await readFile(path.join(auditDir, "call-000001-reasoning-000001.json"), "utf-8"),
    )
    const second = JSON.parse(await readFile(path.join(auditDir, "call-000002.json"), "utf-8"))
    expect(first).toMatchObject({ window_index: 7, sequence: 1, status: "success", child_session_id: "session-1" })
    expect(first.raw_response).toContain("context:current-window")
    expect(first.provider_response_parts).toHaveLength(2)
    expect(first.provider_reasoning_parts).toEqual([
      { type: "reasoning", text: "The current-window candidate directly addresses the missing context." },
    ])
    expect(first.rationale).toBe("The current-window candidate directly addresses the missing context.")
    expect(firstReasoning).toMatchObject({
      window_index: 7,
      call_sequence: 1,
      reasoning_part_index: 1,
      provider_response_part: {
        type: "reasoning",
        text: "The current-window candidate directly addresses the missing context.",
      },
    })
    expect(second).toMatchObject({ window_index: 7, sequence: 2, status: "error", error: "format failure" })
    expect(second.reasoning_visibility).toContain("hidden chain-of-thought is not available")
  })

  test("isolates adaptive seeking and feeds provisional evidence fusion to one prediction reasoner", () => {
    expect(EVIDENCE_GAP_SYSTEM).toContain("Follow this guided assessment")
    expect(EVIDENCE_GAP_SYSTEM).toContain("evidence_coverage")
    expect(EVIDENCE_GAP_SYSTEM).toContain("missing_evidence")
    expect(EVIDENCE_GAP_SYSTEM).toContain("evidence_consistency")
    expect(EVIDENCE_GAP_SYSTEM).toContain("evidence_conflicts")
    expect(EVIDENCE_GAP_SYSTEM).toContain("evidence_confidence")
    expect(EVIDENCE_GAP_SYSTEM).toContain("never treat a summary as evidence")
    expect(EVIDENCE_GAP_SYSTEM).toContain("not the default accepted answer")
    expect(EVIDENCE_GAP_SYSTEM).toContain("initial-prior check")
    expect(EVIDENCE_GAP_SYSTEM).toContain("Plausibility alone is insufficient")
    expect(EVIDENCE_GAP_SYSTEM).toContain("Evidence search does not require changing the prediction")
    expect(EVIDENCE_GAP_SYSTEM).toContain("existence of unused candidates")
    expect(EVIDENCE_GAP_SYSTEM).toContain("current-window context as optional")
    expect(EVIDENCE_GAP_SYSTEM).toContain("availability of exogenous variables does not itself create an evidence gap")
    expect(EVIDENCE_GAP_SYSTEM).toContain("association is known to be weak, unstable, indirect")
    expect(EVIDENCE_GAP_SYSTEM).toContain("never assume that selecting context will be informative")
    expect(EVIDENCE_GAP_SYSTEM).toContain("whenever needed_evidence names context or an exogenous variable")
    expect(EVIDENCE_GAP_SYSTEM).toContain("do not need to prove that the prior is defective")
    expect(EVIDENCE_GAP_SYSTEM).toContain("required StructuredOutput tool")
    expect(EVIDENCE_GAP_SYSTEM).toContain("do not update Critic state")
    expect(EVIDENCE_SEEKING_SYSTEM).toContain("deterministic matching found no match")
    expect(EVIDENCE_SEEKING_SYSTEM).toContain("available_candidate_types")
    expect(UNIFIED_REASONING_SYSTEM).toContain("UnifiedEvidenceReasoner")
    expect(UNIFIED_REASONING_SYSTEM).toContain("provisional fused evidence set")
    expect(UNIFIED_REASONING_SYSTEM).toContain("support, overlap, scope, dependencies, and conflicts")
    expect(UNIFIED_REASONING_SYSTEM).toContain("complete lookback_window_target")
    expect(UNIFIED_REASONING_SYSTEM).toContain("immutable initial prediction")
    expect(UNIFIED_REASONING_SYSTEM).toContain("Never mechanically add")
    expect(UNIFIED_REASONING_SYSTEM).toContain("Never mutate, return, or claim to commit Critic state")
    expect(UNIFIED_REASONING_SYSTEM).toContain("Never use observed targets")
    expect(UNIFIED_REASONING_SYSTEM).toContain('keys "window_index", "points", "rationale", and "evidence_ids"')
    expect(UNIFIED_REASONING_SYSTEM).toContain("under approximately 20,000 tokens")
    expect(UNIFIED_REASONING_SYSTEM).toContain("column-oriented arrays")
    expect(WINDOW_REFLECTION_REASONING_TOKEN_GUIDANCE).toBe(20_000)
    expect(WINDOW_REFLECTION_PREDICTION_OUTPUT_TOKEN_LIMIT).toBe(65_536)
    expect(windowReflectionPredictionOutputTokenLimit()).toBe(32_000)
    expect(SKILL_GUIDED_TEST_SYSTEM).toContain("critic_prediction_state.predictions as the only current prediction")
  })

  test("submits evidence-gap assessments through a required non-mutating structured-output tool", () => {
    const format = evidenceGapStructuredOutputFormat()
    expect(format.type).toBe("json_schema")
    if (format.type !== "json_schema") throw new Error("Expected JSON-schema output format")
    expect(format.retryCount).toBe(0)
    expect(format.schema).toMatchObject({
      type: "object",
      additionalProperties: false,
      required: [
        "evidence_gap",
        "needed_evidence",
        "evidence_coverage",
        "missing_evidence",
        "evidence_consistency",
        "evidence_conflicts",
        "evidence_confidence",
        "rationale",
      ],
    })

    expect(
      parseEvidenceGapStructuredOutput({
        evidence_gap: 1,
        needed_evidence: "Commit current-window context.",
        evidence_coverage: 0.5,
        missing_evidence: ["Current-window context is not committed."],
        evidence_consistency: 1,
        evidence_conflicts: [],
        evidence_confidence: 0.4,
        rationale: "Coverage is incomplete, while committed evidence is internally consistent.",
      }),
    ).toMatchObject({ evidence_gap: 1, evidence_coverage: 0.5 })
    expect(() =>
      parseEvidenceGapStructuredOutput({
        evidence_gap: 1,
        needed_evidence: "Commit current-window context.",
      }),
    ).toThrow()
  })

  test("recovers a schema-valid evidence-gap submission from exact provider text", () => {
    const providerText = JSON.stringify({
      evidence_gap: 1,
      needed_evidence: "Commit current-window context.",
      evidence_coverage: 0,
      missing_evidence: ["No committed evidence supports the forecast."],
      evidence_consistency: 0.5,
      evidence_conflicts: [],
      evidence_confidence: 0.2,
      rationale: "The prediction has no committed support.",
      trace: "Guided evidence-gap assessment completed.",
    })

    expect(recoverEvidenceGapSubmission(undefined, providerText)).toEqual({
      source: "text",
      text: providerText,
      value: {
        evidence_gap: 1,
        needed_evidence: "Commit current-window context.",
        evidence_coverage: 0,
        missing_evidence: ["No committed evidence supports the forecast."],
        evidence_consistency: 0.5,
        evidence_conflicts: [],
        evidence_confidence: 0.2,
        rationale: "The prediction has no committed support.",
      },
    })

    const native = {
      evidence_gap: 0 as const,
      needed_evidence: "",
      evidence_coverage: 1,
      missing_evidence: [],
      evidence_consistency: 1,
      evidence_conflicts: [],
      evidence_confidence: 0.9,
      rationale: "Native structured output wins.",
    }
    expect(recoverEvidenceGapSubmission(native, providerText)).toMatchObject({ source: "tool", value: native })

    expect(() => recoverEvidenceGapSubmission(undefined, `analysis\n${providerText}`)).toThrow()
    expect(() => recoverEvidenceGapSubmission(undefined, `\`\`\`json\n${providerText}\n\`\`\``)).toThrow()
    expect(() =>
      recoverEvidenceGapSubmission(
        undefined,
        JSON.stringify({ ...JSON.parse(providerText), predictions: [] }),
      ),
    ).toThrow()
  })

  test("retries the whole evidence-gap assessor at most twice", async () => {
    let attempts = 0
    await expect(
      retryEvidenceGapAssessment(async () => {
        attempts++
        if (attempts < 3) throw new Error("temporary gap-assessor failure")
        return "recovered"
      }),
    ).resolves.toBe("recovered")
    expect(attempts).toBe(EVIDENCE_GAP_MAX_RETRIES + 1)

    attempts = 0
    await expect(
      retryEvidenceGapAssessment(async () => {
        attempts++
        throw new Error("persistent gap-assessor failure")
      }),
    ).rejects.toThrow("persistent gap-assessor failure")
    expect(attempts).toBe(EVIDENCE_GAP_MAX_RETRIES + 1)
  })

  test("adapts and fuses committed context with newly selected experience before unified reasoning", () => {
    const initialPoint = { horizon_index: 0, timestamp: "2026-01-01 00:00:00", value: 100 }
    const initialState = {
      version: 1 as const,
      window_index: 7,
      evidences: [],
      predictions: [initialPoint],
      evidence_gap: 1 as const,
      needed_evidence: "experience support",
    }
    const state = {
      ...initialState,
      evidences: [
        {
          evidence_candidate_id: "context:current-window",
          candidate_type: "context" as const,
          artifact_path: "/tmp/context.json",
          artifact_sha256: "a".repeat(64),
        },
      ],
      predictions: [{ ...initialPoint, value: 104 }],
    }
    const experience = {
      candidate_type: "experience",
      evidence_candidate_id: "experience:locked",
      payload: { important_guidance_rules: [{ id: "rule-1", action: { type: "add", value: 10 } }] },
    }
    const context = {
      candidate_type: "context",
      evidence_candidate_id: "context:current-window",
      payload: {
        event: "committed",
        lookback_window_exogenous: {
          columns: ["timestamp", "temperature"],
          rows: [["2025-12-31 23:00:00", 8]],
          row_count: 1,
        },
        predict_window_exogenous: {
          columns: ["timestamp", "temperature"],
          rows: [["2026-01-01 00:00:00", 9]],
          row_count: 1,
        },
      },
    }
    const knowledge = {
      candidate_type: "knowledge",
      evidence_candidate_id: "knowledge:frozen-task",
      payload: { should_not_be_visible: true },
    }
    const ruleReference = {
      version: 1 as const,
      window_index: 7,
      semantics: "locked_rules_applied_to_initial_prediction_only" as const,
      experience_candidate_id: "experience:locked",
      experience_candidate_artifact: { path: "/tmp/experience.json", sha256: "b".repeat(64) },
      initial_predictions: [initialPoint],
      predictions: [{ ...initialPoint, value: 110 }],
      reasoning_contract: {
        reference_only: true as const,
        not_a_state_transition: true as const,
        do_not_add_reference_delta_to_current: true as const,
      },
    }

    const committedStateBeforeFusion = structuredClone(state)
    const input = buildUnifiedEvidenceReasoningInput({
      state,
      initialState,
      selectedCandidate: experience,
      allCandidates: { experience, context, knowledge },
      lookbackTargetWindow: { columns: ["timestamp", "value"], rows: [["2025-12-31 23:00:00", 98]] },
      experienceRuleReference: ruleReference,
    })

    expect(input.critic_prediction_state).toEqual(state)
    expect(input.initial_prediction_state).toEqual(initialState)
    expect(input.lookback_window_target).toEqual({
      columns: ["timestamp", "value"],
      rows: [["2025-12-31 23:00:00", 98]],
    })
    expect(input.provisional_evidence_fusion).toMatchObject({
      status: "provisional",
      base_committed_evidence_ids: ["context:current-window"],
      selected_evidence_candidate_id: "experience:locked",
      fused_evidence_ids: ["context:current-window", "experience:locked"],
      fusion_contract: { no_state_mutation_before_proposal_acceptance: true },
    })
    expect(input.provisional_evidence_fusion.evidence_attachments).toEqual([
      {
        evidence_candidate_id: "context:current-window",
        candidate_type: "context",
        adapter: "context_adapter",
        content: context,
        auxiliary: {
          lookback_window_exogenous: context.payload.lookback_window_exogenous,
          predict_window_exogenous: context.payload.predict_window_exogenous,
        },
      },
      {
        evidence_candidate_id: "experience:locked",
        candidate_type: "experience",
        adapter: "experience_adapter",
        content: experience,
        auxiliary: { rule_reference_prediction: ruleReference },
      },
    ])
    expect(JSON.stringify(input)).not.toContain("should_not_be_visible")
    expect(state).toEqual(committedStateBeforeFusion)
    expect(input.reasoning_contract).toMatchObject({
      current_state_is_transition_baseline: true,
      estimate_only_residual_effect_not_already_committed: true,
      unchanged_current_prediction_is_valid: true,
      jointly_reason_over_fused_evidence: true,
    })

    expect(() =>
      buildUnifiedEvidenceReasoningInput({
        state,
        initialState,
        selectedCandidate: experience,
        allCandidates: { experience, context, knowledge },
        lookbackTargetWindow: { columns: ["timestamp", "value"], rows: [["2025-12-31 23:00:00", 98]] },
        experienceRuleReference: { ...ruleReference, initial_predictions: [{ ...initialPoint, value: 99 }] },
      }),
    ).toThrow("not bound to the initial prediction")
    expect(() =>
      buildUnifiedEvidenceReasoningInput({
        state,
        initialState,
        selectedCandidate: context,
        allCandidates: { experience, context, knowledge },
        lookbackTargetWindow: { columns: ["timestamp", "value"], rows: [["2025-12-31 23:00:00", 98]] },
        experienceRuleReference: ruleReference,
      }),
    ).toThrow("already committed")
    expect(() =>
      buildUnifiedEvidenceReasoningInput({
        state: initialState,
        initialState,
        selectedCandidate: {
          candidate_type: "context",
          evidence_candidate_id: "context:summary-only",
          payload: { current_sample_exogenous: { summary: "insufficient" } },
        },
        allCandidates: {},
        lookbackTargetWindow: { columns: ["timestamp", "value"], rows: [["2025-12-31 23:00:00", 98]] },
        experienceRuleReference: null,
      }),
    ).toThrow("missing the complete lookback-window exogenous table")
  })

  test("requires an auditable validation-front calibration contract for controller v2", () => {
    const policy = {
      controller_version: "2",
      enabled: true,
      max_reflection_rounds: 3,
      max_format_retries: 2,
      trigger_policy: { hard_violation_count: 1, independent_soft_violation_count: 2 },
      constraints: {
        nonnegative: true,
        absolute_min: 0,
        absolute_max: 100,
        soft_absolute_min: 10,
        soft_absolute_max: 90,
        soft_abs_step: 10,
        max_abs_step: 20,
        soft_abs_curvature: 12,
        max_abs_curvature: 24,
        hard_min_window_range: 2,
        min_window_range: 5,
        window_mean_abs_delta_limit: 10,
        point_emergency_delta_limit: 30,
      },
      calibration: {
        method: "training_baseline_validation_front_calibration_v2",
        source_split: "validation_front",
        training_row_count: 100,
        validation_front_row_count: 30,
        validation_front_window_count: 4,
        validation_back_used: false,
        test_used: false,
        validation_front_window_statistics: {
          window_count: 4,
          range: { min: 1, q10: 2, q50: 3, q90: 4, q99: 5, max: 6 },
          max_abs_step: { min: 1, q10: 2, q50: 3, q90: 4, q99: 5, max: 6 },
          max_abs_curvature: { min: 1, q10: 2, q50: 3, q90: 4, q99: 5, max: 6 },
        },
        validation_front_reference_replay: {
          max_reference_trigger_rate: 0.1,
          initial_trigger_count: 1,
          initial_trigger_rate: 0.25,
          final_trigger_count: 0,
          final_trigger_rate: 0,
          hard_violation_window_count: 0,
          soft_threshold_adjustment_rounds: 1,
        },
        calibrated_rules: ["absolute_range", "adjacent_step"],
      },
      validation_gate: {
        metric: "mse",
        accept_if: "final_mse <= baseline_mse",
        zero_trigger: "enabled_no_trigger",
      },
    }

    expect(ForecastWindowReflectionPolicy.parse(policy).calibration?.source_split).toBe("validation_front")
    expect(() => ForecastWindowReflectionPolicy.parse({ ...policy, calibration: undefined })).toThrow(
      "controller v2 requires validation-front calibration",
    )
  })

  test("clears the stale candidate result without deleting sibling run artifacts", async () => {
    await using tmp = await tmpdir()
    const runRoot = path.join(tmp.path, ".forecast", "reflection", "test", "run-test")
    const outDir = path.join(runRoot, "raw")
    const correctionsDir = path.join(outDir, "corrections")
    const siblingDir = path.join(runRoot, "candidate-safe")
    await mkdir(correctionsDir, { recursive: true })
    await mkdir(siblingDir, { recursive: true })
    await writeFile(path.join(correctionsDir, "window-000000.json"), '{"accepted":true}\n', "utf-8")
    await writeFile(path.join(outDir, "prepared.json"), '{"stale":true}\n', "utf-8")
    await writeFile(path.join(outDir, "reflection_eval.json"), '{"stale":true}\n', "utf-8")
    await writeFile(path.join(outDir, "final_predictions.csv"), "stale\n", "utf-8")
    await writeFile(path.join(siblingDir, "keep.json"), '{"keep":true}\n', "utf-8")

    const recreatedCorrections = await resetReflectionRunDirectory(runRoot, outDir)

    await expect(readFile(path.join(correctionsDir, "window-000000.json"), "utf-8")).rejects.toThrow()
    await expect(readFile(path.join(outDir, "prepared.json"), "utf-8")).rejects.toThrow()
    await expect(readFile(path.join(outDir, "reflection_eval.json"), "utf-8")).rejects.toThrow()
    await expect(readFile(path.join(outDir, "final_predictions.csv"), "utf-8")).rejects.toThrow()
    expect(recreatedCorrections).toBe(correctionsDir)
    expect(await readFile(path.join(siblingDir, "keep.json"), "utf-8")).toContain('"keep":true')
  })

  test("classifies provider content rejection as a bounded provider retry instead of a format retry", () => {
    expect(isWindowReflectionContentRisk("APIError: Content Exists Risk")).toBe(true)
    expect(isWindowReflectionContentRisk("temporary upstream failure")).toBe(false)
    expect(
      classifyWindowReflectionLlmError({
        name: "APIError",
        data: {
          message: "Content Exists Risk",
          statusCode: 400,
          isRetryable: false,
        },
      }),
    ).toEqual({
      category: "provider",
      retryable: true,
      message: "APIError: Content Exists Risk",
      failureKind: "provider_failed",
    })

    expect(
      classifyWindowReflectionLlmError({
        name: "APIError",
        data: { message: "temporary upstream failure", isRetryable: true },
      }).retryable,
    ).toBe(true)

    expect(
      classifyWindowReflectionLlmError({
        name: "StructuredOutputError",
        data: { message: "response did not match the schema" },
      }),
    ).toEqual({
      category: "format",
      retryable: true,
      message: "StructuredOutputError: response did not match the schema",
      failureKind: "schema_failed",
    })
    expect(
      classifyWindowReflectionLlmError({
        name: "OutputLengthError",
        data: { message: "response reached its output limit" },
      }).failureKind,
    ).toBe("output_truncated")
  })

  test("projects only provider-risk text while preserving sample forecasting evidence", () => {
    const original = {
      task_and_scenario: {
        business_context: "free-form regional context",
        csv_semantics: "free-form schema prose",
        region_info: "free-form region prose",
        target_column: "OT",
        time_column: "date",
        history_window_steps: 711,
        forecast_gap_steps: 57,
        prediction_window_steps: 96,
        window_stride_steps: 96,
      },
      prediction_points: [
        {
          horizon_index: 29,
          timestamp: "2025/11/16 7:15",
          raw_anchor: 54295.1,
          pre_reflection_baseline: 53995.1,
        },
      ],
      visible_history_target_table: { columns: ["timestamp", "OT"], rows: [["2025/11/15 9:30", 57728.95]] },
      recent_three_day_exogenous_table: {
        columns: ["timestamp", "\u5317\u4eac\u5e02_t2m", "\u5929\u6c14", "\u653f\u7b56"],
        rows: [["2025/11/16 7:15", 6.3, "weather:clear", "event_phase:special_event_present"]],
      },
      current_sample_exogenous_context: {
        exogenous_column_count: 3,
        all_exogenous_columns: ["\u5317\u4eac\u5e02_t2m", "\u5929\u6c14", "\u653f\u7b56"],
        skill_primary_columns: ["\u653f\u7b56"],
        secondary_context_columns: ["\u5317\u4eac\u5e02_t2m", "\u5929\u6c14"],
        day_profiles: {
          target_day_tomorrow: {
            text: { "\u653f\u7b56": [{ semantic: "event_phase:special_event_present", domain: "policy" }] },
          },
        },
        reasoning_contract: "Review every secondary context column.",
      },
      deterministic_sample_features: {
        rule_match_evidence_rows: [
          {
            rule_id: "r6",
            horizon_index: 29,
            condition_exogenous_values: { "\u653f\u7b56 [day_offset=+0]": "event_phase:special_event_present" },
          },
          {
            rule_id: "r6",
            horizon_index: 30,
            condition_exogenous_values: { "\u653f\u7b56 [day_offset=+0]": "event_phase:special_event_present" },
          },
        ],
      },
      locked_skill_experience: {
        important_guidance_rules: [
          {
            id: "r6",
            condition: { kind: "exogenous_text_in", column: "\u653f\u7b56" },
            matched_horizon_indices: [29, 30],
          },
        ],
      },
    }
    const originalText = JSON.stringify(original)
    const projectedText = buildProviderContentRiskSafeContext(originalText)
    const projected = JSON.parse(projectedText)

    expect(JSON.stringify(original)).toBe(originalText)
    expect(projectedText).not.toContain("\u653f\u7b56")
    expect(projected.task_and_scenario.business_context).toBeUndefined()
    expect(projected.task_and_scenario.history_window_steps).toBe(711)
    expect(projected.prediction_points).toEqual(original.prediction_points)
    expect(projected.visible_history_target_table).toEqual(original.visible_history_target_table)
    expect(projected.recent_three_day_exogenous_table.columns).toEqual([
      "timestamp",
      "beijing_air_temperature_2m",
      "weather_category",
      "event_signal",
    ])
    expect(projected.recent_three_day_exogenous_table.rows).toEqual(original.recent_three_day_exogenous_table.rows)
    expect(projected.current_sample_exogenous_context.all_exogenous_columns).toEqual([
      "beijing_air_temperature_2m",
      "weather_category",
      "event_signal",
    ])
    expect(projected.current_sample_exogenous_context.skill_primary_columns).toEqual(["event_signal"])
    expect(projected.current_sample_exogenous_context.secondary_context_columns).toEqual([
      "beijing_air_temperature_2m",
      "weather_category",
    ])
    expect(projected.current_sample_exogenous_context.day_profiles.target_day_tomorrow.text.event_signal).toEqual([
      { semantic: "event_phase:special_event_present", domain: "policy" },
    ])
    expect(projected.deterministic_sample_features.rule_match_evidence_rows).toBeUndefined()
    expect(projected.deterministic_sample_features.rule_match_evidence_summary).toEqual([
      {
        rule_id: "r6",
        matched_horizon_indices: [29, 30],
        conditions: [{ "event_signal [day_offset=+0]": "event_phase:special_event_present" }],
      },
    ])
    expect(projected.locked_skill_experience.important_guidance_rules[0].condition.column).toBe("event_signal")
    expect(projected.provider_content_risk_projection.scope).toBe("provider_retry_only")

    const compactText = buildProviderContentRiskCompactContext(originalText)
    const compact = JSON.parse(compactText)
    expect(compact.prediction_points).toBeUndefined()
    expect(compact.gap_prediction_points).toBeUndefined()
    expect(compact.prediction_table).toEqual({
      columns: ["horizon_index", "timestamp", "raw_anchor", "pre_reflection_baseline"],
      rows: [[29, "2025/11/16 7:15", 54295.1, 53995.1]],
    })
    expect(compact.current_sample_exogenous_context).toEqual(projected.current_sample_exogenous_context)
    expect(compact.instructions.calibration).toContain("joint exogenous context")
    expect(compact.instructions.calibration).toContain("weak cross-checks")
    expect(compact.provider_content_risk_projection.level).toBe("compact_semantic")
  })

  test("normalizes provider JSON text without weakening point validation", () => {
    const exact = parseWindowReflectionText(
      JSON.stringify({
        window_index: 7,
        points: [
          { horizon_index: 0, timestamp: "2026-01-01 00:00:00", value: 1 },
          { horizon_index: 1, timestamp: "2026-01-01 00:15:00", value: 2 },
        ],
        rationale: "sample evidence",
        evidence_ids: ["rule-1"],
      }),
      7,
    )
    expect(exact.window_index).toBe(7)
    expect(exact.points).toHaveLength(2)

    const complete = parseWindowReflectionText(
      [
        '{"points":[{"horizon_index":0,"timestamp":"2026-01-01 00:00:00","value":1}]}',
        '{"points":[{"horizon_index":0,"timestamp":"2026-01-01 00:00:00","value":1},{"horizon_index":1,"timestamp":"2026-01-01 00:15:00","value":2}]}',
      ].join("\n"),
      7,
    )
    expect(complete.points).toHaveLength(2)

    const aliased = parseWindowReflectionText(
      JSON.stringify({
        predictions: [
          { horizon_index: 0, timestamp: "2026-01-01 00:00:00", value: 1 },
          { horizon_index: 1, timestamp: "2026-01-01 00:15:00", value: 2 },
        ],
        evidence_ids: ["rule-2"],
      }),
      8,
    )
    expect(aliased.window_index).toBe(8)
    expect(aliased.rationale).toContain("Normalized")

    const columnar = parseWindowReflectionText(
      JSON.stringify({
        horizon_index: [0, 1],
        timestamp: ["2026-01-01 00:00:00", "2026-01-01 00:15:00"],
        value: [1, 2],
      }),
      9,
    )
    expect(columnar.points.map((point) => point.value)).toEqual([1, 2])

    const cropped = parseWindowReflectionText(
      JSON.stringify({
        horizon_index: [0, 1, 2],
        timestamp: ["t0", "t1", "extra"],
        value: [1, 2, 999],
      }),
      9,
      2,
    )
    expect(cropped.points).toEqual([
      { horizon_index: 0, timestamp: "t0", value: 1 },
      { horizon_index: 1, timestamp: "t1", value: 2 },
    ])
    expect(() =>
      parseWindowReflectionText(
        JSON.stringify({ horizon_index: [0], timestamp: ["t0"], value: [1] }),
        9,
        2,
      ),
    ).toThrow("1/2 required points")
  })

  test("recovers structured JSON candidates from provider reasoning parts", () => {
    expect(
      providerStructuredResponseTexts("", [
        { type: "step-start" },
        { type: "reasoning", text: 'analysis\n{"selected_evidence_candidate_ids":[],"rationale":"done"}' },
      ]),
    ).toEqual([
      {
        source: "reasoning",
        text: 'analysis\n{"selected_evidence_candidate_ids":[],"rationale":"done"}',
      },
    ])
  })

  test("admits only complete proposal layouts and trims trailing points", () => {
    const expected = [
      { horizon_index: 0, timestamp: "t0", value: 10 },
      { horizon_index: 1, timestamp: "t1", value: 20 },
    ]
    expect(
      normalizeProposalLayout(
        [
          { horizon_index: 0, timestamp: "t0", value: -999 },
          { horizon_index: 1, timestamp: "t1", value: 999 },
          { horizon_index: 2, timestamp: "extra", value: 0 },
        ],
        expected,
      ),
    ).toEqual([
      { horizon_index: 0, timestamp: "t0", value: -999 },
      { horizon_index: 1, timestamp: "t1", value: 999 },
    ])
    expect(() => normalizeProposalLayout(expected.slice(0, 1), expected)).toThrow("1/2 required points")
    expect(() =>
      normalizeProposalLayout(
        [
          { horizon_index: 1, timestamp: "t1", value: 20 },
          { horizon_index: 0, timestamp: "t0", value: 10 },
        ],
        expected,
      ),
    ).toThrow("layout differs")
  })

  test("recovers DSML-wrapped point arrays and point-level evidence ids", () => {
    const response = [
      "<DSML_TOOL_CALLS>",
      '<DSML_PARAMETER name="points">',
      JSON.stringify([
        {
          horizon_index: 0,
          timestamp: "2026-01-01 00:00:00",
          value: 1,
          evidence_ids: ["rule-a"],
        },
        {
          horizon_index: 1,
          timestamp: "2026-01-01 00:15:00",
          value: 2,
          evidence_ids: ["rule-b"],
        },
      ]),
      "</DSML_PARAMETER>",
      "</DSML_TOOL_CALLS>",
    ].join("\n")
    const parsed = parseWindowReflectionText(response, 10)
    expect(parsed.evidence_ids).toEqual(["rule-a", "rule-b"])
    expect(parsed.points).toHaveLength(2)
  })

  test("accepts fenced JSON but rejects mismatched windows and invalid prose", () => {
    const fenced = parseWindowReflectionText(
      'Result:\n```json\n{"points":[{"horizon_index":0,"timestamp":"2026-01-01 00:00:00","value":1}]}\n```',
      11,
    )
    expect(fenced.window_index).toBe(11)
    expect(() =>
      parseWindowReflectionText(
        '{"window_index":12,"points":[{"horizon_index":0,"timestamp":"2026-01-01 00:00:00","value":1}]}',
        11,
      ),
    ).toThrow("valid sample-specific forecast JSON")
    expect(() =>
      parseWindowReflectionText(
        '{"points":[{"horizon_index":0,"timestamp":"2026-01-01 00:00:00","value":"not-a-number"}]}',
        11,
      ),
    ).toThrow("valid sample-specific forecast JSON")
    expect(() => parseWindowReflectionText("I cannot provide the requested points.", 11)).toThrow(
      "valid sample-specific forecast JSON",
    )
  })

  test("limits independent test-window API work to a safe default concurrency", async () => {
    const limit = createConcurrencyLimiter(WINDOW_REFLECTION_TEST_API_CONCURRENCY)
    let active = 0
    let peak = 0
    await Promise.all(
      Array.from({ length: 75 }, (_, index) =>
        limit(async () => {
          active++
          peak = Math.max(peak, active)
          await new Promise((resolve) => setTimeout(resolve, 5 + (index % 3)))
          active--
        }),
      ),
    )
    expect(peak).toBe(4)
    expect(active).toBe(0)
  })

  test("accepts a bounded test-window concurrency override and rejects unsafe values", () => {
    expect(resolveWindowReflectionTestApiConcurrency(undefined)).toBe(4)
    expect(resolveWindowReflectionTestApiConcurrency("1")).toBe(1)
    expect(resolveWindowReflectionTestApiConcurrency("10")).toBe(10)
    expect(() => resolveWindowReflectionTestApiConcurrency("0")).toThrow("between 1 and 10")
    expect(() => resolveWindowReflectionTestApiConcurrency("11")).toThrow("between 1 and 10")
    expect(() => resolveWindowReflectionTestApiConcurrency("invalid")).toThrow("between 1 and 10")
  })

  test("times out stalled window operations and invokes cancellation", async () => {
    let cancelled = false
    await expect(
      withWindowReflectionTimeout(new Promise<never>(() => {}), {
        timeoutMs: 5,
        label: "test operation",
        onTimeout: () => {
          cancelled = true
        },
      }),
    ).rejects.toThrow("test operation timed out after 5ms")
    expect(cancelled).toBe(true)
    await expect(
      withWindowReflectionTimeout(Promise.resolve("completed"), {
        timeoutMs: 100,
        label: "fast operation",
      }),
    ).resolves.toBe("completed")
  })

  test("uses a bounded forecast Python subprocess timeout", () => {
    expect(resolveForecastPythonTimeoutMs(undefined)).toBe(5 * 60 * 1000)
    expect(resolveForecastPythonTimeoutMs("1000")).toBe(1000)
    expect(() => resolveForecastPythonTimeoutMs("999")).toThrow("between 1000 and 1800000")
    expect(() => resolveForecastPythonTimeoutMs("invalid")).toThrow("between 1000 and 1800000")
  })

  test("uses bounded exponential provider retry delays", () => {
    const delays = Array.from({ length: WINDOW_REFLECTION_PROVIDER_RETRY_LIMIT + 1 }, (_, attempt) =>
      windowReflectionProviderRetryDelayMs(attempt, 1),
    )
    expect(delays).toEqual([750, 1500, 3000, 6000, 12000, 15000])
  })

  test("reuses only complete accepted corrections for the same window", () => {
    const accepted = {
      accepted: true,
      format_retry: false,
      format_errors: [],
      detector_report: { trigger: false },
      window_index: 7,
      points: [{ horizon_index: 0, timestamp: "2026-01-01 00:00:00", value: 1 }],
      rationale: "accepted",
      evidence_ids: ["rule-1"],
    }
    const expected = {
      windowIndex: 7,
      horizonIndices: [0],
      timestamps: ["2026-01-01 00:00:00"],
    }
    expect(isReusableWindowCorrection(accepted, expected)).toBe(true)
    expect(isReusableWindowCorrection(accepted, { ...expected, windowIndex: 8 })).toBe(false)
    expect(isReusableWindowCorrection({ ...accepted, accepted: false }, expected)).toBe(false)
    expect(isReusableWindowCorrection({ ...accepted, points: undefined }, expected)).toBe(false)
    expect(
      isReusableWindowCorrection(
        {
          ...accepted,
          points: [{ horizon_index: 0, timestamp: "2026-01-01 00:15:00", value: 1 }],
        },
        expected,
      ),
    ).toBe(false)
    expect(
      isReusableWindowCorrection(
        {
          ...accepted,
          points: [...accepted.points, { horizon_index: 1, timestamp: "2026-01-01 00:15:00", value: 2 }],
        },
        expected,
      ),
    ).toBe(false)
  })

  test("reuses an evidence-loop correction only while its committed state artifact is intact", async () => {
    await using tmp = await tmpdir()
    const state = {
      version: 1 as const,
      window_index: 7,
      evidences: [],
      predictions: [{ horizon_index: 0, timestamp: "2026-01-01 00:00:00", value: 1 }],
      evidence_gap: 0 as const,
      needed_evidence: "",
    }
    const statePath = path.join(tmp.path, "state.json")
    await writeFile(statePath, JSON.stringify(state, null, 2), "utf-8")
    const correction = {
      accepted: true,
      format_retry: false,
      format_errors: [],
      detector_report: { trigger: false },
      window_index: 7,
      points: state.predictions,
      rationale: "accepted",
      evidence_ids: [],
      assessment_stage: "evidence_loop_terminal",
      critic_prediction_state: state,
      critic_prediction_state_artifact: { path: statePath, sha256: await computeFileSha256(statePath) },
    }
    const expected = {
      windowIndex: 7,
      horizonIndices: [0],
      timestamps: ["2026-01-01 00:00:00"],
    }
    expect(await isReusableEvidenceLoopCorrection(correction, expected)).toBe(true)
    await writeFile(statePath, JSON.stringify({ ...state, evidence_gap: 1 }), "utf-8")
    expect(await isReusableEvidenceLoopCorrection(correction, expected)).toBe(false)
  })

  test("restores the pre-safety Evidence Loop state before resuming a legacy correction", async () => {
    await using tmp = await tmpdir()
    const correctionsDir = path.join(tmp.path, "corrections")
    const statesDir = path.join(correctionsDir, "states", "window-000007")
    await mkdir(statesDir, { recursive: true })
    const preSafetyState = {
      version: 1 as const,
      window_index: 7,
      evidences: [],
      predictions: [{ horizon_index: 0, timestamp: "2026-01-01 00:00:00", value: 9 }],
      evidence_gap: 1 as const,
      needed_evidence: "No remaining evidence resolved the gap.",
    }
    const provisionalPath = path.join(statesDir, "pre-safety.json")
    await writeFile(provisionalPath, JSON.stringify(preSafetyState, null, 2), "utf-8")
    const preSafetySha256 = await computeFileSha256(provisionalPath)
    const preSafetyPath = path.join(statesDir, `state-${preSafetySha256}.json`)
    await writeFile(preSafetyPath, JSON.stringify(preSafetyState, null, 2), "utf-8")
    const safetyState = {
      ...preSafetyState,
      predictions: [{ ...preSafetyState.predictions[0], value: 1 }],
    }
    const safetyPath = path.join(statesDir, "safety-state.json")
    await writeFile(safetyPath, JSON.stringify(safetyState, null, 2), "utf-8")
    await writeFile(
      path.join(statesDir, "current.json"),
      JSON.stringify({ path: safetyPath, sha256: await computeFileSha256(safetyPath) }, null, 2),
      "utf-8",
    )
    const correction = {
      assessment_stage: "state_terminal",
      safety_reflection: {
        status: "accepted",
        pre_safety_state_sha256: preSafetySha256,
        final_state_sha256: await computeFileSha256(safetyPath),
      },
      evidence_loop: {
        termination_reason: "evidence_exhausted",
        predictability: "unpredictable",
        accepted_transitions: 0,
        seeking_trace: [{ exhausted: true }],
        rejected_transitions: [],
      },
    }
    const restored = await restoreLegacyPreSafetyCheckpoint(correction, correctionsDir, {
      windowIndex: 7,
      horizonIndices: [0],
      timestamps: ["2026-01-01 00:00:00"],
    })
    expect(restored).toBe(true)
    expect(JSON.parse(await readFile(path.join(statesDir, "current.json"), "utf-8"))).toEqual({
      path: preSafetyPath,
      sha256: preSafetySha256,
    })
    const checkpoint = JSON.parse(await readFile(path.join(statesDir, "loop-checkpoint.json"), "utf-8"))
    expect(checkpoint.state_sha256).toBe(preSafetySha256)
    expect(checkpoint.termination_reason).toBe("evidence_exhausted")
    expect(checkpoint.predictability).toBe("unpredictable")
  })

  test("reports exact not-started and partially committed windows for explicit resume", async () => {
    await using tmp = await tmpdir()
    const outDir = path.join(tmp.path, ".forecast", "reflection", "test", "run-test", "candidate-1")
    const statesDir = path.join(outDir, "corrections", "states", "window-000000")
    await mkdir(statesDir, { recursive: true })
    const state = {
      version: 1 as const,
      window_index: 0,
      evidences: [],
      predictions: [{ horizon_index: 0, timestamp: "2026-01-01 00:00:00", value: 1 }],
      evidence_gap: 1 as const,
      needed_evidence: "Need current-window context.",
    }
    const statePath = path.join(statesDir, "state.json")
    await writeFile(statePath, JSON.stringify(state, null, 2), "utf-8")
    const stateSha256 = await computeFileSha256(statePath)
    await writeFile(
      path.join(statesDir, "current.json"),
      JSON.stringify({ path: statePath, sha256: stateSha256 }, null, 2),
      "utf-8",
    )
    await writeFile(
      path.join(statesDir, "loop-checkpoint.json"),
      JSON.stringify(
        {
          version: 1,
          window_index: 0,
          state_sha256: stateSha256,
          phase: "ready",
          selected_candidate_id: null,
          attempted_candidate_ids: ["experience:candidate-1"],
          seeking_trace: [],
          rejected_transitions: [],
          accepted_transitions: 0,
          termination_reason: null,
          updated_at: "2026-01-01T00:00:00.000Z",
        },
        null,
        2,
      ),
      "utf-8",
    )
    const makeWindow = (windowIndex: number) => ({
      window_index: windowIndex,
      detection: { trigger: false, hard_violations: [], soft_warnings: [] },
      skill_matched: false,
      llm_required: true,
      reasoning_mode: "skill_guided_test",
      failure_fallback: "raw_anchor",
      horizon_indices: [0],
      timestamps: [`2026-01-01 0${windowIndex}:00:00`],
      blind_context_path: `window-${windowIndex}.json`,
      critic_prediction_state_history: {},
    })
    await writeFile(
      path.join(outDir, "prepared.json"),
      JSON.stringify(
        {
          trigger_count: 0,
          skill_match_count: 0,
          llm_required_count: 2,
          windows: [makeWindow(0), makeWindow(1)],
        },
        null,
        2,
      ),
      "utf-8",
    )
    const auditWriter = await createWindowLlmAuditWriter(path.join(outDir, "corrections"), 0)
    await auditWriter({
      call_type: "evidence_gap",
      system_prompt: "gap-system",
      request: { window_index: 0 },
      status: "success",
      child_session_id: "status-audit-session",
      raw_response: '{"evidence_gap":1,"needed_evidence":"context","evidence_coverage":0.5,"missing_evidence":["current-window context"],"evidence_confidence":0.4,"rationale":"Current support does not establish the target-day context."}',
      parsed_response: {
        evidence_gap: 1,
        needed_evidence: "context",
        evidence_coverage: 0.5,
        missing_evidence: ["current-window context"],
        evidence_confidence: 0.4,
        rationale: "Current support does not establish the target-day context.",
      },
      error: null,
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastWindowReflectionTool.init()
        const response = await tool.execute(
          {
            subcommand: "status",
            run_id: "run-test",
            adjustment_candidate_id: "candidate-1",
            status_split: "test",
          },
          ctx,
        )
        const status = JSON.parse(response.output)
        expect(status.pending_window_indices).toEqual([0, 1])
        expect(status.windows[0]).toMatchObject({
          window_index: 0,
          status: "incomplete",
          evidence_gap: 1,
          attempted_candidate_ids: ["experience:candidate-1"],
          next_action: "seek_evidence",
        })
        expect(status.windows[0].audit_trail.llm_call_artifacts).toHaveLength(1)
        expect(status.windows[0].audit_trail.llm_call_artifacts[0]).toEndWith("call-000001.json")
        expect(status.windows[0].audit_trail.hidden_chain_of_thought_recorded).toBe(false)
        expect(status.windows[1]).toMatchObject({ window_index: 1, status: "not_started" })

        const terminalState = { ...state, evidence_gap: 0 as const, needed_evidence: "" }
        await writeFile(statePath, JSON.stringify(terminalState, null, 2), "utf-8")
        const terminalStateSha256 = await computeFileSha256(statePath)
        await writeFile(
          path.join(statesDir, "current.json"),
          JSON.stringify({ path: statePath, sha256: terminalStateSha256 }, null, 2),
          "utf-8",
        )
        await writeFile(
          path.join(statesDir, "loop-checkpoint.json"),
          JSON.stringify(
            {
              version: 1,
              window_index: 0,
              state_sha256: terminalStateSha256,
              phase: "safety_reflection",
              selected_candidate_id: null,
              attempted_candidate_ids: [],
              seeking_trace: [],
              rejected_transitions: [],
              accepted_transitions: 0,
              termination_reason: "evidence_sufficient",
              updated_at: "2026-01-01T00:01:00.000Z",
            },
            null,
            2,
          ),
          "utf-8",
        )
        const terminalResponse = await tool.execute(
          {
            subcommand: "status",
            run_id: "run-test",
            adjustment_candidate_id: "candidate-1",
            status_split: "test",
          },
          ctx,
        )
        const terminalStatus = JSON.parse(terminalResponse.output)
        expect(terminalStatus.windows[0]).toMatchObject({
          window_index: 0,
          loop_phase: "safety_reflection",
          next_action: "persist_evidence_loop_terminal",
        })
      },
    })
  })

  test("updates Critic prediction state with context evidence and fine-grained predictions", () => {
    const blindContext = JSON.stringify({
      critic_prediction_state: {
        version: 1,
        window_index: 7,
        evidences: [
          {
            evidence_candidate_id: "experience:rule-skill",
            candidate_type: "experience",
            artifact_path: "experience.json",
            artifact_sha256: "a".repeat(64),
          },
        ],
        predictions: [{ horizon_index: 0, timestamp: "2026-01-01 00:00:00", value: 10 }],
        evidence_gap: 1,
      },
      evidence_candidates: {
        context: {
          candidate_type: "context",
          evidence_candidate_id: "context:current-window",
        },
      },
      evidence_candidate_storage: {
        context: { path: "context.json", sha256: "b".repeat(64) },
      },
    })

    const updated = buildContextualCriticPredictionState(blindContext, [
      { horizon_index: 0, timestamp: "2026-01-01 00:00:00", value: 12 },
    ])

    expect(updated.evidences.map((item) => item.candidate_type)).toEqual(["experience", "context"])
    expect(updated.predictions[0]?.value).toBe(12)
    expect(updated.evidence_gap).toBe(1)
    expect(updated.needed_evidence).toBe("")
    const completeLookback = {
      columns: ["timestamp", "OT"],
      rows: [
        ["2025-12-31 22:00:00", 8],
        ["2025-12-31 23:00:00", 9],
      ],
    }
    const assessmentInput = buildEvidenceGapAssessmentInput(
      updated,
      {
        experience: {
          candidate_type: "experience",
          evidence_candidate_id: "experience:rule-skill",
          definition: "Validated adjustment guidance.",
          scope: "matching conditions only",
          payload: { committed_secret: "full experience content" },
        },
        context: {
          candidate_type: "context",
          evidence_candidate_id: "context:current-window",
          definition: "Current-window facts.",
          scope: "current window only",
          payload: { committed_secret: "full context content" },
        },
        knowledge: {
          candidate_type: "knowledge",
          evidence_candidate_id: "knowledge:frozen-task",
          definition: "Stable task semantics.",
          scope: "task-wide",
          payload: { unused_secret: "must not reach assessor" },
        },
      },
      completeLookback,
    )
    expect(assessmentInput.state).not.toHaveProperty("evidence_gap")
    expect(assessmentInput.state).not.toHaveProperty("needed_evidence")
    expect(assessmentInput.committed_evidence_candidates).toHaveLength(2)
    expect(assessmentInput.committed_evidence_candidates[0]).toHaveProperty("payload.committed_secret")
    expect(assessmentInput.committed_evidence_candidates[1]).toHaveProperty("payload.committed_secret")
    expect(assessmentInput.unused_evidence_candidate_summaries).toEqual([
      {
        evidence_candidate_id: "knowledge:frozen-task",
        candidate_type: "knowledge",
        capability_summary: "Stable task semantics. Declared scope: task-wide",
      },
    ])
    expect(JSON.stringify(assessmentInput)).not.toContain("must not reach assessor")
    expect(assessmentInput.lookback_window_target).toEqual(completeLookback)
    expect(
      requireCompleteLookbackTargetWindow({
        task_and_scenario: { history_window_steps: 2 },
        visible_history_target_table: completeLookback,
      }),
    ).toEqual(completeLookback)
    expect(() =>
      requireCompleteLookbackTargetWindow({
        task_and_scenario: { history_window_steps: 3 },
        visible_history_target_table: completeLookback,
      }),
    ).toThrow("incomplete")
    expect(
      parseEvidenceGapText(
        '{"evidence_gap":0,"needed_evidence":"","evidence_coverage":1,"missing_evidence":[],"evidence_consistency":1,"evidence_conflicts":[],"evidence_confidence":0.9,"rationale":"The committed evidence fully supports the prediction."}',
      ),
    ).toEqual({
      evidence_gap: 0,
      needed_evidence: "",
      evidence_coverage: 1,
      missing_evidence: [],
      evidence_consistency: 1,
      evidence_conflicts: [],
      evidence_confidence: 0.9,
      rationale: "The committed evidence fully supports the prediction.",
    })
    expect(
      parseEvidenceGapText(
        '{"evidence_gap":1,"needed_evidence":"Need current-window context to resolve the target-day direction.","evidence_coverage":0.5,"missing_evidence":["Current-window directional support"],"evidence_consistency":0.8,"evidence_conflicts":[],"evidence_confidence":0.4,"rationale":"The current evidence does not resolve direction for this target day."}',
      ),
    ).toEqual({
      evidence_gap: 1,
      needed_evidence: "Need current-window context to resolve the target-day direction.",
      evidence_coverage: 0.5,
      missing_evidence: ["Current-window directional support"],
      evidence_consistency: 0.8,
      evidence_conflicts: [],
      evidence_confidence: 0.4,
      rationale: "The current evidence does not resolve direction for this target day.",
    })
    expect(
      parseEvidenceGapText(
        '{"evidence_gap":1,"needed_evidence":"Need evidence that resolves the conflicting direction.","evidence_coverage":1,"missing_evidence":[],"evidence_consistency":0.3,"evidence_conflicts":["Committed context and experience imply opposite directions for the same segment."],"evidence_confidence":0.3,"rationale":"Coverage is complete, but the committed evidence is materially inconsistent."}',
      ),
    ).toMatchObject({ evidence_gap: 1, evidence_coverage: 1, evidence_consistency: 0.3 })
    expect(() =>
      parseEvidenceGapText(
        '{"evidence_gap":1,"needed_evidence":"","evidence_coverage":0.5,"missing_evidence":["support"],"evidence_consistency":1,"evidence_conflicts":[],"evidence_confidence":0.4,"rationale":"Still missing support."}',
      ),
    ).toThrow()
    expect(() =>
      parseEvidenceGapText(
        '{"evidence_gap":0,"needed_evidence":"more context","evidence_coverage":1,"missing_evidence":[],"evidence_consistency":1,"evidence_conflicts":[],"evidence_confidence":0.9,"rationale":"Contradictory output."}',
      ),
    ).toThrow()
    expect(() => parseEvidenceGapText('{"evidence_gap":0,"needed_evidence":""}')).toThrow()
    expect(() =>
      parseEvidenceGapText(
        '{"evidence_gap":0,"needed_evidence":"","evidence_coverage":1,"missing_evidence":[],"evidence_consistency":1,"evidence_conflicts":[],"evidence_confidence":0.9,"rationale":"Supported.","predictions":[]}',
      ),
    ).toThrow()
    expect(() =>
      parseEvidenceGapText(
        '{"evidence_gap":0,"needed_evidence":"","evidence_coverage":0.9,"missing_evidence":["timing support"],"evidence_consistency":1,"evidence_conflicts":[],"evidence_confidence":0.8,"rationale":"Inconsistent closed gap."}',
      ),
    ).toThrow()
    expect(() =>
      parseEvidenceGapText(
        '{"evidence_gap":0,"needed_evidence":"","evidence_coverage":1,"missing_evidence":[],"evidence_consistency":0.4,"evidence_conflicts":["Material directional conflict"],"evidence_confidence":0.5,"rationale":"A material conflict cannot close the gap."}',
      ),
    ).toThrow()
    expect(() =>
      parseEvidenceGapText(
        '{"evidence_gap":1,"needed_evidence":"timing support","evidence_coverage":0.7,"missing_evidence":[],"evidence_consistency":1,"evidence_conflicts":[],"evidence_confidence":0.6,"rationale":"Inconsistent open gap."}',
      ),
    ).toThrow()
    expect(
      parseEvidenceSeekingText(
        '{"selected_evidence_candidate_ids":["knowledge:frozen-task"],"rationale":"Knowledge addresses the gap.","exhausted":false}',
      ),
    ).toEqual({
      selected_evidence_candidate_ids: ["knowledge:frozen-task"],
      rationale: "Knowledge addresses the gap.",
      exhausted: false,
    })
    expect(
      parseEvidenceSeekingText(
        '{"selected_evidence_candidate_ids":["knowledge:frozen-task"],"rationale":"Knowledge addresses the gap."}',
      ),
    ).toEqual({
      selected_evidence_candidate_ids: ["knowledge:frozen-task"],
      rationale: "Knowledge addresses the gap.",
      exhausted: false,
    })
    expect(
      parseEvidenceSeekingText('{"selected_evidence_candidate_ids":[],"rationale":"Nothing relevant remains."}'),
    ).toEqual({
      selected_evidence_candidate_ids: [],
      rationale: "Nothing relevant remains.",
      exhausted: true,
    })
    expect(
      parseEvidenceSeekingText(
        '{"selected_evidence_candidate_ids":["context:current-window"],"rationale":"selected","exhausted":true}',
      ),
    ).toMatchObject({ selected_evidence_candidate_ids: ["context:current-window"], exhausted: true })

    const seekingCandidates = {
      knowledge: { candidate_type: "knowledge", evidence_candidate_id: "knowledge:frozen-task" },
    }
    const unused = unusedEvidenceCandidates(updated, seekingCandidates)
    expect(Object.keys(unused)).toEqual(["knowledge"])
    expect(buildEvidenceSeekingInput(updated, seekingCandidates).unused_evidence_candidates).toEqual([
      { candidate_type: "knowledge", evidence_candidate_id: "knowledge:frozen-task" },
    ])
    expect(() =>
      buildEvidenceSeekingInput({ ...updated, evidence_gap: 0, needed_evidence: "" }, seekingCandidates),
    ).toThrow("forbidden")

    const allTypes = {
      context: { candidate_type: "context", evidence_candidate_id: "context:current-window" },
      experience: { candidate_type: "experience", evidence_candidate_id: "experience:validated-rule" },
      knowledge: { candidate_type: "knowledge", evidence_candidate_id: "knowledge:frozen-task" },
    }
    expect(deterministicEvidenceTypeFromNeed("weather and validated adjustment residual rule support", allTypes)).toBe(
      "context",
    )
    expect(deterministicEvidenceTypeFromNeed("validated adjustment support", allTypes)).toBe("experience")
    expect(deterministicEvidenceTypeFromNeed("unclassified supporting signal", allTypes)).toBeNull()
    expect(selectEvidenceTypeByMode(allTypes, "fixed_order")).toBe("experience")
    expect(selectEvidenceTypeByMode({ context: allTypes.context, knowledge: allTypes.knowledge }, "fixed_order")).toBe(
      "context",
    )
    expect(selectEvidenceTypeByMode(allTypes, "random", () => 0)).toBe("experience")
    expect(selectEvidenceTypeByMode(allTypes, "random", () => 0.5)).toBe("context")
    expect(selectEvidenceTypeByMode(allTypes, "random", () => 0.999)).toBe("knowledge")
    expect(selectEvidenceTypeByMode({}, "fixed_order")).toBeNull()
    expect(finalizeEvidenceSeekingResponse(allTypes, "context", "matched context")).toEqual({
      selected_evidence_candidate_ids: ["context:current-window"],
      rationale: "matched context",
      exhausted: false,
    })
    expect(
      finalizeEvidenceSeekingResponse(
        { knowledge: allTypes.knowledge },
        "knowledge",
        "only remaining candidate",
      ),
    ).toEqual({
      selected_evidence_candidate_ids: ["knowledge:frozen-task"],
      rationale: "only remaining candidate",
      exhausted: true,
    })

    const rebound = JSON.parse(
      bindCriticStateToBlindContext(blindContext, updated, {
        path: "current-state.json",
        sha256: "c".repeat(64),
      }),
    )
    expect(rebound.critic_prediction_state).toEqual(updated)
    expect(rebound.prediction_points).toBeUndefined()
    expect(rebound.evidence_stage_status.experience_used).toBe(true)

    const initialState = {
      ...updated,
      evidences: [],
      predictions: [{ horizon_index: 0, timestamp: "2026-01-01 00:00:00", value: 10 }],
    }
    const ruleReference = {
      version: 1 as const,
      window_index: 7,
      semantics: "locked_rules_applied_to_initial_prediction_only" as const,
      experience_candidate_id: "experience:rule-skill",
      experience_candidate_artifact: { path: "experience.json", sha256: "a".repeat(64) },
      initial_predictions: initialState.predictions,
      predictions: [{ horizon_index: 0, timestamp: "2026-01-01 00:00:00", value: 11 }],
      reasoning_contract: {
        reference_only: true as const,
        not_a_state_transition: true as const,
        do_not_add_reference_delta_to_current: true as const,
      },
    }
    const fullWindowContextCandidate = {
      candidate_type: "context",
      evidence_candidate_id: "context:current-window",
      payload: {
        lookback_window_exogenous: {
          columns: ["timestamp", "temperature"],
          rows: [
            ["2025-12-31 22:00:00", 7],
            ["2025-12-31 23:00:00", 8],
          ],
          row_count: 2,
        },
        predict_window_exogenous: {
          columns: ["timestamp", "temperature"],
          rows: [["2026-01-01 00:00:00", 9]],
          row_count: 1,
        },
      },
    }
    const unifiedInput = buildUnifiedEvidenceReasoningInput({
      state: updated,
      initialState,
      selectedCandidate: { candidate_type: "knowledge", evidence_candidate_id: "knowledge:frozen-task" },
      allCandidates: {
        experience: { candidate_type: "experience", evidence_candidate_id: "experience:rule-skill" },
        context: fullWindowContextCandidate,
        knowledge: { candidate_type: "knowledge", evidence_candidate_id: "knowledge:frozen-task" },
      },
      lookbackTargetWindow: completeLookback,
      experienceRuleReference: ruleReference,
    })
    expect(unifiedInput.reasoning_contract).toMatchObject({
      reasoner_type: "unified",
      output_role: "Generate a prediction proposal only; state mutation is performed by a separate reducer.",
      selected_candidate_is_only_new_evidence: true,
      jointly_reason_over_fused_evidence: true,
    })
    expect(unifiedInput.provisional_evidence_fusion.fused_evidence_ids).toEqual([
      "experience:rule-skill",
      "context:current-window",
      "knowledge:frozen-task",
    ])
    expect(unifiedInput.provisional_evidence_fusion.evidence_attachments[1]?.auxiliary).toEqual({
      lookback_window_exogenous: fullWindowContextCandidate.payload.lookback_window_exogenous,
      predict_window_exogenous: fullWindowContextCandidate.payload.predict_window_exogenous,
    })
    expect(unifiedInput.lookback_window_target).toEqual(completeLookback)
    expect(() =>
      buildUnifiedEvidenceReasoningInput({
        state: updated,
        selectedCandidate: { candidate_type: "knowledge", evidence_candidate_id: "knowledge:frozen-task" },
        allCandidates: {},
        initialState,
        lookbackTargetWindow: completeLookback,
        experienceRuleReference: ruleReference,
      }),
    ).toThrow("is unavailable")
  })

  test("rejects non-positive concurrency limits", () => {
    expect(() => createConcurrencyLimiter(0)).toThrow("positive integer")
    expect(() => createConcurrencyLimiter(1.5)).toThrow("positive integer")
  })

  test("requires a resumable rerun for provider and structured-output failures only", () => {
    expect(windowOutcomeRequiresRerun("provider_failed")).toBe(true)
    expect(windowOutcomeRequiresRerun("schema_failed")).toBe(true)
    expect(windowOutcomeRequiresRerun("response_empty")).toBe(true)
    expect(windowOutcomeRequiresRerun("output_truncated")).toBe(true)
    expect(windowOutcomeRequiresRerun("format_failed")).toBe(true)
    expect(windowOutcomeRequiresRerun("semantic_failed")).toBe(false)
    expect(windowOutcomeRequiresRerun("accepted")).toBe(false)
  })

  test("reopens only abnormal Evidence Loop checkpoints that did not exhaust evidence", () => {
    const checkpoint = (overrides: Record<string, unknown> = {}) => ({
      version: 1,
      window_index: 7,
      state_sha256: "a".repeat(64),
      phase: "terminated",
      selected_candidate_id: null,
      attempted_candidate_ids: [],
      seeking_trace: [],
      rejected_transitions: [],
      accepted_transitions: 0,
      termination_reason: "provider_failed",
      predictability: "predictable",
      updated_at: "2026-08-22T00:00:00.000Z",
      ...overrides,
    })

    expect(shouldReopenAbnormalEvidenceLoopCheckpoint(checkpoint())).toBe(true)
    expect(
      shouldReopenAbnormalEvidenceLoopCheckpoint(checkpoint({ termination_reason: "schema_failed" })),
    ).toBe(true)
    expect(
      shouldReopenAbnormalEvidenceLoopCheckpoint(
        checkpoint({ seeking_trace: [{ selected_evidence_candidate_ids: ["knowledge:frozen-task"], exhausted: true }] }),
      ),
    ).toBe(false)
    expect(
      shouldReopenAbnormalEvidenceLoopCheckpoint(checkpoint({ predictability: "unpredictable" })),
    ).toBe(false)
    expect(
      shouldReopenAbnormalEvidenceLoopCheckpoint(checkpoint({ termination_reason: "evidence_sufficient" })),
    ).toBe(false)
    expect(
      shouldReopenAbnormalEvidenceLoopCheckpoint(checkpoint({ termination_reason: "evidence_exhausted" })),
    ).toBe(false)
    expect(shouldReopenAbnormalEvidenceLoopCheckpoint(checkpoint({ phase: "ready" }))).toBe(false)
  })

  test("rejects resetting the reflection run root itself", async () => {
    await using tmp = await tmpdir()
    const runRoot = path.join(tmp.path, ".forecast", "reflection", "test", "run-test")
    await mkdir(runRoot, { recursive: true })
    await expect(resetReflectionRunDirectory(runRoot, runRoot)).rejects.toThrow("Unsafe reflection run directory")
  })

  test("accepts only a matching completed artifact and detects stale metadata", async () => {
    await using tmp = await tmpdir()
    const resultDir = path.join(tmp.path, ".forecast", "reflection", "validation", "run-val", "raw")
    const metrics = { mse: 1, mae: 0.5, wape: 0.25, mase: null }
    const record = ForecastReflectionValidationRecord.parse({
      validation_run_id: "run-val",
      baseline_kind: "raw_anchor",
      adjustment_candidate_id: null,
      status: "enabled_no_trigger",
      trigger_count: 0,
      corrected_count: 0,
      fallback_count: 0,
      baseline_metrics: metrics,
      final_metrics: metrics,
      delta_mse: 0,
      result_path: ".forecast/reflection/validation/run-val/raw/reflection_eval.json",
      completed_at: "2026-07-15T00:00:00.000Z",
    })
    await mkdir(resultDir, { recursive: true })
    const predictionsPath = path.join(resultDir, "final_predictions.csv")
    const auditPath = path.join(resultDir, "leakage-audit.json")
    const evalPath = path.join(resultDir, "reflection_eval.json")
    await writeFile(predictionsPath, "predicted\n1\n", "utf-8")
    await writeFile(auditPath, '{"passed":true}\n', "utf-8")
    const artifact = {
      controller_version: "1",
      source_run_id: "run-val",
      split: "val",
      baseline_kind: "raw_anchor",
      adjustment_candidate_id: null,
      status: "enabled_no_trigger",
      trigger_count: 0,
      corrected_count: 0,
      fallback_count: 0,
      baseline_metrics: metrics,
      final_metrics: metrics,
      delta_mse: 0,
      final_predictions_path: ".forecast/reflection/validation/run-val/raw/final_predictions.csv",
      final_predictions_sha256: await computeFileSha256(predictionsPath),
      leakage_audit_sha256: await computeFileSha256(auditPath),
      completed_at: "2026-07-15T00:00:00.000Z",
    }
    await writeFile(evalPath, `${JSON.stringify(artifact)}\n`, "utf-8")

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const valid = await validateWindowReflectionRecordArtifact(path.join(tmp.path, ".forecast"), record, "val")
        expect(valid?.auditPassed).toBe(true)

        const predictionAlias = path.join(tmp.path, "final-predictions-alias.csv")
        await symlink(predictionsPath, predictionAlias)
        await writeFile(
          evalPath,
          `${JSON.stringify({ ...artifact, final_predictions_path: predictionAlias })}\n`,
          "utf-8",
        )
        expect(
          (await validateWindowReflectionRecordArtifact(path.join(tmp.path, ".forecast"), record, "val"))
            ?.auditPassed,
        ).toBe(true)

        await writeFile(evalPath, `${JSON.stringify({ ...artifact, trigger_count: 1 })}\n`, "utf-8")
        expect(await validateWindowReflectionRecordArtifact(path.join(tmp.path, ".forecast"), record, "val")).toBeNull()

        await writeFile(evalPath, `${JSON.stringify(artifact)}\n`, "utf-8")
        await writeFile(predictionsPath, "predicted\n2\n", "utf-8")
        expect(await validateWindowReflectionRecordArtifact(path.join(tmp.path, ".forecast"), record, "val")).toBeNull()
      },
    })
  })

  test("rejects a reflection policy changed after its hash was frozen", async () => {
    await using tmp = await tmpdir()
    const reflectionDir = path.join(tmp.path, ".forecast", "reflection")
    const policyPath = path.join(reflectionDir, "policy.json")
    const policy = {
      controller_version: "1",
      enabled: true,
      max_reflection_rounds: 3,
      max_format_retries: 2,
      trigger_policy: { hard_violation_count: 1, independent_soft_violation_count: 2 },
      constraints: {
        nonnegative: true,
        absolute_min: 0,
        absolute_max: 100,
        soft_abs_step: 10,
        max_abs_step: 20,
        window_mean_abs_delta_limit: 10,
        point_emergency_delta_limit: 20,
      },
      validation_gate: { metric: "mse", accept_if: "final_mse <= baseline_mse", zero_trigger: "enabled_no_trigger" },
    }
    await mkdir(reflectionDir, { recursive: true })
    await writeFile(policyPath, `${JSON.stringify(policy)}\n`, "utf-8")
    const policySha256 = await computeFileSha256(policyPath)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const expected = {
          enabled: true,
          controllerVersion: "1",
          maxReflectionRounds: 3,
          maxFormatRetries: 2,
          policySha256,
        }
        expect(
          await validateWindowReflectionPolicyArtifact(
            path.join(tmp.path, ".forecast"),
            ".forecast/reflection/policy.json",
            expected,
          ),
        ).not.toBeNull()

        await writeFile(policyPath, `${JSON.stringify({ ...policy, max_reflection_rounds: 2 })}\n`, "utf-8")
        expect(
          await validateWindowReflectionPolicyArtifact(
            path.join(tmp.path, ".forecast"),
            ".forecast/reflection/policy.json",
            expected,
          ),
        ).toBeNull()
      },
    })
  })

  test("rejects an unsafe adjustment candidate id before resolving artifacts", async () => {
    await using tmp = await tmpdir()
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastWindowReflectionTool.init()
        await expect(
          tool.execute(
            {
              subcommand: "run_validation",
              run_id: "run-val",
              adjustment_candidate_id: "../stale-candidate",
            },
            ctx,
          ),
        ).rejects.toThrow("Invalid adjustment_candidate_id")
      },
    })
  })

  test("does not execute validation after an explicit task opt-out", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path, { reflection_enabled: false })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastWindowReflectionTool.init()
        await expect(tool.execute({ subcommand: "run_validation", run_id: "run-val" }, ctx)).rejects.toThrow(
          "disabled in the frozen task",
        )
      },
    })
  })

  test("freezes policy only before forecast and never reconfigures a completed freeze", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        await writeState("forecast")
        const tool = await ForecastWindowReflectionTool.init()
        await expect(tool.execute({ subcommand: "configure" }, ctx)).rejects.toThrow(
          "only be frozen before entering the forecast phase",
        )

        await writeState("skills-review")
        const state = ForecastPhaseState.parse({
          ...(await readPhaseState()),
          reflection: {
            configured: true,
            enabled: true,
            policy_path: ".forecast/reflection/policy.json",
          },
        })
        await writePhaseState(state)
        await expect(tool.execute({ subcommand: "configure" }, ctx)).rejects.toThrow(
          "already frozen and cannot be reconfigured",
        )
      },
    })
  })

  test("runs validation and test reflection only in their fixed phases", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastWindowReflectionTool.init()
        await writeState("post-forecast")
        await expect(tool.execute({ subcommand: "run_validation", run_id: "run-val" }, ctx)).rejects.toThrow(
          "validation is only allowed during the forecast phase",
        )

        await writeState("forecast")
        await expect(
          tool.execute({ subcommand: "run_test", run_id: "run-test", validation_run_id: "run-val" }, ctx),
        ).rejects.toThrow("test execution is only allowed during the post-forecast phase")
      },
    })
  })

  test("does not rerun an already completed audited validation baseline", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path)
    const metrics = { mse: 1, mae: 0.5, wape: 0.25, mase: null }
    const resultDir = path.join(tmp.path, ".forecast", "reflection", "validation", "run-val", "raw")
    const resultPath = ".forecast/reflection/validation/run-val/raw/reflection_eval.json"
    const record = ForecastReflectionValidationRecord.parse({
      validation_run_id: "run-val",
      baseline_kind: "raw_anchor",
      adjustment_candidate_id: null,
      status: "enabled_no_trigger",
      trigger_count: 0,
      corrected_count: 0,
      fallback_count: 0,
      baseline_metrics: metrics,
      final_metrics: metrics,
      delta_mse: 0,
      result_path: resultPath,
      completed_at: "2026-07-15T00:00:00.000Z",
    })
    await mkdir(resultDir, { recursive: true })
    const predictionsPath = path.join(resultDir, "final_predictions.csv")
    const auditPath = path.join(resultDir, "leakage-audit.json")
    await writeFile(predictionsPath, "predicted\n1\n", "utf-8")
    await writeFile(auditPath, '{"passed":true}\n', "utf-8")
    const artifact = {
      controller_version: "1",
      source_run_id: "run-val",
      split: "val",
      baseline_kind: "raw_anchor",
      adjustment_candidate_id: null,
      status: "enabled_no_trigger",
      trigger_count: 0,
      corrected_count: 0,
      fallback_count: 0,
      baseline_metrics: metrics,
      final_metrics: metrics,
      delta_mse: 0,
      final_predictions_path: ".forecast/reflection/validation/run-val/raw/final_predictions.csv",
      final_predictions_sha256: await computeFileSha256(predictionsPath),
      leakage_audit_sha256: await computeFileSha256(auditPath),
      completed_at: "2026-07-15T00:00:00.000Z",
    }
    await writeFile(path.join(resultDir, "reflection_eval.json"), `${JSON.stringify(artifact)}\n`, "utf-8")
    const policyDir = path.join(tmp.path, ".forecast", "reflection")
    const policyPath = path.join(policyDir, "policy.json")
    await writeFile(
      policyPath,
      `${JSON.stringify({
        controller_version: "1",
        enabled: true,
        max_reflection_rounds: 3,
        max_format_retries: 2,
        trigger_policy: { hard_violation_count: 1, independent_soft_violation_count: 2 },
        constraints: {
          nonnegative: true,
          absolute_min: 0,
          absolute_max: 100,
          soft_abs_step: 10,
          max_abs_step: 20,
          window_mean_abs_delta_limit: 10,
          point_emergency_delta_limit: 20,
        },
        validation_gate: { metric: "mse", accept_if: "final_mse <= baseline_mse", zero_trigger: "enabled_no_trigger" },
      })}\n`,
      "utf-8",
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        await writeState("forecast")
        const state = ForecastPhaseState.parse({
          ...(await readPhaseState()),
          reflection: {
            enabled: true,
            configured: true,
            policy_path: ".forecast/reflection/policy.json",
            policy_sha256: await computeFileSha256(policyPath),
            controller_version: "1",
            max_reflection_rounds: 3,
            max_format_retries: 2,
            validation_records: [record],
            test_records: [],
          },
        })
        await writePhaseState(state)
        const tool = await ForecastWindowReflectionTool.init()
        await expect(tool.execute({ subcommand: "run_validation", run_id: "run-val" }, ctx)).rejects.toThrow(
          "already complete and cannot be rerun",
        )
      },
    })
  })
})
