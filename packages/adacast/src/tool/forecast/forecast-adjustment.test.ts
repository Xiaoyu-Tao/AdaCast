import { afterEach, describe, expect, test } from "bun:test"
import { mkdir, readFile, writeFile } from "fs/promises"
import { createHash } from "node:crypto"
import path from "path"
import matter from "gray-matter"
import { Instance } from "@/project/instance"
import { Question } from "@/question"
import { MessageID, SessionID } from "@/session/schema"
import { tmpdir } from "../../../test/fixture/fixture"
import { ForecastAdjustmentApplyTool } from "./forecast-adjustment-apply"
import { ForecastAdjustmentCandidateTool } from "./forecast-adjustment-candidate"
import { ForecastAdjustmentDecisionTool } from "./forecast-adjustment-decision"
import { ForecastAdjustmentDiagnosticsTool } from "./forecast-adjustment-diagnostics"
import { ForecastAdjustmentFitRunsTool } from "./forecast-adjustment-fit-runs"
import { ForecastAdjustmentPolicyTool } from "./forecast-adjustment-policy"
import { ForecastAdjustmentSkillTool } from "./forecast-adjustment-skill"
import { ForecastAdjustmentSourceModelsTool } from "./forecast-adjustment-source-models"
import { normalizeForecastStateFrontmatter } from "./forecast-progress"
import { computeSpecHash } from "./hash"
import { statisticalModelArtifact } from "./model-family"
import { AdjustmentUserPriorParameterCompletion, ForecastPhaseState } from "./types"

const ctx = {
  sessionID: SessionID.make("ses_forecast-adjustment-test"),
  messageID: MessageID.make("msg_forecast-adjustment-test"),
  callID: "forecast-adjustment-test-call",
  agent: "forecast-adjustment-test-agent",
  abort: AbortSignal.any([]),
  messages: [],
  metadata: () => {},
  ask: async () => {},
}

async function writeJson(file: string, value: unknown) {
  await mkdir(path.dirname(file), { recursive: true })
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`, "utf-8")
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex")
}

async function writeState(directory: string) {
  await mkdir(path.join(directory, ".forecast"), { recursive: true })
  const state = ForecastPhaseState.parse({
    phase: "forecast",
    skills_confirmed: true,
    skills_confirmed_at: "2026-06-03T00:00:00.000Z",
    pre_forecast_completed: true,
    pre_forecast_completed_at: "2026-06-03T00:00:00.000Z",
    task_frozen_at: "2026-06-03T00:00:00.000Z",
    transitions: [],
  })
  await writeFile(path.join(directory, ".forecast", "STATE.md"), matter.stringify("\n# Forecast State Log\n", state))
}

async function readState(directory: string) {
  return ForecastPhaseState.parse(
    normalizeForecastStateFrontmatter(
      matter(await readFile(path.join(directory, ".forecast", "STATE.md"), "utf-8")).data as Record<string, unknown>,
    ),
  )
}

async function writeStateFrontmatter(directory: string, state: ForecastPhaseState) {
  await writeFile(
    path.join(directory, ".forecast", "STATE.md"),
    matter.stringify("\n# Forecast State Log\n", state),
    "utf-8",
  )
}

async function setAutoMode(directory: string, enabled = true) {
  const state = await readState(directory)
  await writeStateFrontmatter(directory, {
    ...state,
    auto_mode: {
      ...state.auto_mode,
      enabled,
      active_agent: state.phase === "post-forecast" ? "critic" : "forecaster",
      state: "running",
      pause_reason: null,
    },
  })
}

async function waitForPendingQuestion() {
  for (let attempt = 0; attempt < 30; attempt++) {
    const pending = await Question.list()
    if (pending[0]) return pending[0]
    await new Promise((resolve) => setTimeout(resolve, 5))
  }
  throw new Error("No pending question was created")
}

async function answerPendingQuestion(label: string) {
  const pending = await waitForPendingQuestion()
  const options = pending.questions[0]?.options.map((option) => option.label) ?? []
  const selected = options.includes(label)
    ? label
    : label.startsWith("A")
      ? (options.find((option) => option.startsWith("A")) ?? label)
      : label.startsWith("B")
        ? (options.find((option) => option.startsWith("B")) ?? label)
        : label
  await Question.reply({
    requestID: pending.id,
    answers: [[selected]],
  })
  return pending
}

async function answerPendingQuestionOption(index = 0) {
  const pending = await waitForPendingQuestion()
  const label = pending.questions[0]?.options[index]?.label
  if (!label) throw new Error(`Pending question option ${index} is missing`)
  await Question.reply({
    requestID: pending.id,
    answers: [[label]],
  })
  return pending
}

async function writeSourceSet(directory: string, sourceModelSetId = "source-set") {
  await writeJson(path.join(directory, ".forecast", "adjustments", "source-model-set.json"), {
    source_model_set_id: sourceModelSetId,
    created_at: "2026-06-03T00:00:00.000Z",
    created_from: "post_raw_forecast_search",
    members: [
      {
        family: "deep_learning",
        validation_run_id: "run-val",
        model: "DLinear",
        spec_hash: "validation-spec-hash",
        validation_metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
      },
    ],
  })
  await writeJson(path.join(directory, ".forecast", "adjustments", "fit-runs", "manifest.json"), {
    source_model_set_id: sourceModelSetId,
    created_at: "2026-06-03T00:00:00.000Z",
    updated_at: "2026-06-03T00:00:00.000Z",
    status: "complete",
    members: [
      {
        family: "deep_learning",
        validation_run_id: "run-val",
        model: "DLinear",
        validation_spec_hash: "validation-spec-hash",
        fit_spec_hash: "fit-spec-hash",
        fit_run_id: "fit-run",
        status: "success",
        fit_method: "runner_reuse_validation_front_half",
        spec_path: ".forecast/adjustments/fit-runs/specs/fit-run.json",
        fit_run_path: ".forecast/adjustments/fit-runs/runs/fit-run",
        predictions_path: ".forecast/adjustments/fit-runs/runs/fit-run/predictions.csv",
        eval_path: ".forecast/adjustments/fit-runs/runs/fit-run/eval.json",
        metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
        error: null,
      },
    ],
  })
  await writeJson(
    path.join(directory, ".forecast", "adjustments", "diagnostics", "fit", "fit-run", "diagnostics.json"),
    {
      status: "success",
      split_scope: "fit_only_validation_front",
      observed_condition_domains: {
        horizon_indices: Array.from({ length: 96 }, (_, index) => index),
        timestamp_hours: Array.from({ length: 24 }, (_, index) => index),
        timestamp_months: Array.from({ length: 12 }, (_, index) => index + 1),
        raw_predicted_original: { min: -1000, max: 1000 },
        exogenous_numeric: {
          temperature: { finite_count: 100, min: -50, max: 60 },
        },
      },
    },
  )
  const state = ForecastPhaseState.parse(
    normalizeForecastStateFrontmatter(
      matter(await readFile(path.join(directory, ".forecast", "STATE.md"), "utf-8")).data as Record<string, unknown>,
    ),
  )
  await writeFile(
    path.join(directory, ".forecast", "STATE.md"),
    matter.stringify("\n# Forecast State Log\n", {
      ...state,
      adjustment: {
        ...state.adjustment,
        enabled: true,
        mode: "rule_skill",
        source_model_set_locked: true,
        source_model_set_path: ".forecast/adjustments/source-model-set.json",
        fit_run_manifest_path: ".forecast/adjustments/fit-runs/manifest.json",
        fit_runs_ready: true,
      },
    }),
    "utf-8",
  )
}

async function writeLockedActiveSkill(
  directory: string,
  phase: ForecastPhaseState["phase"] = "post-forecast",
  options: { validationRunId?: string; model?: string } = {},
) {
  const validationRunId = options.validationRunId ?? "run-val"
  const model = options.model ?? "DLinear"
  await writeValidationRunArtifacts(
    directory,
    validationRunId,
    {
      model,
      phase: "forecast",
      eval_split: "val",
    },
    { mse: 2, mae: 1, wape: 0.5, mase: null },
  )
  await writeJson(path.join(directory, ".forecast", "adjustments", "policy.json"), {
    enabled: true,
    mode: "rule_skill",
    strategy: "b_comprehensive",
  })
  await writeJson(path.join(directory, ".forecast", "adjustments", "skills", "active", "manifest.json"), {
    version: "v001",
    status: "locked",
    candidate_id: "candidate-test",
    source_model_set_id: "source-set",
    accepted_at: "2026-06-03T00:00:00.000Z",
    locked_at: "2026-06-03T00:00:00.000Z",
    validation_results: [`.forecast/adjustments/validation/${validationRunId}/candidate-test/adjusted_eval.json`],
    policy_path: ".forecast/adjustments/policy.json",
    parent_version: null,
  })
  await writeJson(
    path.join(
      directory,
      ".forecast",
      "adjustments",
      "validation",
      validationRunId,
      "candidate-test",
      "adjusted_eval.json",
    ),
    {
      run_id: validationRunId,
      candidate_id: "candidate-test",
      split: "val",
      raw_eval_path: `.forecast/runs/${validationRunId}/eval.json`,
      adjusted_predictions_path: `.forecast/adjustments/validation/${validationRunId}/candidate-test/adjusted_predictions.csv`,
      raw_metrics: { mse: 2, mae: 1, wape: 0.5, mase: null },
      adjusted_metrics: { mse: 1, mae: 0.8, wape: 0.4, mase: null },
      delta_vs_raw: { mse: -1, mae: -0.2, wape: -0.1, mase: null },
      acceptance_decision: { accepted: true, reason: "accepted", checks: {} },
      timestamp: "2026-06-03T00:00:00.000Z",
    },
  )

  const state = await readState(directory)
  await writeStateFrontmatter(directory, {
    ...state,
    phase,
    adjustment: {
      ...state.adjustment,
      enabled: true,
      mode: "rule_skill",
      strategy: "b_comprehensive",
      user_decision: "enabled_b",
      user_decision_at: "2026-06-03T00:00:00.000Z",
      policy_path: ".forecast/adjustments/policy.json",
      active_skill_version: "v001",
      active_skill_path: ".forecast/adjustments/skills/active",
      forecaster_locked: true,
      no_effective_skill: false,
      test_application_decision: "pending",
      test_application_decision_at: null,
      critic_user_confirmed: null,
    },
  })
}

async function writeRawTestRun(
  directory: string,
  runId = "run-test-best",
  mse = 0.75,
  spec: Record<string, unknown> = { model: "DLinear", phase: "post-forecast", eval_split: "test" },
  options: { writePredictions?: boolean } = {},
) {
  const model = String(spec.model ?? "DLinear")
  let sourceRunId: string | null = null
  const manifestPath = path.join(directory, ".forecast", "adjustments", "skills", "active", "manifest.json")
  try {
    const manifest = JSON.parse(await readFile(manifestPath, "utf-8")) as { validation_results?: string[] }
    for (const validationPath of manifest.validation_results ?? []) {
      const validation = JSON.parse(await readFile(path.join(directory, validationPath), "utf-8")) as {
        run_id?: string
      }
      if (!validation.run_id) continue
      const validationSpec = JSON.parse(
        await readFile(path.join(directory, ".forecast", "runs", validation.run_id, "spec.json"), "utf-8"),
      ) as Record<string, unknown>
      if (String(validationSpec.model ?? "") === model) {
        sourceRunId = validation.run_id
        break
      }
    }
  } catch {
    sourceRunId = null
  }
  sourceRunId ??= `${runId}-source`
  const sourceRunDir = path.join(directory, ".forecast", "runs", sourceRunId)
  if (!(await Bun.file(path.join(sourceRunDir, "eval.json")).exists())) {
    await writeValidationRunArtifacts(
      directory,
      sourceRunId,
      {
        ...spec,
        phase: "forecast",
        eval_split: "val",
      },
      { mse: mse + 0.25 },
    )
  }

  const runDir = path.join(directory, ".forecast", "runs", runId)
  await writeJson(path.join(runDir, "spec.json"), {
    ...spec,
  })
  const predictions = "timestamp,predicted\n2026-01-01T00:00:00Z,1\n"
  const actual = "actual"
  const pred = "pred"
  const trainScale = JSON.stringify({ train_scale: 1 })
  if (options.writePredictions ?? true) {
    await writeFile(path.join(runDir, "predictions.csv"), predictions, "utf-8")
  }
  await writeFile(path.join(runDir, "actual.npy"), actual, "utf-8")
  await writeFile(path.join(runDir, "pred.npy"), pred, "utf-8")
  await writeFile(path.join(runDir, "train_scale.json"), trainScale, "utf-8")
  await writeJson(path.join(runDir, "model", "reuse.json"), {
    source_run_id: sourceRunId,
    source_artifact_kind: "torch_state_dict",
    source_artifact_sha256: sha256("test-artifact"),
    foundation_revision: null,
    training_performed: false,
    evaluated_split: "test",
  })
  await writeJson(path.join(runDir, "eval.json"), {
    run_id: runId,
    evaluator_hash: "eval-hash",
    spec_hash: "test-spec-hash",
    metrics: { mse, mae: mse / 2, wape: mse / 3, mase: null },
    eval_split: "test",
    timestamp: "2026-06-03T00:00:00.000Z",
    artifact_binding: {
      model_artifact: {
        kind: "torch_state_dict",
        sha256: sha256("test-artifact"),
        source_run_id: sourceRunId,
        foundation_revision: null,
      },
      prediction_artifacts: {
        predictions_sha256: sha256(predictions),
        actual_sha256: sha256(actual),
        pred_sha256: sha256(pred),
        train_scale_sha256: sha256(trainScale),
      },
    },
  })
}

async function addAcceptedValidationToLockedSkill(directory: string, validationRunId: string, model: string) {
  await writeValidationRunArtifacts(
    directory,
    validationRunId,
    {
      model,
      phase: "forecast",
      eval_split: "val",
    },
    { mse: 2, mae: 1, wape: 0.5, mase: null },
  )
  const validationPath = `.forecast/adjustments/validation/${validationRunId}/candidate-test/adjusted_eval.json`
  await writeJson(path.join(directory, validationPath), {
    run_id: validationRunId,
    candidate_id: "candidate-test",
    split: "val",
    raw_eval_path: `.forecast/runs/${validationRunId}/eval.json`,
    adjusted_predictions_path: `.forecast/adjustments/validation/${validationRunId}/candidate-test/adjusted_predictions.csv`,
    raw_metrics: { mse: 2, mae: 1, wape: 0.5, mase: null },
    adjusted_metrics: { mse: 1, mae: 0.8, wape: 0.4, mase: null },
    delta_vs_raw: { mse: -1, mae: -0.2, wape: -0.1, mase: null },
    acceptance_decision: { accepted: true, reason: "accepted", checks: {} },
    timestamp: "2026-06-03T00:00:00.000Z",
  })
  const manifestPath = path.join(directory, ".forecast", "adjustments", "skills", "active", "manifest.json")
  const manifest = JSON.parse(await readFile(manifestPath, "utf-8"))
  await writeJson(manifestPath, {
    ...manifest,
    validation_results: [...manifest.validation_results, validationPath],
  })
}

async function writeValidationRunArtifacts(
  directory: string,
  runId: string,
  spec: Record<string, unknown>,
  metrics: { mse: number; mae?: number; wape?: number; mase?: number | null },
) {
  const runDir = path.join(directory, ".forecast", "runs", runId)
  await writeJson(path.join(runDir, "spec.json"), spec)
  const predictions = "timestamp,actual,predicted\n2026-01-01T00:00:00Z,1,1.1\n"
  const actual = "actual"
  const pred = "pred"
  const trainScale = JSON.stringify({ train_scale: 1 })
  await writeFile(path.join(runDir, "predictions.csv"), predictions, "utf-8")
  await writeFile(path.join(runDir, "actual.npy"), actual, "utf-8")
  await writeFile(path.join(runDir, "pred.npy"), pred, "utf-8")
  await writeFile(path.join(runDir, "train_scale.json"), trainScale, "utf-8")
  const model = String(spec.model ?? "")
  const statisticalArtifact = statisticalModelArtifact(model)
  const reusableArtifact = statisticalArtifact?.filename ?? "checkpoint.pth"
  const artifactKind = statisticalArtifact?.kind ?? "torch_state_dict"
  await mkdir(path.join(runDir, "model"), { recursive: true })
  await writeFile(path.join(runDir, "model", reusableArtifact), "test-artifact", "utf-8")
  await writeJson(path.join(runDir, "model", "metadata.json"), {
    kind: statisticalArtifact?.metadataKind ?? "checkpoint",
    path: `model/${reusableArtifact}`,
    sha256: sha256("test-artifact"),
    model,
    eval_split: "val",
    phase: "forecast",
    trained_on: "train",
    format: statisticalArtifact?.format ?? "torch_state_dict",
  })
  await writeJson(path.join(runDir, "eval.json"), {
    run_id: runId,
    evaluator_hash: "eval-hash",
    spec_hash: runId,
    metrics: {
      mse: metrics.mse,
      mae: metrics.mae ?? metrics.mse / 2,
      wape: metrics.wape ?? metrics.mse / 3,
      mase: metrics.mase ?? null,
    },
    eval_split: "val",
    timestamp: "2026-06-03T00:00:00.000Z",
    artifact_binding: {
      model_artifact: {
        kind: artifactKind,
        sha256: sha256("test-artifact"),
        source_run_id: runId,
        foundation_revision: null,
      },
      prediction_artifacts: {
        predictions_sha256: sha256(predictions),
        actual_sha256: sha256(actual),
        pred_sha256: sha256(pred),
        train_scale_sha256: sha256(trainScale),
      },
    },
  })
}

async function writeFitReplayArtifacts(directory: string) {
  await writeValidationRunArtifacts(
    directory,
    "run-val",
    { model: "DLinear", phase: "forecast", eval_split: "val" },
    { mse: 1, mae: 0.5, wape: 0.25, mase: null },
  )

  const runDir = path.join(directory, ".forecast", "adjustments", "fit-runs", "runs", "fit-run")
  const predictions = "timestamp,actual,predicted\n2026-01-01T00:00:00Z,1,1.1\n"
  const actual = "actual"
  const pred = "pred"
  const trainScale = JSON.stringify({ train_scale: 1 })
  await writeJson(path.join(runDir, "spec.json"), {
    model: "DLinear",
    phase: "forecast",
    eval_split: "adjustment_fit",
  })
  await writeFile(path.join(runDir, "predictions.csv"), predictions, "utf-8")
  await writeFile(path.join(runDir, "actual.npy"), actual, "utf-8")
  await writeFile(path.join(runDir, "pred.npy"), pred, "utf-8")
  await writeFile(path.join(runDir, "train_scale.json"), trainScale, "utf-8")
  await writeJson(path.join(runDir, "model", "reuse.json"), {
    kind: "model_reuse",
    model: "DLinear",
    source_run_id: "run-val",
    source_artifact_kind: "torch_state_dict",
    source_artifact_sha256: sha256("test-artifact"),
    foundation_revision: null,
    training_performed: false,
    evaluated_split: "adjustment_fit",
  })
  await writeJson(path.join(runDir, "eval.json"), {
    run_id: "fit-run",
    evaluator_hash: "eval-hash",
    spec_hash: "fit-run",
    metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
    eval_split: "adjustment_fit",
    timestamp: "2026-06-03T00:00:00.000Z",
    artifact_binding: {
      model_artifact: {
        kind: "torch_state_dict",
        sha256: sha256("test-artifact"),
        source_run_id: "run-val",
        foundation_revision: null,
      },
      prediction_artifacts: {
        predictions_sha256: sha256(predictions),
        actual_sha256: sha256(actual),
        pred_sha256: sha256(pred),
        train_scale_sha256: sha256(trainScale),
      },
    },
  })
}

function experiment(runId: string, family: string, mse: number, decision?: "keep" | "discard") {
  return {
    run_id: runId,
    model: family === "statistical" ? "ARIMA" : "DLinear",
    status: "success",
    metrics: { mse, mae: mse / 2, wape: mse / 3, mase: null },
    spec: {
      model: family === "statistical" ? "ARIMA" : "DLinear",
      eval_split: "val",
      phase: "forecast",
      adjustment_fit_holdout_ratio: 1 / 2,
    },
    window_features: { model_family: family },
    elapsed_seconds: 1,
    timestamp: "2026-06-03T00:00:00.000Z",
    ...(decision ? { decision } : {}),
  }
}

afterEach(async () => {
  await Instance.disposeAll()
})

describe("forecast adjustment tools", () => {
  test("decision ask_start records the user-selected lightweight prediction reasoning strategy", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        const resultPromise = tool.execute({ subcommand: "ask_start" }, ctx)
        const question = await answerPendingQuestion("轻量预测推理")
        const result = JSON.parse((await resultPromise).output)

        expect(question.questions[0].custom).toBe(false)
        expect(question.questions[0].header).toBe("生成式预测推理")
        expect(question.questions[0].question).toContain("是否启用生成式预测推理")
        expect(question.questions[0].question).not.toContain("后修正")
        expect(question.questions[0].options.map((option) => option.label)).toEqual([
          "不开启",
          "轻量预测推理",
          "深度预测推理",
        ])
        expect(question.questions[0].question).not.toMatch(/token|耗时|用时|运行时间/i)
        expect(question.questions[0].options.map((option) => option.description).join("\n")).not.toMatch(
          /token|耗时|用时|运行时间/i,
        )
        expect(result.decision).toBe("enabled_a")
      },
    })

    const policy = JSON.parse(await readFile(path.join(tmp.path, ".forecast", "adjustments", "policy.json"), "utf-8"))
    const state = await readState(tmp.path)

    expect(policy).toMatchObject({
      enabled: true,
      mode: "rule_skill",
      strategy: "a_lightweight",
      max_candidate_iterations: 2,
      auto_confirm_if_validation_passes: false,
    })
    expect(state.adjustment).toMatchObject({
      enabled: true,
      mode: "rule_skill",
      strategy: "a_lightweight",
      user_decision: "enabled_a",
      user_decision_actor: "user",
      test_application_decision: "pending",
    })
    expect(state.adjustment.user_decision_at).toEqual(expect.any(String))
  })

  test("decision auto_start records the Auto Mode LLM choice without opening a question", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await setAutoMode(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        const result = JSON.parse(
          (
            await tool.execute(
              {
                subcommand: "auto_start",
                auto_start_choice: "enabled_b",
                reason:
                  "Multiple model families retain systematic validation error and the task has safe calendar covariates.",
              },
              ctx,
            )
          ).output,
        )

        expect(result.decision).toBe("enabled_b")
        expect(result.decision_actor).toBe("llm_auto")
        await expect(
          tool.execute(
            {
              subcommand: "auto_start",
              auto_start_choice: "disabled",
              reason: "A later duplicate decision must not overwrite the already recorded Auto Mode choice.",
            },
            ctx,
          ),
        ).rejects.toThrow("before the start decision is recorded")
        expect(await Question.list()).toHaveLength(0)
      },
    })

    const state = await readState(tmp.path)
    expect(state.adjustment).toMatchObject({
      enabled: true,
      strategy: "b_comprehensive",
      user_decision: "enabled_b",
      user_decision_actor: "llm_auto",
      user_decision_reason:
        "Multiple model families retain systematic validation error and the task has safe calendar covariates.",
    })
  })

  test("decision mode isolation rejects manual questions in Auto Mode and auto decisions in Manual mode", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        await expect(
          tool.execute(
            {
              subcommand: "auto_start",
              auto_start_choice: "disabled",
              reason: "The raw forecast is already stable and no split-safe correction signal is available.",
            },
            ctx,
          ),
        ).rejects.toThrow("only allowed while Auto Mode is enabled")

        await setAutoMode(tmp.path)
        await expect(tool.execute({ subcommand: "ask_start" }, ctx)).rejects.toThrow(
          "unavailable while Auto Mode is enabled",
        )
        expect(await Question.list()).toHaveLength(0)
      },
    })
  })

  test("decision ask_start cannot overwrite a recorded start decision", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        const resultPromise = tool.execute({ subcommand: "ask_start" }, ctx)
        await answerPendingQuestion("轻量预测推理")
        await resultPromise

        await expect(tool.execute({ subcommand: "ask_start" }, ctx)).rejects.toThrow(
          "before the start decision is recorded",
        )
        expect(await Question.list()).toHaveLength(0)
      },
    })
  })

  test("decision ask_test_application records whether the locked skill is kept for test", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeLockedActiveSkill(tmp.path)
    await writeRawTestRun(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        const resultPromise = tool.execute(
          { subcommand: "ask_test_application", raw_test_run_id: "run-test-best" },
          ctx,
        )
        const question = await answerPendingQuestion("应用生成式预测推理")
        const result = JSON.parse((await resultPromise).output)

        expect(question.questions[0].custom).toBe(false)
        expect(question.questions[0].header).toBe("测试阶段生成式预测推理")
        expect(question.questions[0].question).not.toContain("后修正")
        expect(question.questions[0].question).toContain("验证集")
        expect(question.questions[0].question).toContain("MSE: raw 2.0000 -> adjusted 1.0000")
        expect(question.questions[0].question).toContain("run-test-best")
        expect(question.questions[0].options.map((option) => option.label)).toEqual([
          "应用生成式预测推理",
          "保持原始结果",
        ])
        expect(result.decision).toBe("apply")
        expect(result.raw_test_run_id).toBe("run-test-best")
        expect(result.active_skill_version).toBe("v001")
      },
    })

    const state = await readState(tmp.path)
    expect(state.adjustment.test_application_decision).toBe("apply")
    expect(state.adjustment.test_application_run_id).toBe("run-test-best")
    expect(state.adjustment.test_application_decision_at).toEqual(expect.any(String))
    expect(state.adjustment.test_application_decision_actor).toBe("user")
    expect(state.adjustment.critic_user_confirmed).toEqual(expect.any(String))
  })

  test("decision auto_test_application lets the Auto Mode Critic apply a locked skill without a question", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeLockedActiveSkill(tmp.path)
    await setAutoMode(tmp.path)
    await writeRawTestRun(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        const result = JSON.parse(
          (
            await tool.execute(
              {
                subcommand: "auto_test_application",
                raw_test_run_id: "run-test-best",
                auto_test_application_choice: "apply",
                reason:
                  "The matching validation run shows a material MSE gain with simultaneous MAE and WAPE improvement.",
              },
              ctx,
            )
          ).output,
        )

        expect(result.decision).toBe("apply")
        expect(result.decision_actor).toBe("auto_policy")
        expect(result.matched_validation_run_id).toBe("run-val")
        expect(await Question.list()).toHaveLength(0)
      },
    })

    const state = await readState(tmp.path)
    expect(state.adjustment.test_application_decision).toBe("apply")
    expect(state.adjustment.test_application_decision_actor).toBe("auto_policy")
    expect(state.adjustment.test_application_run_id).toBe("run-test-best")
    expect(state.adjustment.critic_user_confirmed).toEqual(expect.any(String))
  })

  test("decision auto_test_application rejects skipping an eligible model", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeLockedActiveSkill(tmp.path)
    await setAutoMode(tmp.path)
    await writeRawTestRun(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        await expect(
          tool.execute(
            {
              subcommand: "auto_test_application",
              raw_test_run_id: "run-test-best",
              auto_test_application_choice: "skip",
            },
            ctx,
          ),
        ).rejects.toThrow("Auto Mode defaults to applying the locked skill to every eligible test model")
        expect(await Question.list()).toHaveLength(0)
      },
    })

    const state = await readState(tmp.path)
    expect(state.adjustment.test_application_decision).toBe("pending")
    expect(state.adjustment.test_application_decision_actor).toBeNull()
    expect(state.adjustment.critic_user_confirmed).toBeNull()
  })

  test("manual test collection asks about an eligible non-raw-best model", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeLockedActiveSkill(tmp.path, "post-forecast", {
      validationRunId: "run-val-segrnn",
      model: "SegRNN",
    })
    await writeRawTestRun(tmp.path, "run-dlinear-best", 0.3, {
      model: "DLinear",
      phase: "post-forecast",
      eval_split: "test",
    })
    await writeRawTestRun(tmp.path, "run-segrnn", 0.4, {
      model: "SegRNN",
      phase: "post-forecast",
      eval_split: "test",
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        const prepared = JSON.parse(
          (
            await tool.execute(
              {
                subcommand: "prepare_test_applications",
                test_run_ids: ["run-dlinear-best", "run-segrnn"],
              },
              ctx,
            )
          ).output,
        )
        expect(prepared.eligible.map((entry: { test_run_id: string }) => entry.test_run_id)).toEqual(["run-segrnn"])
        expect(prepared.ineligible.map((entry: { test_run_id: string }) => entry.test_run_id)).toEqual([
          "run-dlinear-best",
        ])

        const decisionPromise = tool.execute({ subcommand: "ask_test_application" }, ctx)
        const question = await answerPendingQuestion("应用生成式预测推理")
        const result = JSON.parse((await decisionPromise).output)
        expect(question.questions[0].question).toContain("SegRNN")
        expect(question.questions[0].question).toContain("run-segrnn")
        expect(question.questions[0].question).not.toContain("run-dlinear-best")
        expect(result.decisions).toHaveLength(1)
        expect(result.decisions[0].raw_test_run_id).toBe("run-segrnn")
      },
    })

    const state = await readState(tmp.path)
    expect(state.adjustment.test_collection_run_ids).toEqual(["run-dlinear-best", "run-segrnn"])
    expect(state.adjustment.test_eligible_run_ids).toEqual(["run-segrnn"])
    expect(state.adjustment.test_application_records).toHaveLength(1)
    expect(state.adjustment.test_application_records[0]).toMatchObject({
      test_run_id: "run-segrnn",
      validation_run_id: "run-val-segrnn",
      model: "SegRNN",
      decision: "apply",
      decision_actor: "user",
    })
  })

  test("Auto Mode defaults every eligible non-raw-best model to apply", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeLockedActiveSkill(tmp.path, "post-forecast", {
      validationRunId: "run-val-segrnn",
      model: "SegRNN",
    })
    await setAutoMode(tmp.path)
    await writeRawTestRun(tmp.path, "run-dlinear-best", 0.3, {
      model: "DLinear",
      phase: "post-forecast",
      eval_split: "test",
    })
    await writeRawTestRun(tmp.path, "run-segrnn", 0.4, {
      model: "SegRNN",
      phase: "post-forecast",
      eval_split: "test",
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        await tool.execute(
          {
            subcommand: "prepare_test_applications",
            test_run_ids: ["run-dlinear-best", "run-segrnn"],
          },
          ctx,
        )
        const result = JSON.parse((await tool.execute({ subcommand: "auto_test_application" }, ctx)).output)
        expect(result.decisions).toHaveLength(1)
        expect(result.decisions[0]).toMatchObject({
          decision: "apply",
          decision_actor: "auto_policy",
          raw_test_run_id: "run-segrnn",
          matched_validation_run_id: "run-val-segrnn",
        })
        await tool.execute(
          {
            subcommand: "prepare_test_applications",
            test_run_ids: ["run-dlinear-best", "run-segrnn"],
          },
          ctx,
        )
        expect(await Question.list()).toHaveLength(0)
      },
    })

    const state = await readState(tmp.path)
    expect(state.adjustment.test_application_records).toHaveLength(1)
    expect(state.adjustment.test_application_records[0]).toMatchObject({
      test_run_id: "run-segrnn",
      decision: "apply",
      decision_actor: "auto_policy",
    })
    expect(state.adjustment.test_application_decision).toBe("apply")
    expect(state.adjustment.test_application_decision_actor).toBe("auto_policy")
    expect(state.adjustment.test_application_records.some((record) => record.test_run_id === "run-dlinear-best")).toBe(
      false,
    )
  })

  test("manual mode asks separately for every eligible model in the test collection", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeLockedActiveSkill(tmp.path, "post-forecast", {
      validationRunId: "run-val-dlinear",
      model: "DLinear",
    })
    await addAcceptedValidationToLockedSkill(tmp.path, "run-val-segrnn", "SegRNN")
    await writeRawTestRun(tmp.path, "run-dlinear", 0.3, {
      model: "DLinear",
      phase: "post-forecast",
      eval_split: "test",
    })
    await writeRawTestRun(tmp.path, "run-segrnn", 0.4, {
      model: "SegRNN",
      phase: "post-forecast",
      eval_split: "test",
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        const prepared = JSON.parse(
          (
            await tool.execute(
              {
                subcommand: "prepare_test_applications",
                test_run_ids: ["run-dlinear", "run-segrnn"],
              },
              ctx,
            )
          ).output,
        )
        expect(prepared.eligible).toHaveLength(2)

        const decisionPromise = tool.execute({ subcommand: "ask_test_application" }, ctx)
        const firstQuestion = await answerPendingQuestion("应用生成式预测推理")
        const secondQuestion = await answerPendingQuestion("保持原始结果")
        const result = JSON.parse((await decisionPromise).output)

        expect(firstQuestion.questions[0].question).toContain("run-dlinear")
        expect(secondQuestion.questions[0].question).toContain("run-segrnn")
        expect(firstQuestion.questions[0].question).not.toMatch(/token|耗时|用时|运行时间/i)
        expect(firstQuestion.questions[0].options.map((option) => option.description).join("\n")).not.toMatch(
          /token|耗时|用时|运行时间/i,
        )
        expect(result.decisions.map((decision: { decision: string }) => decision.decision)).toEqual(["apply", "skip"])
      },
    })

    const state = await readState(tmp.path)
    expect(state.adjustment.test_application_records).toHaveLength(2)
    expect(
      state.adjustment.test_application_records.map((record) => ({
        run: record.test_run_id,
        decision: record.decision,
        actor: record.decision_actor,
      })),
    ).toEqual([
      { run: "run-dlinear", decision: "apply", actor: "user" },
      { run: "run-segrnn", decision: "skip", actor: "user" },
    ])
  })

  test("decision ask_test_application requires the selected raw test run id", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeLockedActiveSkill(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        await expect(tool.execute({ subcommand: "ask_test_application" }, ctx)).rejects.toThrow(
          "found no pending eligible test runs",
        )
        expect(await Question.list()).toHaveLength(0)
      },
    })
  })

  test("decision ask_test_application is blocked before post-forecast", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeLockedActiveSkill(tmp.path, "forecast")

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        await expect(tool.execute({ subcommand: "ask_test_application" }, ctx)).rejects.toThrow(
          "only allowed during post-forecast",
        )
        expect(await Question.list()).toHaveLength(0)
      },
    })
  })

  test("decision ask_test_application requires the raw test run to match the accepted validation spec", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeLockedActiveSkill(tmp.path)
    await writeRawTestRun(tmp.path, "run-test-other-model", 0.75, {
      model: "TiDE",
      phase: "post-forecast",
      eval_split: "test",
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        await expect(
          tool.execute({ subcommand: "ask_test_application", raw_test_run_id: "run-test-other-model" }, ctx),
        ).rejects.toThrow("not validation-accepted on this test model spec")
        expect(await Question.list()).toHaveLength(0)
      },
    })
  })

  test("decision ask_test_application requires raw test predictions before showing the user decision", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeLockedActiveSkill(tmp.path)
    await writeRawTestRun(
      tmp.path,
      "run-test-best",
      0.75,
      { model: "DLinear", phase: "post-forecast", eval_split: "test" },
      { writePredictions: false },
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        await expect(
          tool.execute({ subcommand: "ask_test_application", raw_test_run_id: "run-test-best" }, ctx),
        ).rejects.toThrow("predictions.csv")
        expect(await Question.list()).toHaveLength(0)
      },
    })
  })

  test("decision ask_escalation upgrades A to B while preserving frozen evidence", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    const state = await readState(tmp.path)
    await writeStateFrontmatter(tmp.path, {
      ...state,
      adjustment: {
        ...state.adjustment,
        enabled: true,
        mode: "rule_skill",
        strategy: "a_lightweight",
        user_decision: "enabled_a",
        user_decision_at: "2026-06-03T00:00:00.000Z",
        source_model_set_locked: true,
        source_model_set_path: ".forecast/adjustments/source-model-set.json",
        fit_run_manifest_path: ".forecast/adjustments/fit-runs/manifest.json",
        fit_runs_ready: true,
      },
    })
    await writeJson(path.join(tmp.path, ".forecast", "adjustments", "policy.json"), {
      enabled: true,
      mode: "rule_skill",
      strategy: "a_lightweight",
      max_rules: 4,
      max_candidate_iterations: 2,
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        const resultPromise = tool.execute(
          { subcommand: "ask_escalation", reason: "Lightweight prediction reasoning did not pass validation" },
          ctx,
        )
        const question = await answerPendingQuestion("切换至深度预测推理")
        const result = JSON.parse((await resultPromise).output)

        expect(question.questions[0].custom).toBe(false)
        expect(question.questions[0].options.map((option) => option.label)).toEqual([
          "切换至深度预测推理",
          "结束生成式预测推理",
        ])
        expect(question.questions[0].question).not.toMatch(/token|耗时|用时|运行时间/i)
        expect(question.questions[0].options.map((option) => option.description).join("\n")).not.toMatch(
          /token|耗时|用时|运行时间/i,
        )
        expect(result.decision).toBe("enabled_b")
      },
    })

    const updated = await readState(tmp.path)
    const policy = JSON.parse(await readFile(path.join(tmp.path, ".forecast", "adjustments", "policy.json"), "utf-8"))
    expect(policy.strategy).toBe("b_comprehensive")
    expect(policy.max_candidate_iterations).toBe(7)
    expect(updated.adjustment.strategy).toBe("b_comprehensive")
    expect(updated.adjustment.user_decision).toBe("enabled_b")
    expect(updated.adjustment.source_model_set_locked).toBe(true)
    expect(updated.adjustment.fit_runs_ready).toBe(true)
  })

  test("decision ask_escalation requires the user to have selected A first", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        await expect(tool.execute({ subcommand: "ask_escalation" }, ctx)).rejects.toThrow(
          "requires an active lightweight prediction reasoning decision",
        )
        expect(await Question.list()).toHaveLength(0)
      },
    })
  })

  test("decision auto_escalation upgrades A to B and preserves frozen evidence", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await setAutoMode(tmp.path)
    const state = await readState(tmp.path)
    await writeStateFrontmatter(tmp.path, {
      ...state,
      adjustment: {
        ...state.adjustment,
        enabled: true,
        mode: "rule_skill",
        strategy: "a_lightweight",
        user_decision: "enabled_a",
        user_decision_at: "2026-06-03T00:00:00.000Z",
        user_decision_actor: "llm_auto",
        user_decision_reason: "A small rule budget initially matched the expected correction complexity.",
        source_model_set_locked: true,
        source_model_set_path: ".forecast/adjustments/source-model-set.json",
        fit_run_manifest_path: ".forecast/adjustments/fit-runs/manifest.json",
        fit_runs_ready: true,
      },
    })
    await writeJson(path.join(tmp.path, ".forecast", "adjustments", "policy.json"), {
      enabled: true,
      mode: "rule_skill",
      strategy: "a_lightweight",
      max_rules: 4,
      max_candidate_iterations: 2,
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        const result = JSON.parse(
          (
            await tool.execute(
              {
                subcommand: "auto_escalation",
                auto_escalation_choice: "upgrade_b",
                reason:
                  "Validation-front diagnostics still show distinct horizon and calendar bias buckets not covered by A.",
              },
              ctx,
            )
          ).output,
        )
        expect(result.decision).toBe("enabled_b")
        expect(result.decision_actor).toBe("llm_auto")
        expect(await Question.list()).toHaveLength(0)
      },
    })

    const updated = await readState(tmp.path)
    const policy = JSON.parse(await readFile(path.join(tmp.path, ".forecast", "adjustments", "policy.json"), "utf-8"))
    expect(policy.max_candidate_iterations).toBe(7)
    expect(updated.adjustment.user_decision).toBe("enabled_b")
    expect(updated.adjustment.user_decision_actor).toBe("llm_auto")
    expect(updated.adjustment.source_model_set_locked).toBe(true)
    expect(updated.adjustment.fit_runs_ready).toBe(true)
  })

  test("decision auto_escalation stops when A failure leaves no credible remaining signal", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await setAutoMode(tmp.path)
    const state = await readState(tmp.path)
    await writeStateFrontmatter(tmp.path, {
      ...state,
      adjustment: {
        ...state.adjustment,
        enabled: true,
        mode: "rule_skill",
        strategy: "a_lightweight",
        user_decision: "enabled_a",
        user_decision_at: "2026-06-03T00:00:00.000Z",
        user_decision_actor: "llm_auto",
        user_decision_reason: "A was selected for a small expected correction surface.",
      },
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentDecisionTool.init()
        const result = JSON.parse(
          (
            await tool.execute(
              {
                subcommand: "auto_escalation",
                auto_escalation_choice: "stop",
                reason:
                  "A exhausted its budget and the validation-front diagnostics contain no stable untested blind-safe signal.",
              },
              ctx,
            )
          ).output,
        )
        expect(result.decision).toBe("stop_adjustment")
        expect(result.decision_actor).toBe("llm_auto")
        expect(await Question.list()).toHaveLength(0)
      },
    })

    const updated = await readState(tmp.path)
    expect(updated.adjustment.no_effective_skill).toBe(true)
    expect(updated.adjustment.forecaster_locked).toBe(true)
    expect(updated.adjustment.test_application_decision).toBe("skip")
    expect(updated.adjustment.test_application_decision_actor).toBe("llm_auto")
  })

  test("policy configure writes policy and state adjustment summary", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentPolicyTool.init()
        await tool.execute(
          {
            subcommand: "configure",
            policy: { enabled: true, mode: "rule_skill", max_point_delta_ratio: 0.05 },
          },
          ctx,
        )
      },
    })

    const policy = JSON.parse(await readFile(path.join(tmp.path, ".forecast", "adjustments", "policy.json"), "utf-8"))
    const state = ForecastPhaseState.parse(
      normalizeForecastStateFrontmatter(
        matter(await readFile(path.join(tmp.path, ".forecast", "STATE.md"), "utf-8")).data as Record<string, unknown>,
      ),
    )

    expect(policy.enabled).toBe(true)
    expect(policy.max_point_delta_ratio).toBe(0.05)
    expect(policy.min_relative_mse_gain).toBe(-1)
    expect(state.adjustment.enabled).toBe(true)
    expect(state.adjustment.policy_path).toBe(".forecast/adjustments/policy.json")
  })

  test("policy fixes the MSE gain floor at minus one and normalizes legacy positive values", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeJson(path.join(tmp.path, ".forecast", "adjustments", "policy.json"), {
      enabled: true,
      mode: "rule_skill",
      strategy: "b_comprehensive",
      min_relative_mse_gain: 0.05,
      max_point_delta_ratio: 0.04,
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentPolicyTool.init()
        const read = JSON.parse((await tool.execute({ subcommand: "read" }, ctx)).output)
        expect(read.policy.enabled).toBe(true)
        expect(read.policy.strategy).toBe("b_comprehensive")
        expect(read.policy.max_point_delta_ratio).toBe(0.04)
        expect(read.policy.min_relative_mse_gain).toBe(-1)

        await expect(
          tool.execute(
            {
              subcommand: "configure",
              policy: { min_relative_mse_gain: 0.01 as never },
            },
            ctx,
          ),
        ).rejects.toThrow()
      },
    })
  })

  test("explicit expert task cannot activate a later exploratory candidate", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeJson(path.join(tmp.path, ".forecast", "task.json"), {
      adjustment_skill_priors: "Use the supplied expert correction experience.",
    })
    await writeJson(
      path.join(
        tmp.path,
        ".forecast",
        "adjustments",
        "candidates",
        "later-exploratory",
        "candidate.json",
      ),
      {
        candidate_id: "later-exploratory",
        source_model_set_id: "source-set",
        parent_skill_version: null,
        status: "candidate",
        origin: "llm_exploratory",
        user_prior_contract: null,
        rules: [
          {
            id: "rule",
            condition: { kind: "always" },
            action: { type: "add", value: 1 },
            rationale: "Exploratory candidate.",
            evidence_refs: ["validation-front:test"],
            bounds: {},
          },
        ],
        skill_markdown: "# Later exploratory",
        created_at: "2026-07-20T00:00:00.000Z",
        llm_provenance: {},
      },
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const skillTool = await ForecastAdjustmentSkillTool.init()
        await expect(
          skillTool.execute(
            { subcommand: "accept_candidate", candidate_id: "later-exploratory", run_id: "run-val" },
            ctx,
          ),
        ).rejects.toThrow("must retain the first user_prior_mirror")
      },
    })
  })


  test("accepted zero-trigger validation cannot be discarded as no-effective or rejected", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)
    const dormantCandidatePath = path.join(
      tmp.path,
      ".forecast",
      "adjustments",
      "candidates",
      "dormant",
      "candidate.json",
    )
    await writeJson(dormantCandidatePath, {
      candidate_id: "dormant",
      source_model_set_id: "source-set",
      parent_skill_version: null,
      status: "candidate",
      created_at: "2026-06-03T00:00:00.000Z",
      rules: [
        {
          id: "rare-event",
          condition: { kind: "timestamp_month_in", values: [2] },
          action: { type: "add", value: 1 },
          rationale: "Retain a rare blind-safe rule when its event is absent from validation-back.",
          evidence_refs: [".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"],
          bounds: {},
        },
      ],
      skill_markdown: "# Dormant skill\n\nApply the rare-event rule when it occurs.",
      llm_provenance: {},
    })
    await writeFile(
      path.join(tmp.path, ".forecast", "adjustments", "candidates", "dormant", "SKILL.md"),
      "# Dormant skill\n\nApply the rare-event rule when it occurs.\n",
      "utf-8",
    )
    await writeJson(
      path.join(tmp.path, ".forecast", "adjustments", "validation", "run-val", "dormant", "adjusted_eval.json"),
      {
        run_id: "run-val",
        candidate_id: "dormant",
        split: "val",
        raw_eval_path: ".forecast/runs/run-val/eval.json",
        adjusted_predictions_path: ".forecast/adjustments/validation/run-val/dormant/adjusted_predictions.csv",
        raw_metrics: { mse: 2, mae: 1, wape: 0.5, mase: null },
        adjusted_metrics: { mse: 2, mae: 1, wape: 0.5, mase: null },
        delta_vs_raw: { mse: 0, mae: 0, wape: 0, mase: null },
        application_summary: {
          matched_rows: 0,
          changed_rows: 0,
          triggered_rule_count: 0,
          zero_trigger: true,
          row_count: 4,
          metrics_reused_from_raw: true,
        },
        acceptance_decision: {
          accepted: true,
          reason: "accepted: candidate did not trigger and preserved the baseline exactly",
          checks: { zero_trigger: true },
        },
        timestamp: "2026-06-03T00:00:00.000Z",
      },
    )
    await writeFile(
      path.join(tmp.path, ".forecast", "adjustments", "validation", "run-val", "dormant", "leakage-audit.jsonl"),
      `${JSON.stringify({ check: "blind_input_forbidden_keys", passed: true })}\n`,
      "utf-8",
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const policyTool = await ForecastAdjustmentPolicyTool.init()
        await expect(
          policyTool.execute(
            { subcommand: "mark_no_effective", reason: "The rule did not occur in validation-back." },
            ctx,
          ),
        ).rejects.toThrow("cannot discard validation-accepted candidate")

        const skillTool = await ForecastAdjustmentSkillTool.init()
        await expect(
          skillTool.execute(
            { subcommand: "reject_candidate", candidate_id: "dormant", reason: "No validation trigger." },
            ctx,
          ),
        ).rejects.toThrow("already has a validation-accepted result")
      },
    })

    const legacyRejected = JSON.parse(await readFile(dormantCandidatePath, "utf-8"))
    await writeJson(dormantCandidatePath, { ...legacyRejected, status: "rejected" })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const skillTool = await ForecastAdjustmentSkillTool.init()
        const accepted = JSON.parse(
          (
            await skillTool.execute(
              { subcommand: "accept_candidate", candidate_id: "dormant", run_id: "run-val" },
              ctx,
            )
          ).output,
        )
        expect(accepted.accepted).toBe(true)
      },
    })

    const state = await readState(tmp.path)
    expect(state.adjustment.no_effective_skill).toBe(false)
    expect(state.adjustment.active_skill_version).toBe("v001")
  })

  test("policy configure rejects unsupported acceptance objectives", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentPolicyTool.init()
        await expect(
          tool.execute(
            {
              subcommand: "configure",
              policy: {
                enabled: true,
                mode: "rule_skill",
                primary_metric: "mae" as never,
              },
            },
            ctx,
          ),
        ).rejects.toThrow("mse")
      },
    })
  })

  test("source model freeze selects best successful run per family", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await mkdir(path.join(tmp.path, ".forecast"), { recursive: true })
    await writeValidationRunArtifacts(tmp.path, "deep-best", experiment("deep-best", "deep_learning", 1, "keep").spec, {
      mse: 1,
    })
    const statisticalBest = {
      ...experiment("stat-best", "statistical", 2),
      model: "ETS",
      spec: {
        ...experiment("stat-best", "statistical", 2).spec,
        model: "ETS",
      },
    }
    await writeValidationRunArtifacts(tmp.path, "stat-best", statisticalBest.spec, { mse: 2 })
    await writeFile(
      path.join(tmp.path, ".forecast", "history.jsonl"),
      [
        JSON.stringify(experiment("deep-slow", "deep_learning", 3)),
        JSON.stringify(experiment("deep-discarded", "deep_learning", 0.5, "discard")),
        JSON.stringify({
          ...experiment("deep-test-leak", "deep_learning", 0.1, "keep"),
          spec: { model: "DLinear", eval_split: "test", phase: "post-forecast" },
        }),
        JSON.stringify(experiment("deep-best", "deep_learning", 1, "keep")),
        JSON.stringify(statisticalBest),
        "",
      ].join("\n"),
      "utf-8",
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentSourceModelsTool.init()
        await tool.execute({ subcommand: "freeze" }, ctx)
      },
    })

    const sourceSet = JSON.parse(
      await readFile(path.join(tmp.path, ".forecast", "adjustments", "source-model-set.json"), "utf-8"),
    )
    expect(sourceSet.members.map((item: { validation_run_id: string }) => item.validation_run_id).sort()).toEqual([
      "deep-best",
      "stat-best",
    ])
    expect(
      sourceSet.members.every(
        (item: { artifacts?: { predictions_path?: string } }) => item.artifacts?.predictions_path,
      ),
    ).toBe(true)
    expect(
      sourceSet.members.find((item: { validation_run_id: string }) => item.validation_run_id === "stat-best"),
    ).toMatchObject({
      family: "statistical",
      model: "ETS",
    })
  })

  test("source model freeze infers missing family labels without collapsing model families", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await mkdir(path.join(tmp.path, ".forecast"), { recursive: true })

    const deepRecord = {
      ...experiment("deep-unlabeled", "deep_learning", 1, "keep"),
      window_features: undefined,
    }
    const statisticalRecord = {
      ...experiment("stat-unlabeled", "statistical", 2, "keep"),
      window_features: undefined,
    }
    await writeValidationRunArtifacts(tmp.path, deepRecord.run_id, deepRecord.spec, { mse: 1 })
    await writeValidationRunArtifacts(tmp.path, statisticalRecord.run_id, statisticalRecord.spec, { mse: 2 })
    await writeFile(
      path.join(tmp.path, ".forecast", "history.jsonl"),
      `${JSON.stringify(deepRecord)}\n${JSON.stringify(statisticalRecord)}\n`,
      "utf-8",
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentSourceModelsTool.init()
        await tool.execute({ subcommand: "freeze" }, ctx)
      },
    })

    const sourceSet = JSON.parse(
      await readFile(path.join(tmp.path, ".forecast", "adjustments", "source-model-set.json"), "utf-8"),
    )
    expect(sourceSet.members.map((item: { family: string }) => item.family).sort()).toEqual([
      "deep_learning",
      "statistical",
    ])
    expect(sourceSet.members.map((item: { validation_run_id: string }) => item.validation_run_id).sort()).toEqual([
      "deep-unlabeled",
      "stat-unlabeled",
    ])
  })

  test("source model freeze resolves history run ids to artifact-complete prediction runs", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await mkdir(path.join(tmp.path, ".forecast", "runs", "history-alias"), { recursive: true })

    const spec = {
      model: "DLinear",
      eval_split: "val",
      phase: "forecast",
      seed: 7,
      adjustment_fit_holdout_ratio: 1 / 2,
    }
    const artifactRunId = computeSpecHash(spec)
    await writeJson(path.join(tmp.path, ".forecast", "runs", "history-alias", "spec.json"), spec)
    await writeValidationRunArtifacts(tmp.path, artifactRunId, spec, { mse: 0.75 })
    await writeFile(
      path.join(tmp.path, ".forecast", "history.jsonl"),
      `${JSON.stringify({
        ...experiment("history-alias", "deep_learning", 0.75, "keep"),
        spec,
      })}\n`,
      "utf-8",
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentSourceModelsTool.init()
        const result = JSON.parse((await tool.execute({ subcommand: "freeze" }, ctx)).output)
        expect(result.source_model_set.members[0].validation_run_id).toBe(artifactRunId)
      },
    })

    const sourceSet = JSON.parse(
      await readFile(path.join(tmp.path, ".forecast", "adjustments", "source-model-set.json"), "utf-8"),
    )
    expect(sourceSet.members[0]).toMatchObject({
      validation_run_id: artifactRunId,
      history_run_id: "history-alias",
      artifacts: {
        predictions_path: `.forecast/runs/${artifactRunId}/predictions.csv`,
      },
    })
  })

  test("source model freeze accepts legacy Python Unicode hashes and a missing phase", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)

    const unicodeDatasetDir = path.join(tmp.path, "桌面大文件", "学习资料")
    const unicodeDatasetPath = path.join(unicodeDatasetDir, "光伏数据.csv")
    await mkdir(unicodeDatasetDir, { recursive: true })
    await writeFile(
      unicodeDatasetPath,
      [
        "date,real_power",
        ...Array.from(
          { length: 1400 },
          (_, index) => `2026-01-${String((index % 28) + 1).padStart(2, "0")}T00:00:00Z,${index}`,
        ),
        "",
      ].join("\n"),
      "utf-8",
    )
    const spec = {
      model: "SegRNN",
      dataset_path: unicodeDatasetPath,
      target_col: "real_power",
      time_col: "date",
      pred_len: 96,
      seq_len: 336,
      label_len: 336,
      train_ratio: 0.7,
      val_ratio: 0.1,
      test_ratio: 0.2,
      adjustment_fit_holdout_ratio: 1 / 2,
      metrics: ["mse", "mae"],
      eval_split: "val",
    }
    const legacyPythonRunId = "f".repeat(64)
    await writeValidationRunArtifacts(tmp.path, legacyPythonRunId, spec, {
      mse: 0.4572,
      mae: 0.4,
      wape: 0.3,
      mase: null,
    })
    await writeFile(
      path.join(tmp.path, ".forecast", "history.jsonl"),
      `${JSON.stringify({
        ...experiment("typescript-history-hash", "deep_learning", 0.4572, "keep"),
        model: "SegRNN",
        spec,
      })}\n`,
      "utf-8",
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentSourceModelsTool.init()
        await tool.execute({ subcommand: "freeze" }, ctx)
      },
    })

    const sourceSet = JSON.parse(
      await readFile(path.join(tmp.path, ".forecast", "adjustments", "source-model-set.json"), "utf-8"),
    )
    expect(sourceSet.members[0]).toMatchObject({
      validation_run_id: legacyPythonRunId,
      history_run_id: "typescript-history-hash",
      model: "SegRNN",
      spec_hash: computeSpecHash(spec),
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentFitRunsTool.init()
        await tool.execute({ subcommand: "prepare" }, ctx)
      },
    })
    const fitManifest = JSON.parse(
      await readFile(path.join(tmp.path, ".forecast", "adjustments", "fit-runs", "manifest.json"), "utf-8"),
    )
    expect(fitManifest).toMatchObject({
      status: "prepared",
      members: [
        {
          validation_run_id: legacyPythonRunId,
          validation_spec_hash: computeSpecHash(spec),
          status: "prepared",
        },
      ],
    })
  })

  test("fit run prepare writes validation-front replay specs from frozen source set", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    const datasetPath = path.join(tmp.path, "dataset.csv")
    await writeFile(
      datasetPath,
      [
        "timestamp,value",
        ...Array.from(
          { length: 600 },
          (_, index) => `2026-01-01T${String(index % 24).padStart(2, "0")}:00:00Z,${index}`,
        ),
        "",
      ].join("\n"),
      "utf-8",
    )
    const validationSpec = {
      model: "DLinear",
      dataset_path: datasetPath,
      target_col: "value",
      time_col: "timestamp",
      pred_len: 24,
      seq_len: 48,
      label_len: 24,
      train_ratio: 0.7,
      val_ratio: 0.2,
      test_ratio: 0.1,
      adjustment_fit_holdout_ratio: 1 / 2,
      metrics: ["mse", "mae"],
      eval_split: "val",
      window_features: { model_family: "deep_learning" },
    }
    await writeJson(path.join(tmp.path, ".forecast", "runs", "deep-best", "spec.json"), validationSpec)
    await writeJson(path.join(tmp.path, ".forecast", "adjustments", "source-model-set.json"), {
      source_model_set_id: "source-set",
      created_at: "2026-06-03T00:00:00.000Z",
      created_from: "post_raw_forecast_search",
      members: [
        {
          family: "deep_learning",
          validation_run_id: "deep-best",
          model: "DLinear",
          spec_hash: computeSpecHash(validationSpec),
          validation_metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
        },
      ],
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentFitRunsTool.init()
        await tool.execute({ subcommand: "prepare" }, ctx)
      },
    })

    const manifest = JSON.parse(
      await readFile(path.join(tmp.path, ".forecast", "adjustments", "fit-runs", "manifest.json"), "utf-8"),
    )
    const fitSpec = JSON.parse(
      await readFile(
        path.join(tmp.path, ".forecast", "adjustments", "fit-runs", "specs", "deep_learning-deep-best.json"),
        "utf-8",
      ),
    )
    const state = ForecastPhaseState.parse(
      normalizeForecastStateFrontmatter(
        matter(await readFile(path.join(tmp.path, ".forecast", "STATE.md"), "utf-8")).data as Record<string, unknown>,
      ),
    )

    expect(manifest.status).toBe("prepared")
    expect(manifest.members[0]).toMatchObject({
      family: "deep_learning",
      validation_run_id: "deep-best",
      status: "prepared",
      fit_method: "runner_reuse_validation_front_half",
      spec_path: ".forecast/adjustments/fit-runs/specs/deep_learning-deep-best.json",
    })
    expect(fitSpec.phase).toBe("forecast")
    expect(fitSpec.eval_split).toBe("adjustment_fit")
    expect(fitSpec.train_ratio).toBeCloseTo(0.7)
    expect(fitSpec.val_ratio).toBeCloseTo(0.2)
    expect(fitSpec.test_ratio).toBeCloseTo(0.1)
    expect(fitSpec.adjustment_fit_holdout_ratio).toBeCloseTo(1 / 2)
    expect(fitSpec.window_features).toBeUndefined()
    expect(manifest.members[0].fit_spec_hash).toBe(computeSpecHash(fitSpec))
    expect(manifest.members[0].fit_run_id).toBe(computeSpecHash(fitSpec))
    expect(state.adjustment.fit_run_manifest_path).toBe(".forecast/adjustments/fit-runs/manifest.json")
    expect(state.adjustment.fit_runs_ready).toBe(false)
  })

  test("fit run reuses a matching complete manifest without preparing or running again", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)
    await writeFitReplayArtifacts(tmp.path)
    const manifestPath = path.join(tmp.path, ".forecast", "adjustments", "fit-runs", "manifest.json")

    const output = await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastAdjustmentFitRunsTool.init()
        return JSON.parse((await tool.execute({ subcommand: "run" }, ctx)).output)
      },
    })

    expect(output.status).toBe("complete")
    expect(output.members[0].fit_run_id).toBe("fit-run")
    expect(output.members[0].status).toBe("cached")
    const persisted = JSON.parse(await readFile(manifestPath, "utf-8"))
    expect(persisted.members[0].status).toBe("cached")
  })

  test("candidate write does not impose a legacy policy rule budget", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeJson(path.join(tmp.path, ".forecast", "adjustments", "policy.json"), { max_rules: 1 })
    await writeSourceSet(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        const result = await candidateTool.execute(
            {
              subcommand: "write",
              candidate_id: "too-many-rules",
              source_model_set_id: "source-set",
              rules: [
                {
                  id: "r1",
                  condition: { kind: "always" },
                  action: { type: "multiply", value: 1.01 },
                  rationale: "First rule.",
                  evidence_refs: ["fit-a"],
                  bounds: {},
                },
                {
                  id: "r2",
                  condition: { kind: "always" },
                  action: { type: "multiply", value: 0.99 },
                  rationale: "Second rule.",
                  evidence_refs: ["fit-b"],
                  bounds: {},
                },
              ],
              skill_markdown: "# Adjustment Skill\n\nTwo rules.",
            },
            ctx,
          )
        expect(JSON.parse(result.output).written).toBe(true)
      },
    })
  })

  test("candidate write accepts combined blind-safe adjustment conditions", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        await candidateTool.execute(
          {
            subcommand: "write",
            candidate_id: "targeted-condition",
            source_model_set_id: "source-set",
            rules: [
              {
                id: "winter-late-low-prediction",
                condition: {
                  kind: "all",
                  conditions: [
                    { kind: "timestamp_month_in", values: [1, 2, 12] },
                    { kind: "horizon_index_in", values: [18, 19, 20, 21, 22, 23] },
                    { kind: "raw_predicted_original_between", min: 0, max: 100 },
                  ],
                },
                action: { type: "add", value: 2 },
                rationale: "Validation-front fit evidence showed a narrow winter late-horizon underprediction bucket.",
                evidence_refs: [".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"],
                bounds: { max_delta_ratio: 0.03 },
              },
            ],
            skill_markdown: "# Adjustment Skill\n\n## Evidence\n\nNarrow validation-front bucket only.",
          },
          ctx,
        )

      },
    })

    const candidate = JSON.parse(
      await readFile(
        path.join(tmp.path, ".forecast", "adjustments", "candidates", "targeted-condition", "candidate.json"),
        "utf-8",
      ),
    )
    expect(candidate.rules[0].condition.kind).toBe("all")
    expect(candidate.rules[0].condition.conditions.map((condition: { kind: string }) => condition.kind)).toEqual([
      "timestamp_month_in",
      "horizon_index_in",
      "raw_predicted_original_between",
    ])
  })

  test("candidate write accepts numeric exogenous feature conditions", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)
    await writeJson(path.join(tmp.path, ".forecast", "task.json"), {
      dataset_path: "dataset.csv",
      target_col: "value",
      time_col: "timestamp",
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        await candidateTool.execute(
          {
            subcommand: "write",
            candidate_id: "exogenous-condition",
            source_model_set_id: "source-set",
            rules: [
              {
                id: "high-temperature-late-horizon",
                condition: {
                  kind: "all",
                  conditions: [
                    { kind: "exogenous_numeric_between", column: "temperature", min: 25, max: 40 },
                    { kind: "horizon_index_in", values: [18, 19, 20, 21, 22, 23] },
                  ],
                },
                action: { type: "add", value: 1.5 },
                rationale: "Validation-front fit diagnostics showed high-temperature late-horizon underprediction.",
                evidence_refs: [".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"],
                bounds: { max_delta_ratio: 0.03 },
              },
            ],
            skill_markdown: "# Adjustment Skill\n\n## Evidence\n\nUses a forecast-time temperature covariate.",
          },
          ctx,
        )
      },
    })

    const candidate = JSON.parse(
      await readFile(
        path.join(tmp.path, ".forecast", "adjustments", "candidates", "exogenous-condition", "candidate.json"),
        "utf-8",
      ),
    )
    expect(candidate.rules[0].condition.conditions[0]).toMatchObject({
      kind: "exogenous_numeric_between",
      column: "temperature",
    })
  })

  test("candidate write rejects target columns as exogenous feature conditions", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)
    await writeJson(path.join(tmp.path, ".forecast", "task.json"), {
      dataset_path: "dataset.csv",
      target_col: "value",
      time_col: "timestamp",
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        await expect(
          candidateTool.execute(
            {
              subcommand: "write",
              candidate_id: "leaky-exogenous-condition",
              source_model_set_id: "source-set",
              rules: [
                {
                  id: "target-as-feature",
                  condition: { kind: "exogenous_numeric_between", column: "value", min: 0, max: 1 },
                  action: { type: "add", value: 1 },
                  rationale: "This should be rejected because it targets the actual target column.",
                  evidence_refs: ["fit"],
                  bounds: {},
                },
              ],
              skill_markdown: "# Adjustment Skill\n",
            },
            ctx,
          ),
        ).rejects.toThrow("cannot target time, target, actual, residual, or metric columns")
      },
    })
  })

  test("candidate write accepts non-fit artifacts as rule evidence", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        await candidateTool.execute(
          {
            subcommand: "write",
            candidate_id: "non-fit-evidence",
            source_model_set_id: "source-set",
            rules: [
              {
                id: "r1",
                condition: { kind: "always" },
                action: { type: "add", value: 1 },
                rationale: "Non-fit evidence is allowed without fit-domain validation.",
                evidence_refs: [".forecast/adjustments/validation/run-val/candidate/adjusted_eval.json"],
                bounds: {},
              },
            ],
            skill_markdown: "# Adjustment Skill\n",
          },
          ctx,
        )
        expect(
          await Bun.file(
            path.join(tmp.path, ".forecast", "adjustments", "candidates", "non-fit-evidence", "candidate.json"),
          ).exists(),
        ).toBe(true)
      },
    })
  })

  test("candidate write rejects condition values absent from validation-front evidence", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)
    await writeJson(
      path.join(tmp.path, ".forecast", "adjustments", "diagnostics", "fit", "fit-run", "diagnostics.json"),
      {
        status: "success",
        split_scope: "fit_only_validation_front",
        observed_condition_domains: {
          horizon_indices: [0, 1, 2],
          timestamp_hours: [8, 9, 10],
          timestamp_months: [9, 10],
          raw_predicted_original: { min: 10, max: 20 },
          exogenous_numeric: {},
        },
      },
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        await expect(
          candidateTool.execute(
            {
              subcommand: "write",
              candidate_id: "unseen-winter-rule",
              source_model_set_id: "source-set",
              rules: [
                {
                  id: "winter",
                  condition: { kind: "timestamp_month_in", values: [12] },
                  action: { type: "add", value: 1 },
                  rationale: "A winter condition cannot be invented when winter was absent from fit evidence.",
                  evidence_refs: [".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"],
                  bounds: {},
                },
              ],
              skill_markdown: "# Adjustment Skill\n",
            },
            ctx,
          ),
        ).rejects.toThrow("values not observed in the validation-front fit evidence: 12")
      },
    })
  })

  test("candidate write rejects a rule-specific point bound above the global policy", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)
    await writeJson(path.join(tmp.path, ".forecast", "adjustments", "policy.json"), {
      max_point_delta_ratio: 0.05,
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        await expect(
          candidateTool.execute(
            {
              subcommand: "write",
              candidate_id: "oversized-bound",
              source_model_set_id: "source-set",
              rules: [
                {
                  id: "r1",
                  condition: { kind: "always" },
                  action: { type: "add", value: 1 },
                  rationale: "The rule-specific bound cannot weaken the global safety policy.",
                  evidence_refs: [".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"],
                  bounds: { max_delta_ratio: 0.2 },
                },
              ],
              skill_markdown: "# Adjustment Skill\n",
            },
            ctx,
          ),
        ).rejects.toThrow("exceeds policy.max_point_delta_ratio=0.05")
      },
    })
  })

  test("candidate write rejects diagnostics from a stale source model set", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)
    const manifestPath = path.join(tmp.path, ".forecast", "adjustments", "fit-runs", "manifest.json")
    const manifest = JSON.parse(await readFile(manifestPath, "utf-8"))
    await writeJson(manifestPath, { ...manifest, source_model_set_id: "stale-source-set" })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        await expect(
          candidateTool.execute(
            {
              subcommand: "write",
              candidate_id: "stale-evidence",
              source_model_set_id: "source-set",
              rules: [
                {
                  id: "r1",
                  condition: { kind: "always" },
                  action: { type: "add", value: 1 },
                  rationale: "Stale diagnostics must not cross source-set boundaries.",
                  evidence_refs: [".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"],
                  bounds: {},
                },
              ],
              skill_markdown: "# Adjustment Skill\n",
            },
            ctx,
          ),
        ).rejects.toThrow("does not match frozen source set source-set")
      },
    })
  })

  test("candidate write rejects invalid clip actions before persistence", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        await expect(
          candidateTool.execute(
            {
              subcommand: "write",
              candidate_id: "invalid-clip",
              source_model_set_id: "source-set",
              rules: [
                {
                  id: "r1",
                  condition: { kind: "always" },
                  action: { type: "clip", min: 10, max: 5 },
                  rationale: "Invalid clip bounds must be rejected.",
                  evidence_refs: [".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"],
                  bounds: {},
                },
              ],
              skill_markdown: "# Adjustment Skill\n",
            },
            ctx,
          ),
        ).rejects.toThrow("clip action requires min <= max")
      },
    })
  })

  test("candidate write accepts user-prior style text, shifted-day, minute, and profile rules", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)
    await writeJson(
      path.join(tmp.path, ".forecast", "adjustments", "diagnostics", "fit", "fit-run", "diagnostics.json"),
      {
        status: "success",
        split_scope: "fit_only_validation_front",
        observed_condition_domains: {
          horizon_indices: [0, 1, 2, 3],
          timestamp_hours: [6, 7, 8, 9],
          timestamp_months: [1],
          timestamp_minutes_of_day: [360, 420, 480, 540],
          raw_predicted_original: { min: 50000, max: 70000 },
          exogenous_numeric: {},
          exogenous_text: {
            天气: { finite_count: 4, values: ["阴"] },
            是否极端天气: { finite_count: 4, values: ["是"] },
            "day_p1__节假日": { finite_count: 4, values: ["元旦复工"] },
          },
        },
      },
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        await candidateTool.execute(
          {
            subcommand: "write",
            candidate_id: "user-prior-mirror",
            source_model_set_id: "source-set",
            rules: [
              {
                id: "risky-day-up-only-after-morning",
                condition: {
                  kind: "all",
                  conditions: [
                    { kind: "timestamp_time_between", start: "07:15", end: "23:45" },
                    { kind: "exogenous_text_in", column: "天气", values: ["阴"] },
                    { kind: "exogenous_bool_is", column: "是否极端天气", value: true },
                    {
                      kind: "exogenous_text_contains",
                      column: "节假日",
                      values: ["元旦复工"],
                      day_offset: 1,
                    },
                  ],
                },
                action: { type: "add_profile", by: "horizon_index", values: [0, 0, 3000, 3000] },
                rationale: "Mirror the user-supplied high-risk-day prior as a validation-gated deterministic skill.",
                evidence_refs: [".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"],
                bounds: { min_delta_original: 0, max_delta_original: 3000 },
              },
            ],
            skill_markdown: "# User Prior Mirror\n\nMirrors the supplied high-risk-day correction rule.",
            llm_provenance: {},
          },
          ctx,
        )
      },
    })

    const candidate = JSON.parse(
      await readFile(
        path.join(tmp.path, ".forecast", "adjustments", "candidates", "user-prior-mirror", "candidate.json"),
        "utf-8",
      ),
    )
    expect(candidate.rules[0].condition.conditions.map((condition: { kind: string }) => condition.kind)).toContain(
      "exogenous_text_contains",
    )
    expect(candidate.rules[0].action.type).toBe("add_profile")
    expect(candidate.origin).toBe("llm_exploratory")
  })

  test("user-prior inspection reads a referenced Python file without executing it", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)
    const scriptPath = path.join(tmp.path, "expert-values.py")
    const executionMarker = path.join(tmp.path, "python-was-executed.txt")
    await writeFile(
      scriptPath,
      `from pathlib import Path\nPath(${JSON.stringify(executionMarker)}).write_text("executed")\nAMOUNT = 120\n`,
      "utf-8",
    )
    const sourceExcerpt = `Increase the forecast using the amount defined in ${scriptPath}.`
    await writeJson(path.join(tmp.path, ".forecast", "task.json"), {
      adjustment_skill_priors: sourceExcerpt,
      target_col: "target",
      time_col: "timestamp",
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        const inspected = JSON.parse(
          (
            await candidateTool.execute(
              { subcommand: "inspect_user_prior", reference_paths: [scriptPath] },
              ctx,
            )
          ).output,
        )
        expect(inspected.execution_performed).toBe(false)
        expect(inspected.evidence.references[0].content).toContain("AMOUNT = 120")
        expect(await Bun.file(executionMarker).exists()).toBe(false)

        await candidateTool.execute(
          {
            subcommand: "write",
            candidate_id: "readonly-user-prior",
            source_model_set_id: "source-set",
            origin: "user_prior_mirror",
            rules: [
              {
                id: "readonly-amount",
                condition: { kind: "always" },
                action: { type: "add", value: 120 },
                rationale: "Use the read-only expert magnitude without executing its Python source.",
                evidence_refs: [".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"],
                bounds: {},
              },
            ],
            skill_markdown: "# Read-only user prior\n",
            llm_provenance: { user_prior_mirror: true },
            user_prior_contract: {
              evidence_path: inspected.evidence_path,
              source_sha256: inspected.evidence.source_sha256,
              validation_front_match: "full",
              rule_mappings: [],
              unresolved_parameters_before_llm: [],
              parameter_completions: [
                {
                  rule_id: "readonly-amount",
                  field: "condition.kind",
                  value: "always",
                  source: "explicit_prompt",
                  rationale: "The expert instruction applies without an additional trigger.",
                },
                {
                  rule_id: "readonly-amount",
                  field: "action.type",
                  value: "add",
                  source: "explicit_prompt",
                  rationale: "Increase is represented as an additive upward correction.",
                },
                {
                  rule_id: "readonly-amount",
                  field: "action.value",
                  value: 120,
                  source: "readonly_reference",
                  evidence_ref: scriptPath,
                  rationale: "AMOUNT is explicitly defined in the inspected file.",
                },
              ],
            },
          },
          ctx,
        )

        const policyTool = await ForecastAdjustmentPolicyTool.init()
        await policyTool.execute({ subcommand: "configure", policy: { max_candidate_iterations: 1 } }, ctx)
        const exploratoryRule = {
          id: "exploratory-always",
          condition: { kind: "always" as const },
          action: { type: "add" as const, value: 1 },
          rationale: "Verify that a user-prior mirror does not consume the ordinary exploration budget.",
          evidence_refs: [".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"],
          bounds: {},
        }
        await candidateTool.execute(
          {
            subcommand: "write",
            candidate_id: "first-exploratory-after-user-prior",
            source_model_set_id: "source-set",
            rules: [exploratoryRule],
            skill_markdown: "# Independent exploratory budget\n",
          },
          ctx,
        )
        await expect(
          candidateTool.execute(
            {
              subcommand: "write",
              candidate_id: "second-exploratory-after-user-prior",
              source_model_set_id: "source-set",
              rules: [exploratoryRule],
              skill_markdown: "# Exhausted exploratory budget\n",
            },
            ctx,
          ),
        ).rejects.toThrow("independent llm_exploratory budget")
      },
    })

    const candidate = JSON.parse(
      await readFile(
        path.join(tmp.path, ".forecast", "adjustments", "candidates", "readonly-user-prior", "candidate.json"),
        "utf-8",
      ),
    )
    expect(candidate.origin).toBe("user_prior_mirror")
    expect(candidate.rules[0].action.value).toBe(120)
    expect(await Bun.file(executionMarker).exists()).toBe(false)
  })

  test("user-prior completion rejects tampered readonly evidence and accepts explicit prompt values", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)
    const scriptPath = path.join(tmp.path, "expert-values.py")
    await writeFile(scriptPath, "AMOUNT = 120\n", "utf-8")
    const sourceExcerpt = `Increase the forecast using the amount defined in ${scriptPath}.`
    await writeJson(path.join(tmp.path, ".forecast", "task.json"), {
      adjustment_skill_priors: sourceExcerpt,
      target_col: "target",
      time_col: "timestamp",
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        const inspected = JSON.parse(
          (
            await candidateTool.execute(
              { subcommand: "inspect_user_prior", reference_paths: [scriptPath] },
              ctx,
            )
          ).output,
        )
        const evidencePath = path.join(tmp.path, ...inspected.evidence_path.split("/"))
        const originalEvidence = JSON.parse(await readFile(evidencePath, "utf-8"))
        const diagnosticsPath = ".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"
        const rule = {
          id: "guarded-up",
          condition: { kind: "always" as const },
          action: { type: "add" as const, value: 120 },
          rationale: "Preserve the user direction and use only verified numeric evidence.",
          evidence_refs: [diagnosticsPath],
          bounds: {},
        }
        const baseCompletions = [
          {
            rule_id: "guarded-up",
            field: "condition.kind",
            value: "always",
            source: "explicit_prompt" as const,
            rationale: "The mapped instruction has no additional trigger.",
          },
          {
            rule_id: "guarded-up",
            field: "action.type",
            value: "add",
            source: "explicit_prompt" as const,
            rationale: "Increase is encoded as an additive correction.",
          },
        ]
        const contract = (completion: AdjustmentUserPriorParameterCompletion) => ({
          evidence_path: inspected.evidence_path,
          source_sha256: inspected.evidence.source_sha256,
          validation_front_match: "full" as const,
          rule_mappings: [{ rule_id: "guarded-up", source_excerpt: "Summarized upward adjustment guidance." }],
          unresolved_parameters_before_llm: [],
          parameter_completions: [...baseCompletions, completion],
        })

        await writeJson(evidencePath, {
          ...originalEvidence,
          references: [{ ...originalEvidence.references[0], content: "AMOUNT = 999\n" }],
        })
        await expect(
          candidateTool.execute(
            {
              subcommand: "write",
              candidate_id: "tampered-captured-content",
              source_model_set_id: "source-set",
              origin: "user_prior_mirror",
              rules: [rule],
              skill_markdown: "# Tampered evidence\n",
              llm_provenance: { user_prior_mirror: true },
              user_prior_contract: contract({
                rule_id: "guarded-up",
                field: "action.value",
                value: 120,
                source: "readonly_reference",
                evidence_ref: scriptPath,
                rationale: "Claim the captured value.",
              }),
            },
            ctx,
          ),
        ).rejects.toThrow("captured content is invalid")
        await writeJson(evidencePath, originalEvidence)

        await writeFile(scriptPath, "AMOUNT = 121\n", "utf-8")
        await expect(
          candidateTool.execute(
            {
              subcommand: "write",
              candidate_id: "changed-readonly-file",
              source_model_set_id: "source-set",
              origin: "user_prior_mirror",
              rules: [rule],
              skill_markdown: "# Changed file\n",
              llm_provenance: { user_prior_mirror: true },
              user_prior_contract: contract({
                rule_id: "guarded-up",
                field: "action.value",
                value: 120,
                source: "readonly_reference",
                evidence_ref: scriptPath,
                rationale: "The inspected file originally contained this value.",
              }),
            },
            ctx,
          ),
        ).rejects.toThrow("changed after inspection")
        await writeFile(scriptPath, "AMOUNT = 120\n", "utf-8")

        await expect(
          candidateTool.execute(
            {
              subcommand: "write",
              candidate_id: "fabricated-readonly-number",
              source_model_set_id: "source-set",
              origin: "user_prior_mirror",
              rules: [{ ...rule, action: { type: "add", value: 121 } }],
              skill_markdown: "# Fabricated file value\n",
              llm_provenance: { user_prior_mirror: true },
              user_prior_contract: contract({
                rule_id: "guarded-up",
                field: "action.value",
                value: 121,
                source: "readonly_reference",
                evidence_ref: scriptPath,
                rationale: "This value is absent from the file.",
              }),
            },
            ctx,
          ),
        ).rejects.toThrow("is not present in the verified reference")

        await candidateTool.execute(
          {
            subcommand: "write",
            candidate_id: "prompt-number-not-in-excerpt",
            source_model_set_id: "source-set",
            origin: "user_prior_mirror",
            rules: [{ ...rule, action: { type: "add", value: 121 } }],
            skill_markdown: "# Prompt value not copied literally\n",
            llm_provenance: { user_prior_mirror: true },
            user_prior_contract: contract({
              rule_id: "guarded-up",
              field: "action.value",
              value: 121,
              source: "explicit_prompt",
              rationale: "Explicit prompt values need not occur literally in the mapped excerpt.",
            }),
          },
          ctx,
        )

        await expect(
          candidateTool.execute(
            {
              subcommand: "write",
              candidate_id: "fit-rewrites-action-type",
              source_model_set_id: "source-set",
              origin: "user_prior_mirror",
              rules: [rule],
              skill_markdown: "# Invalid fit semantics\n",
              llm_provenance: { user_prior_mirror: true },
              user_prior_contract: {
                evidence_path: inspected.evidence_path,
                source_sha256: inspected.evidence.source_sha256,
                validation_front_match: "full",
                rule_mappings: [{ rule_id: "guarded-up", source_excerpt: sourceExcerpt }],
                unresolved_parameters_before_llm: [],
                parameter_completions: [
                  baseCompletions[0],
                  {
                    rule_id: "guarded-up",
                    field: "action.type",
                    value: "add",
                    source: "validation_front_fit",
                    evidence_ref: diagnosticsPath,
                    rationale: "Invalid attempt to derive semantics from matching evidence.",
                  },
                  {
                    rule_id: "guarded-up",
                    field: "action.value",
                    value: 120,
                    source: "readonly_reference",
                    evidence_ref: scriptPath,
                    rationale: "The file contains this value.",
                  },
                ],
              },
            },
            ctx,
          ),
        ).rejects.toThrow("may fill only missing numeric values")
      },
    })
  })

  test("user-prior mirror may infer magnitude when its immutable trigger is absent from validation-front", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)
    const sourceExcerpt = "When weather is snow, only increase the forecast."
    await writeJson(path.join(tmp.path, ".forecast", "task.json"), {
      adjustment_skill_priors: sourceExcerpt,
      target_col: "target",
      time_col: "timestamp",
    })
    await writeJson(
      path.join(tmp.path, ".forecast", "adjustments", "diagnostics", "fit", "fit-run", "diagnostics.json"),
      {
        status: "success",
        split_scope: "fit_only_validation_front",
        observed_condition_domains: {
          horizon_indices: [0, 1, 2, 3],
          timestamp_hours: [0, 1],
          timestamp_months: [1],
          timestamp_minutes_of_day: [0, 15, 30, 45],
          raw_predicted_original: { min: 0, max: 1000 },
          exogenous_numeric: {},
          exogenous_text: { weather: { finite_count: 4, values: ["sunny"] } },
        },
      },
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        const inspected = JSON.parse((await candidateTool.execute({ subcommand: "inspect_user_prior" }, ctx)).output)
        const rule = {
          id: "snow-up",
          condition: { kind: "exogenous_text_in" as const, column: "weather", values: ["snow"] },
          action: { type: "add" as const, value: 250 },
          rationale: "Keep the expert snow trigger and conservatively infer only its missing magnitude.",
          evidence_refs: [".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"],
          bounds: { min_delta_original: 0 },
        }
        await candidateTool.execute(
          {
            subcommand: "write",
            candidate_id: "dormant-user-prior",
            source_model_set_id: "source-set",
            origin: "user_prior_mirror",
            rules: [rule],
            skill_markdown: "# Dormant user prior\n",
            llm_provenance: { user_prior_mirror: true, numeric_completion: "llm_inferred" },
            user_prior_contract: {
              evidence_path: inspected.evidence_path,
              source_sha256: inspected.evidence.source_sha256,
              validation_front_match: "none",
              rule_mappings: [{ rule_id: "snow-up", source_excerpt: sourceExcerpt }],
              unresolved_parameters_before_llm: [{ rule_id: "snow-up", field: "action.value" }],
              parameter_completions: [
                {
                  rule_id: "snow-up",
                  field: "condition.kind",
                  value: "exogenous_text_in",
                  source: "explicit_prompt",
                  rationale: "The expert instruction names an exact weather category.",
                },
                {
                  rule_id: "snow-up",
                  field: "condition.column",
                  value: "weather",
                  source: "explicit_prompt",
                  rationale: "The expert instruction explicitly names weather.",
                },
                {
                  rule_id: "snow-up",
                  field: "condition.values",
                  value: ["snow"],
                  source: "explicit_prompt",
                  rationale: "Snow is the immutable expert trigger value.",
                },
                {
                  rule_id: "snow-up",
                  field: "action.type",
                  value: "add",
                  source: "explicit_prompt",
                  rationale: "Increase is represented as an additive upward correction.",
                },
                {
                  rule_id: "snow-up",
                  field: "action.value",
                  value: 250,
                  source: "llm_inferred",
                  rationale: "No snow rows matched; infer a conservative amount from target scale and policy bounds.",
                },
                {
                  rule_id: "snow-up",
                  field: "bounds.min_delta_original",
                  value: 0,
                  source: "explicit_prompt",
                  rationale: "The expert prior says only increase.",
                },
              ],
            },
          },
          ctx,
        )

        await expect(
          candidateTool.execute(
            {
              subcommand: "write",
              candidate_id: "rewritten-snow-trigger",
              source_model_set_id: "source-set",
              origin: "user_prior_mirror",
              rules: [{ ...rule, condition: { ...rule.condition, values: ["rain"] } }],
              skill_markdown: "# Rewritten trigger\n",
              llm_provenance: { user_prior_mirror: true },
              user_prior_contract: {
                evidence_path: inspected.evidence_path,
                source_sha256: inspected.evidence.source_sha256,
                validation_front_match: "none",
                rule_mappings: [{ rule_id: "snow-up", source_excerpt: sourceExcerpt }],
                unresolved_parameters_before_llm: [{ rule_id: "snow-up", field: "action.value" }],
                parameter_completions: [
                  {
                    rule_id: "snow-up",
                    field: "condition.kind",
                    value: "exogenous_text_in",
                    source: "explicit_prompt",
                    rationale: "The user supplied a categorical weather trigger.",
                  },
                  {
                    rule_id: "snow-up",
                    field: "condition.column",
                    value: "weather",
                    source: "explicit_prompt",
                    rationale: "Weather is explicit in the user rule.",
                  },
                  {
                    rule_id: "snow-up",
                    field: "condition.values",
                    value: ["rain"],
                    source: "explicit_prompt",
                    rationale: "Invalid attempt to replace the snow category.",
                  },
                  {
                    rule_id: "snow-up",
                    field: "action.type",
                    value: "add",
                    source: "explicit_prompt",
                    rationale: "Increase is represented as an additive correction.",
                  },
                  {
                    rule_id: "snow-up",
                    field: "action.value",
                    value: 250,
                    source: "llm_inferred",
                    rationale: "Infer only the missing amount.",
                  },
                  {
                    rule_id: "snow-up",
                    field: "bounds.min_delta_original",
                    value: 0,
                    source: "explicit_prompt",
                    rationale: "Only increase forbids a negative delta.",
                  },
                ],
              },
            },
            ctx,
          ),
        ).rejects.toThrow("semantic parameter snow-up/condition.values is not present")

        await expect(
          candidateTool.execute(
            {
              subcommand: "write",
              candidate_id: "exploratory-unobserved-snow",
              source_model_set_id: "source-set",
              rules: [rule],
              skill_markdown: "# Exploratory rule\n",
            },
            ctx,
          ),
        ).rejects.toThrow("not observed in validation-front evidence")
      },
    })

    const candidate = JSON.parse(
      await readFile(
        path.join(tmp.path, ".forecast", "adjustments", "candidates", "dormant-user-prior", "candidate.json"),
        "utf-8",
      ),
    )
    expect(candidate.origin).toBe("user_prior_mirror")
    expect(candidate.user_prior_contract.validation_front_match).toBe("none")
    expect(candidate.rules[0].condition.values).toEqual(["snow"])
    expect(candidate.rules[0].action.value).toBe(250)
  })

  test("user-prior mirror must complete held-out validation before the workflow can mark no-effective", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)
    const sourceExcerpt = "Always increase the forecast; the amount is unspecified."
    await writeJson(path.join(tmp.path, ".forecast", "task.json"), {
      adjustment_skill_priors: sourceExcerpt,
      target_col: "target",
      time_col: "timestamp",
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        const inspected = JSON.parse((await candidateTool.execute({ subcommand: "inspect_user_prior" }, ctx)).output)
        await candidateTool.execute(
          {
            subcommand: "write",
            candidate_id: "independent-user-prior",
            source_model_set_id: "source-set",
            origin: "user_prior_mirror",
            rules: [
              {
                id: "always-up",
                condition: { kind: "always" },
                action: { type: "add", value: 100 },
                rationale: "Preserve the expert direction and infer only the missing amount.",
                evidence_refs: [".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"],
                bounds: { min_delta_original: 0 },
              },
            ],
            skill_markdown: "# Independent user prior\n",
            llm_provenance: { user_prior_mirror: true },
            user_prior_contract: {
              evidence_path: inspected.evidence_path,
              source_sha256: inspected.evidence.source_sha256,
              validation_front_match: "full",
              rule_mappings: [{ rule_id: "always-up", source_excerpt: sourceExcerpt }],
              unresolved_parameters_before_llm: [{ rule_id: "always-up", field: "action.value" }],
              parameter_completions: [
                {
                  rule_id: "always-up",
                  field: "condition.kind",
                  value: "always",
                  source: "explicit_prompt",
                  rationale: "Always is explicit in the expert instruction.",
                },
                {
                  rule_id: "always-up",
                  field: "action.type",
                  value: "add",
                  source: "explicit_prompt",
                  rationale: "Increase is represented as an additive upward correction.",
                },
                {
                  rule_id: "always-up",
                  field: "action.value",
                  value: 100,
                  source: "llm_inferred",
                  rationale: "Infer a conservative amount from validation-front target scale.",
                },
                {
                  rule_id: "always-up",
                  field: "bounds.min_delta_original",
                  value: 0,
                  source: "explicit_prompt",
                  rationale: "The expert rule explicitly permits increases only.",
                },
              ],
            },
          },
          ctx,
        )

        const policyTool = await ForecastAdjustmentPolicyTool.init()
        await expect(
          policyTool.execute({ subcommand: "mark_no_effective", reason: "Premature stop." }, ctx),
        ).rejects.toThrow("adjustment_skill_priors remains unresolved")

        const skillTool = await ForecastAdjustmentSkillTool.init()
        await expect(
          skillTool.execute(
            { subcommand: "reject_candidate", candidate_id: "independent-user-prior", reason: "Premature reject." },
            ctx,
          ),
        ).rejects.toThrow("cannot be rejected before a held-out validation-back evaluation")
      },
    })

    await writeJson(
      path.join(
        tmp.path,
        ".forecast",
        "adjustments",
        "validation",
        "run-val",
        "independent-user-prior",
        "adjusted_eval.json",
      ),
      {
        run_id: "run-val",
        candidate_id: "independent-user-prior",
        split: "val",
        raw_eval_path: ".forecast/runs/run-val/eval.json",
        adjusted_predictions_path:
          ".forecast/adjustments/validation/run-val/independent-user-prior/adjusted_predictions.csv",
        raw_metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
        adjusted_metrics: { mse: 1.1, mae: 0.55, wape: 0.275, mase: null },
        delta_vs_raw: { mse: 0.1, mae: 0.05, wape: 0.025, mase: null },
        application_summary: {
          matched_rows: 4,
          changed_rows: 4,
          triggered_rule_count: 1,
          zero_trigger: false,
          row_count: 4,
          metrics_reused_from_raw: false,
        },
        acceptance_decision: {
          accepted: false,
          reason: "rejected: held-out validation MSE regressed",
          checks: { mse_non_regression: false },
        },
        timestamp: "2026-06-03T00:00:00.000Z",
      },
    )
    await writeFile(
      path.join(
        tmp.path,
        ".forecast",
        "adjustments",
        "validation",
        "run-val",
        "independent-user-prior",
        "leakage-audit.jsonl",
      ),
      `${JSON.stringify({ check: "blind_input_forbidden_keys", passed: true })}\n`,
      "utf-8",
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const skillTool = await ForecastAdjustmentSkillTool.init()
        const rejected = JSON.parse(
          (
            await skillTool.execute(
              {
                subcommand: "reject_candidate",
                candidate_id: "independent-user-prior",
                reason: "Held-out validation regressed.",
              },
              ctx,
            )
          ).output,
        )
        expect(rejected.rejected).toBe(true)

        const policyTool = await ForecastAdjustmentPolicyTool.init()
        const stopped = JSON.parse(
          (
            await policyTool.execute(
              { subcommand: "mark_no_effective", reason: "The independently validated expert prior regressed." },
              ctx,
            )
          ).output,
        )
        expect(stopped.marked).toBe(true)
      },
    })

    const candidate = JSON.parse(
      await readFile(
        path.join(
          tmp.path,
          ".forecast",
          "adjustments",
          "candidates",
          "independent-user-prior",
          "candidate.json",
        ),
        "utf-8",
      ),
    )
    expect(candidate.status).toBe("rejected")
    expect((await readState(tmp.path)).adjustment.no_effective_skill).toBe(true)
  })

  test("artifact-blocked expert prior can record no-effective before a source evidence chain exists", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeJson(path.join(tmp.path, ".forecast", "task.json"), {
      adjustment_skill_priors: "When a rare event occurs, increase the forecast by an unspecified amount.",
      target_col: "target",
      time_col: "timestamp",
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const policyTool = await ForecastAdjustmentPolicyTool.init()
        const result = JSON.parse(
          (
            await policyTool.execute(
              {
                subcommand: "mark_no_effective",
                reason: "Source model artifacts were unavailable, so validation-front evidence could not be prepared.",
              },
              ctx,
            )
          ).output,
        )
        expect(result.marked).toBe(true)
      },
    })

    const state = await readState(tmp.path)
    expect(state.adjustment.source_model_set_locked).toBe(false)
    expect(state.adjustment.fit_runs_ready).toBe(false)
    expect(state.adjustment.no_effective_skill).toBe(true)
  })

  test("user-prior mirror accepts partial provenance and undeclared LLM-inferred completions", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)
    const sourceExcerpt = "When temperature is below zero, increase the forecast."
    await writeJson(path.join(tmp.path, ".forecast", "task.json"), {
      adjustment_skill_priors: sourceExcerpt,
      target_col: "target",
      time_col: "timestamp",
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        const inspected = JSON.parse((await candidateTool.execute({ subcommand: "inspect_user_prior" }, ctx)).output)
        await candidateTool.execute(
          {
            subcommand: "write",
            candidate_id: "partial-provenance",
            source_model_set_id: "source-set",
            origin: "user_prior_mirror",
            rules: [
              {
                id: "cold-up",
                condition: { kind: "exogenous_numeric_between", column: "temperature", min: -50, max: 0 },
                action: { type: "add", value: 100 },
                rationale: "Keep the cold trigger and infer its missing upward magnitude.",
                evidence_refs: [".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"],
                bounds: {},
              },
            ],
            skill_markdown: "# Partial provenance\n",
            llm_provenance: { user_prior_mirror: true },
            user_prior_contract: {
              evidence_path: inspected.evidence_path,
              source_sha256: inspected.evidence.source_sha256,
              validation_front_match: "partial",
              rule_mappings: [{ rule_id: "cold-up", source_excerpt: sourceExcerpt }],
              unresolved_parameters_before_llm: [],
              parameter_completions: [
                {
                  rule_id: "cold-up",
                  field: "action.value",
                  value: 100,
                  source: "llm_inferred",
                  rationale: "Infer the unspecified upward amount.",
                },
              ],
            },
          },
          ctx,
        )
      },
    })

    expect(
      await Bun.file(
        path.join(tmp.path, ".forecast", "adjustments", "candidates", "partial-provenance", "candidate.json"),
      ).exists(),
    ).toBe(true)
  })

  test("user-prior mirror rejects incomplete numeric provenance and direction rewrites", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)
    const sourceExcerpt = "When temperature is below zero, increase the forecast."
    await writeJson(path.join(tmp.path, ".forecast", "task.json"), {
      adjustment_skill_priors: sourceExcerpt,
      target_col: "target",
      time_col: "timestamp",
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        const inspected = JSON.parse((await candidateTool.execute({ subcommand: "inspect_user_prior" }, ctx)).output)
        const contractBase = {
          evidence_path: inspected.evidence_path,
          source_sha256: inspected.evidence.source_sha256,
          validation_front_match: "partial" as const,
          rule_mappings: [{ rule_id: "cold-up", source_excerpt: sourceExcerpt }],
        }
        const rule = {
          id: "cold-up",
          condition: { kind: "exogenous_numeric_between" as const, column: "temperature", min: -50, max: 0 },
          action: { type: "add" as const, value: 100 },
          rationale: "Keep the cold trigger and fill only the missing upward magnitude.",
          evidence_refs: [".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"],
          bounds: {},
        }
        await expect(
          candidateTool.execute(
            {
              subcommand: "write",
              candidate_id: "missing-completion",
              source_model_set_id: "source-set",
              origin: "user_prior_mirror",
              rules: [rule],
              skill_markdown: "# Missing completion\n",
              llm_provenance: { user_prior_mirror: true },
              user_prior_contract: {
                ...contractBase,
                unresolved_parameters_before_llm: [{ rule_id: "cold-up", field: "action.value" }],
                parameter_completions: [
                  {
                    rule_id: "cold-up",
                    field: "condition.max",
                    value: 0,
                    source: "explicit_prompt",
                    rationale: "Below zero is explicit.",
                  },
                ],
              },
            },
            ctx,
          ),
        ).rejects.toThrow("must be completed by llm_inferred")

        await expect(
          candidateTool.execute(
            {
              subcommand: "write",
              candidate_id: "rewritten-direction",
              source_model_set_id: "source-set",
              origin: "user_prior_mirror",
              rules: [{ ...rule, action: { type: "add", value: -100 } }],
              skill_markdown: "# Rewritten direction\n",
              llm_provenance: { user_prior_mirror: true },
              user_prior_contract: {
                ...contractBase,
                unresolved_parameters_before_llm: [{ rule_id: "cold-up", field: "action.value" }],
                parameter_completions: [
                  {
                    rule_id: "cold-up",
                    field: "condition.kind",
                    value: "exogenous_numeric_between",
                    source: "explicit_prompt",
                    rationale: "The expert instruction defines a numeric temperature threshold.",
                  },
                  {
                    rule_id: "cold-up",
                    field: "condition.column",
                    value: "temperature",
                    source: "explicit_prompt",
                    rationale: "Temperature is explicitly named.",
                  },
                  {
                    rule_id: "cold-up",
                    field: "condition.min",
                    value: -50,
                    source: "validation_front_fit",
                    evidence_ref: ".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json",
                    rationale: "Use the observed lower envelope without broadening the zero threshold.",
                  },
                  {
                    rule_id: "cold-up",
                    field: "condition.max",
                    value: 0,
                    source: "explicit_prompt",
                    rationale: "Below zero is explicit.",
                  },
                  {
                    rule_id: "cold-up",
                    field: "action.type",
                    value: "add",
                    source: "explicit_prompt",
                    rationale: "Increase is represented as an additive correction.",
                  },
                  {
                    rule_id: "cold-up",
                    field: "action.value",
                    value: -100,
                    source: "llm_inferred",
                    rationale: "Invalid attempt to reverse the expert direction.",
                  },
                ],
              },
            },
            ctx,
          ),
        ).rejects.toThrow("action direction rewrites")

        await expect(
          candidateTool.execute(
            {
              subcommand: "write",
              candidate_id: "llm-rewrites-action-type",
              source_model_set_id: "source-set",
              origin: "user_prior_mirror",
              rules: [rule],
              skill_markdown: "# Invalid semantic completion\n",
              llm_provenance: { user_prior_mirror: true },
              user_prior_contract: {
                ...contractBase,
                unresolved_parameters_before_llm: [
                  { rule_id: "cold-up", field: "action.type" },
                  { rule_id: "cold-up", field: "action.value" },
                ],
                parameter_completions: [
                  {
                    rule_id: "cold-up",
                    field: "condition.kind",
                    value: "exogenous_numeric_between",
                    source: "explicit_prompt",
                    rationale: "The expert instruction defines a numeric temperature threshold.",
                  },
                  {
                    rule_id: "cold-up",
                    field: "condition.column",
                    value: "temperature",
                    source: "explicit_prompt",
                    rationale: "Temperature is explicitly named.",
                  },
                  {
                    rule_id: "cold-up",
                    field: "condition.min",
                    value: -50,
                    source: "validation_front_fit",
                    evidence_ref: ".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json",
                    rationale: "Use the observed lower envelope.",
                  },
                  {
                    rule_id: "cold-up",
                    field: "condition.max",
                    value: 0,
                    source: "explicit_prompt",
                    rationale: "Below zero is explicit.",
                  },
                  {
                    rule_id: "cold-up",
                    field: "action.type",
                    value: "add",
                    source: "llm_inferred",
                    rationale: "Invalid attempt to let the LLM choose a non-numeric semantic field.",
                  },
                  {
                    rule_id: "cold-up",
                    field: "action.value",
                    value: 100,
                    source: "llm_inferred",
                    rationale: "Infer only the missing numeric amount.",
                  },
                ],
              },
            },
            ctx,
          ),
        ).rejects.toThrow("LLM may fill only missing numeric values")
      },
    })
  })

  test("fit apply requires predictions from adjustment fit-run artifacts", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)
    await writeJson(path.join(tmp.path, ".forecast", "task.json"), {})
    await writeJson(path.join(tmp.path, ".forecast", "adjustments", "candidates", "candidate-test", "candidate.json"), {
      candidate_id: "candidate-test",
      source_model_set_id: "source-set",
      parent_skill_version: null,
      status: "candidate",
      rules: [
        {
          id: "r1",
          condition: { kind: "always" },
          action: { type: "multiply", value: 1.01 },
          rationale: "Small validation-front correction.",
          evidence_refs: ["fit"],
          bounds: {},
        },
      ],
      skill_markdown: "# Adjustment Skill\n",
      created_at: "2026-06-03T00:00:00.000Z",
      llm_provenance: {},
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const applyTool = await ForecastAdjustmentApplyTool.init()
        await expect(
          applyTool.execute({ run_id: "fit-run", candidate_id: "candidate-test", split: "fit" }, ctx),
        ).rejects.toThrow(".forecast/adjustments/fit-runs/runs/fit-run/predictions.csv")
      },
    })
  })

  test("fit diagnostics requires completed adjustment fit-run artifacts", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)
    await writeJson(path.join(tmp.path, ".forecast", "task.json"), {})
    await writeJson(path.join(tmp.path, ".forecast", "adjustments", "fit-runs", "manifest.json"), {
      source_model_set_id: "source-set",
      created_at: "2026-06-03T00:00:00.000Z",
      updated_at: "2026-06-03T00:00:00.000Z",
      status: "complete",
      members: [
        {
          family: "deep_learning",
          validation_run_id: "run-val",
          model: "DLinear",
          validation_spec_hash: "validation-spec-hash",
          fit_spec_hash: "fit-run",
          fit_run_id: "fit-run",
          status: "success",
          fit_method: "runner_reuse_validation_front_half",
          spec_path: ".forecast/adjustments/fit-runs/specs/deep_learning-run-val.json",
          fit_run_path: ".forecast/adjustments/fit-runs/runs/fit-run",
          predictions_path: ".forecast/adjustments/fit-runs/runs/fit-run/predictions.csv",
          eval_path: ".forecast/adjustments/fit-runs/runs/fit-run/eval.json",
          metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
          error: null,
        },
      ],
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const diagnosticsTool = await ForecastAdjustmentDiagnosticsTool.init()
        await expect(diagnosticsTool.execute({ run_id: "fit-run", split: "fit", top_k: 20 }, ctx)).rejects.toThrow(
          ".forecast/adjustments/fit-runs/runs/fit-run/predictions.csv",
        )
      },
    })
  })

  test("test apply requires locked active skill before Python execution", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeJson(path.join(tmp.path, ".forecast", "task.json"), {})
    await mkdir(path.join(tmp.path, ".forecast", "runs", "run-test"), { recursive: true })
    await writeFile(
      path.join(tmp.path, ".forecast", "runs", "run-test", "predictions.csv"),
      "timestamp,predicted\nx,0\n",
    )
    await writeJson(
      path.join(tmp.path, ".forecast", "adjustments", "candidates", "candidate-test", "candidate.json"),
      {},
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const applyTool = await ForecastAdjustmentApplyTool.init()
        await expect(
          applyTool.execute({ run_id: "run-test", candidate_id: "candidate-test", split: "test" }, ctx),
        ).rejects.toThrow("locked active adjustment skill")
      },
    })
  })

  test("test apply requires an explicit apply decision for test", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeLockedActiveSkill(tmp.path)
    await writeJson(path.join(tmp.path, ".forecast", "task.json"), {})
    await mkdir(path.join(tmp.path, ".forecast", "runs", "run-test"), { recursive: true })
    await writeFile(
      path.join(tmp.path, ".forecast", "runs", "run-test", "predictions.csv"),
      "timestamp,predicted\nx,0\n",
    )
    await writeJson(
      path.join(tmp.path, ".forecast", "adjustments", "candidates", "candidate-test", "candidate.json"),
      {},
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const applyTool = await ForecastAdjustmentApplyTool.init()
        await expect(
          applyTool.execute({ run_id: "run-test", candidate_id: "candidate-test", split: "test" }, ctx),
        ).rejects.toThrow("apply decision for this eligible test run")
      },
    })
  })

  test("test apply requires the raw test run recorded by the adjustment decision", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeLockedActiveSkill(tmp.path)
    await writeRawTestRun(tmp.path, "run-confirmed", 0.75)
    await writeRawTestRun(tmp.path, "run-other", 0.5)
    await writeJson(path.join(tmp.path, ".forecast", "task.json"), {})
    await writeJson(
      path.join(tmp.path, ".forecast", "adjustments", "candidates", "candidate-test", "candidate.json"),
      {},
    )

    const state = await readState(tmp.path)
    await writeStateFrontmatter(tmp.path, {
      ...state,
      adjustment: {
        ...state.adjustment,
        critic_user_confirmed: "2026-06-03T00:00:00.000Z",
        test_application_decision: "apply",
        test_application_decision_at: "2026-06-03T00:00:00.000Z",
        test_application_run_id: "run-confirmed",
      },
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const applyTool = await ForecastAdjustmentApplyTool.init()
        await expect(
          applyTool.execute({ run_id: "run-other", candidate_id: "candidate-test", split: "test" }, ctx),
        ).rejects.toThrow("apply decision for this eligible test run")
      },
    })
  })

  test("test apply requires the locked skill to be accepted on the confirmed raw test spec", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeLockedActiveSkill(tmp.path)
    await writeRawTestRun(tmp.path, "run-confirmed", 0.75, {
      model: "TiDE",
      phase: "post-forecast",
      eval_split: "test",
    })
    await writeJson(path.join(tmp.path, ".forecast", "task.json"), {})
    await writeJson(
      path.join(tmp.path, ".forecast", "adjustments", "candidates", "candidate-test", "candidate.json"),
      {},
    )

    const state = await readState(tmp.path)
    await writeStateFrontmatter(tmp.path, {
      ...state,
      adjustment: {
        ...state.adjustment,
        critic_user_confirmed: "2026-06-03T00:00:00.000Z",
        test_application_decision: "apply",
        test_application_decision_at: "2026-06-03T00:00:00.000Z",
        test_application_run_id: "run-confirmed",
      },
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const applyTool = await ForecastAdjustmentApplyTool.init()
        await expect(
          applyTool.execute({ run_id: "run-confirmed", candidate_id: "candidate-test", split: "test" }, ctx),
        ).rejects.toThrow("not validation-accepted on this test model spec")
      },
    })
  })

  test("skill accept_candidate records accepted zero-trigger and improved validation results", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)
    await writeJson(path.join(tmp.path, ".forecast", "adjustments", "source-model-set.json"), {
      source_model_set_id: "source-set",
      created_at: "2026-06-03T00:00:00.000Z",
      created_from: "post_raw_forecast_search",
      members: [
        {
          family: "deep_learning",
          validation_run_id: "run-val",
          model: "DLinear",
          spec_hash: "validation-spec-hash",
          validation_metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
        },
        {
          family: "statistical",
          validation_run_id: "run-alt",
          model: "ARIMA",
          spec_hash: "validation-spec-hash-alt",
          validation_metrics: { mse: 1.2, mae: 0.6, wape: 0.3, mase: null },
        },
      ],
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        await candidateTool.execute(
          {
            subcommand: "write",
            candidate_id: "candidate-test",
            source_model_set_id: "source-set",
            rules: [
              {
                id: "r1",
                condition: { kind: "always" },
                action: { type: "multiply", value: 1.01 },
                rationale: "Small systematic correction supported by validation-front evidence.",
                evidence_refs: [".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"],
                bounds: {},
              },
            ],
            skill_markdown: "# Adjustment Skill\n\nApply r1.",
          },
          ctx,
        )
      },
    })

    for (const runId of ["run-val", "run-alt"]) {
      const zeroTrigger = runId === "run-val"
      await writeJson(
        path.join(tmp.path, ".forecast", "adjustments", "validation", runId, "candidate-test", "adjusted_eval.json"),
        {
          run_id: runId,
          candidate_id: "candidate-test",
          split: "val",
          raw_eval_path: `.forecast/runs/${runId}/eval.json`,
          adjusted_predictions_path: `.forecast/adjustments/validation/${runId}/candidate-test/adjusted_predictions.csv`,
          raw_metrics: { mse: 2, mae: 1, wape: 0.5, mase: null },
          adjusted_metrics: zeroTrigger
            ? { mse: 2, mae: 1, wape: 0.5, mase: null }
            : { mse: 1, mae: 0.8, wape: 0.4, mase: null },
          delta_vs_raw: zeroTrigger
            ? { mse: 0, mae: 0, wape: 0, mase: null }
            : { mse: -1, mae: -0.2, wape: -0.1, mase: null },
          application_summary: {
            matched_rows: zeroTrigger ? 0 : 4,
            changed_rows: zeroTrigger ? 0 : 4,
            triggered_rule_count: zeroTrigger ? 0 : 1,
            zero_trigger: zeroTrigger,
            row_count: 4,
            metrics_reused_from_raw: zeroTrigger,
          },
          acceptance_decision: {
            accepted: true,
            reason: zeroTrigger
              ? "accepted: candidate did not trigger and preserved the baseline exactly"
              : "accepted",
            checks: { zero_trigger: zeroTrigger },
          },
          timestamp: "2026-06-03T00:00:00.000Z",
        },
      )
      await writeFile(
        path.join(tmp.path, ".forecast", "adjustments", "validation", runId, "candidate-test", "leakage-audit.jsonl"),
        `${JSON.stringify({ check: "blind_input_forbidden_keys", passed: true })}\n`,
        "utf-8",
      )
    }

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const skillTool = await ForecastAdjustmentSkillTool.init()
        await skillTool.execute(
          { subcommand: "accept_candidate", candidate_id: "candidate-test", run_id: "run-val" },
          ctx,
        )
      },
    })

    const manifest = JSON.parse(
      await readFile(path.join(tmp.path, ".forecast", "adjustments", "skills", "active", "manifest.json"), "utf-8"),
    )
    expect(manifest.validation_results.sort()).toEqual([
      ".forecast/adjustments/validation/run-alt/candidate-test/adjusted_eval.json",
      ".forecast/adjustments/validation/run-val/candidate-test/adjusted_eval.json",
    ])
  })

  test("candidate write and accept create active version", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path)
    await writeSourceSet(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const candidateTool = await ForecastAdjustmentCandidateTool.init()
        await candidateTool.execute(
          {
            subcommand: "write",
            candidate_id: "candidate-test",
            source_model_set_id: "source-set",
            rules: [
              {
                id: "r1",
                condition: { kind: "always" },
                action: { type: "multiply", value: 1.01 },
                rationale: "Small systematic underprediction correction.",
                evidence_refs: [".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"],
                bounds: { max_delta_ratio: 0.05 },
              },
            ],
            skill_markdown: "# Adjustment Skill\n\n## Rules\n\nApply r1.",
          },
          ctx,
        )
      },
    })

    await writeJson(
      path.join(tmp.path, ".forecast", "adjustments", "validation", "run-val", "candidate-test", "adjusted_eval.json"),
      {
        run_id: "run-val",
        candidate_id: "candidate-test",
        split: "val",
        raw_eval_path: ".forecast/runs/run-val/eval.json",
        adjusted_predictions_path: ".forecast/adjustments/validation/run-val/candidate-test/adjusted_predictions.csv",
        raw_metrics: { mse: 2, mae: 1, wape: 0.5, mase: null },
        adjusted_metrics: { mse: 1, mae: 0.8, wape: 0.4, mase: null },
        delta_vs_raw: { mse: -1, mae: -0.2, wape: -0.1, mase: null },
        acceptance_decision: { accepted: true, reason: "accepted", checks: {} },
        timestamp: "2026-06-03T00:00:00.000Z",
      },
    )
    await writeFile(
      path.join(tmp.path, ".forecast", "adjustments", "validation", "run-val", "candidate-test", "leakage-audit.jsonl"),
      `${JSON.stringify({ check: "blind_input_forbidden_keys", passed: true })}\n`,
      "utf-8",
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const skillTool = await ForecastAdjustmentSkillTool.init()
        await skillTool.execute(
          { subcommand: "accept_candidate", candidate_id: "candidate-test", run_id: "run-val" },
          ctx,
        )
      },
    })

    const manifest = JSON.parse(
      await readFile(path.join(tmp.path, ".forecast", "adjustments", "skills", "active", "manifest.json"), "utf-8"),
    )
    const state = ForecastPhaseState.parse(
      normalizeForecastStateFrontmatter(
        matter(await readFile(path.join(tmp.path, ".forecast", "STATE.md"), "utf-8")).data as Record<string, unknown>,
      ),
    )

    expect(manifest.version).toBe("v001")
    expect(manifest.candidate_id).toBe("candidate-test")
    expect(state.adjustment.active_skill_version).toBe("v001")
  })
})
