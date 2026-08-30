import { afterEach, describe, expect, test } from "bun:test"
import { mkdir, readFile, writeFile } from "fs/promises"
import { createHash } from "node:crypto"
import path from "path"
import matter from "gray-matter"
import { Instance } from "@/project/instance"
import { MessageID, SessionID } from "@/session/schema"
import { tmpdir } from "../../../test/fixture/fixture"
import { BestModelExportTool } from "./best-model-export"
import { computeSpecHash } from "./hash"
import { foundationModelReference, statisticalModelArtifact } from "./model-family"
import { ForecastPhaseState } from "./types"

const ctx = {
  sessionID: SessionID.make("ses_best-model-export-test"),
  messageID: MessageID.make("msg_best-model-export-test"),
  callID: "best-model-export-test-call",
  agent: "best-model-export-test-agent",
  abort: AbortSignal.any([]),
  messages: [],
  metadata: () => {},
  ask: async () => {},
}

const task = {
  dataset_path: "/tmp/dataset.csv",
  target_col: "value",
  time_col: "timestamp",
  pred_len: 24,
  seq_len: 48,
  label_len: 24,
  train_ratio: 0.7,
  val_ratio: 0.2,
  test_ratio: 0.1,
  eval_metrics: ["mse", "mae", "wape", "mase"],
  model_families: ["DLinear", "ARIMA"],
  frozen_at: "2026-04-30T00:00:00.000Z",
}
const FOUNDATION_REVISION = "c".repeat(40)

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex")
}

type RunFixture = {
  runId: string
  model?: string
  evalSplit: "val" | "test"
  mse: number
  checkpoint?: boolean
  arimaResults?: boolean
  statisticalResults?: boolean
  reuseSourceRunId?: string
  missing?: string[]
}

async function writeJson(file: string, value: unknown) {
  await mkdir(path.dirname(file), { recursive: true })
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`, "utf-8")
}

async function writeText(file: string, value: string) {
  await mkdir(path.dirname(file), { recursive: true })
  await writeFile(file, value, "utf-8")
}

async function writeForecastTask(directory: string) {
  await writeJson(path.join(directory, ".forecast", "task.json"), task)
}

async function writeLockedAdjustmentState(
  directory: string,
  runId: string,
  candidateId: string,
  includeReflection = false,
) {
  const state = ForecastPhaseState.parse({
    phase: "post-forecast",
    skills_confirmed: true,
    skills_confirmed_at: "2026-04-30T00:00:00.000Z",
    pre_forecast_completed: true,
    pre_forecast_completed_at: "2026-04-30T00:00:00.000Z",
    task_frozen_at: "2026-04-30T00:00:00.000Z",
    transitions: [],
    adjustment: {
      enabled: true,
      mode: "rule_skill",
      policy_path: ".forecast/adjustments/policy.json",
      source_model_set_locked: true,
      source_model_set_path: ".forecast/adjustments/source-model-set.json",
      fit_run_manifest_path: ".forecast/adjustments/fit-runs/manifest.json",
      fit_runs_ready: true,
      active_skill_version: "v001",
      active_skill_path: ".forecast/adjustments/skills/active/SKILL.md",
      forecaster_locked: true,
      no_effective_skill: false,
      no_effective_skill_reason: null,
      test_collection_run_ids: [runId],
      test_eligible_run_ids: [runId],
      test_application_records: [
        {
          test_run_id: runId,
          validation_run_id: "validation-run",
          model: "DLinear",
          decision: "apply",
          decision_at: "2026-04-30T00:00:02.000Z",
          decision_actor: "auto_policy",
          decision_reason: "Auto Mode defaulted to apply for this exact validation-matched model.",
          adjustment_locked: true,
          adjustment_candidate_id: candidateId,
          adjustment_evaluated: true,
          adjusted_eval_path: `.forecast/adjustments/test/${runId}/${candidateId}/adjusted_eval.json`,
        },
      ],
      critic_user_confirmed: "2026-04-30T00:00:02.000Z",
      test_application_decision: "apply",
      test_application_decision_at: "2026-04-30T00:00:02.000Z",
      test_application_decision_actor: "auto_policy",
      test_application_decision_reason: "Auto Mode defaulted to apply for this exact validation-matched model.",
      test_application_run_id: runId,
      test_adjustment_locked: true,
      test_adjustment_run_id: runId,
      test_adjustment_candidate_id: candidateId,
    },
    reflection: includeReflection
      ? {
          enabled: true,
          configured: true,
          policy_path: ".forecast/reflection/policy.json",
          controller_version: "1",
          max_reflection_rounds: 3,
          max_format_retries: 2,
          validation_records: [],
          test_records: [
            {
              test_run_id: runId,
              validation_run_id: "validation-run",
              baseline_kind: "skill_adjusted",
              adjustment_candidate_id: candidateId,
              validation_status: "enabled_improved",
              llm_refinement_enabled: true,
              trigger_count: 2,
              skill_match_count: 2,
              llm_required_count: 2,
              corrected_count: 1,
              fallback_count: 1,
              deterministic_fallback_count: 1,
              raw_fallback_count: 0,
              baseline_metrics: { mse: 0.8, mae: 0.4, wape: 0.2, mase: null },
              final_metrics: { mse: 0.7, mae: 0.35, wape: 0.18, mase: null },
              result_path: `.forecast/reflection/test/${runId}/${candidateId}/reflection_eval.json`,
              completed_at: "2026-04-30T00:00:03.000Z",
            },
          ],
        }
      : undefined,
  })
  await writeText(path.join(directory, ".forecast", "STATE.md"), matter.stringify("\n# Forecast State Log\n", state))
}

async function writeReflectionArtifacts(directory: string, runId: string, candidateId: string) {
  const root = path.join(directory, ".forecast", "reflection", "test", runId, candidateId)
  const predictionsPath = `.forecast/reflection/test/${runId}/${candidateId}/final_predictions.csv`
  const predictionsContents =
    "window_index,horizon_index,timestamp,actual,predicted,baseline_predicted\n0,0,2026-01-01,1,1.02,1.05\n"
  const audit = {
    passed: true,
    forbidden_context_keys: [],
  }
  const auditContents = `${JSON.stringify(audit, null, 2)}\n`
  const protocolContents = `${JSON.stringify({ protocol_version: 1, mode: "skill_guided_test" }, null, 2)}\n`
  await writeText(path.join(root, "final_predictions.csv"), predictionsContents)
  await writeText(path.join(root, "leakage-audit.json"), auditContents)
  await writeText(path.join(root, "llm-protocol.json"), protocolContents)
  await writeJson(path.join(root, "reflection_eval.json"), {
    controller_version: "1",
    source_run_id: runId,
    split: "test",
    baseline_kind: "skill_adjusted",
    adjustment_candidate_id: candidateId,
    status: "completed",
    baseline_metrics: { mse: 0.8, mae: 0.4, wape: 0.2, mase: null },
    final_metrics: { mse: 0.7, mae: 0.35, wape: 0.18, mase: null },
    delta_mse: -0.1,
    trigger_count: 2,
    skill_match_count: 2,
    llm_required_count: 2,
    corrected_count: 1,
    fallback_count: 1,
    deterministic_fallback_count: 1,
    raw_fallback_count: 0,
    final_predictions_path: predictionsPath,
    final_predictions_sha256: sha256(predictionsContents),
    leakage_audit_sha256: sha256(auditContents),
    llm_protocol_path: `.forecast/reflection/test/${runId}/${candidateId}/llm-protocol.json`,
    llm_protocol_sha256: sha256(protocolContents),
    completed_at: "2026-04-30T00:00:03.000Z",
  })
}

async function writeRun(directory: string, input: RunFixture) {
  const runDir = path.join(directory, ".forecast", "runs", input.runId)
  const model = input.model ?? "DLinear"
  const spec = {
    model,
    dataset_path: task.dataset_path,
    target_col: task.target_col,
    time_col: task.time_col,
    pred_len: task.pred_len,
    seq_len: task.seq_len,
    label_len: task.label_len,
    train_ratio: task.train_ratio,
    val_ratio: task.val_ratio,
    test_ratio: task.test_ratio,
    metrics: task.eval_metrics,
    phase: input.evalSplit === "test" ? "post-forecast" : "forecast",
    eval_split: input.evalSplit,
  }
  const specHash = computeSpecHash(spec)
  const missing = new Set(input.missing ?? [])
  const statisticalArtifact = statisticalModelArtifact(model)

  if (!missing.has("spec.json")) {
    await writeJson(path.join(runDir, "spec.json"), spec)
  }
  const shouldWriteCheckpoint =
    input.checkpoint ?? (input.evalSplit === "val" && !statisticalArtifact && !foundationModelReference(model))
  const shouldWriteArimaResults =
    input.arimaResults ?? (input.evalSplit === "val" && statisticalArtifact?.kind === "statsmodels_arima_results")
  const shouldWriteStatisticalResults =
    input.statisticalResults ??
    (input.evalSplit === "val" && statisticalArtifact?.kind === "statsmodels_statistical_results")
  if (shouldWriteCheckpoint) {
    const checkpoint = "checkpoint"
    await writeText(path.join(runDir, "model", "checkpoint.pth"), checkpoint)
    await writeJson(path.join(runDir, "model", "metadata.json"), {
      kind: "checkpoint",
      path: "model/checkpoint.pth",
      sha256: sha256(checkpoint),
      model,
      eval_split: "val",
      phase: "forecast",
      trained_on: "train",
      format: "torch_state_dict",
    })
  }
  if (shouldWriteArimaResults) {
    const results = "arima-results"
    await writeText(path.join(runDir, "model", "arima-results.pkl"), results)
    await writeJson(path.join(runDir, "model", "metadata.json"), {
      kind: "arima_results",
      path: "model/arima-results.pkl",
      sha256: sha256(results),
      model,
      eval_split: "val",
      phase: "forecast",
      trained_on: "train",
      format: "statsmodels_arima_results",
    })
  }
  if (shouldWriteStatisticalResults) {
    const results = "statistical-results"
    await writeText(path.join(runDir, "model", "statistical-results.pkl"), results)
    await writeJson(path.join(runDir, "model", "metadata.json"), {
      kind: "statistical_results",
      path: "model/statistical-results.pkl",
      sha256: sha256(results),
      model,
      eval_split: "val",
      phase: "forecast",
      trained_on: "train",
      format: "statsmodels_results_pickle",
    })
  }
  const foundationReference = foundationModelReference(model)
  if (input.evalSplit === "val" && foundationReference) {
    await writeJson(path.join(runDir, "model", "foundation.json"), {
      kind: "foundation_model_reference",
      model,
      repo_id: foundationReference,
      revision: FOUNDATION_REVISION,
      sha256: computeSpecHash({
        repo_id: foundationReference,
        revision: FOUNDATION_REVISION,
      }),
      format: "huggingface_snapshot",
    })
  }

  const predictionFiles: Record<string, string> = {
    "predictions.csv": "timestamp,actual,predicted\n2026-01-01,1,1.1\n",
    "actual.npy": "actual",
    "pred.npy": "pred",
    "train_scale.json": JSON.stringify({ train_scale: 1 }, null, 2),
    "train.log": "runner log",
  }
  for (const [file, content] of Object.entries(predictionFiles)) {
    if (!missing.has(file)) {
      await writeText(path.join(runDir, file), content)
    }
  }

  if (input.evalSplit === "test") {
    const requiredFiles: Record<string, string> = {
      "model/reuse.json": JSON.stringify({
        source_run_id: input.reuseSourceRunId ?? "validation-run",
        source_artifact_kind: statisticalArtifact
          ? statisticalArtifact.kind
          : foundationModelReference(model)
            ? "foundation_model_reference"
            : "torch_state_dict",
        source_artifact_sha256:
          statisticalArtifact?.kind === "statsmodels_arima_results"
            ? sha256("arima-results")
            : statisticalArtifact?.kind === "statsmodels_statistical_results"
              ? sha256("statistical-results")
              : foundationModelReference(model)
                ? computeSpecHash({
                    repo_id: foundationModelReference(model),
                    revision: FOUNDATION_REVISION,
                  })
                : sha256("checkpoint"),
        foundation_revision: foundationModelReference(model) ? FOUNDATION_REVISION : null,
        training_performed: false,
        evaluated_split: "test",
      }),
    }

    for (const [file, content] of Object.entries(requiredFiles)) {
      if (!missing.has(file)) {
        await writeText(path.join(runDir, file), content)
      }
    }
  }

  if (!missing.has("eval.json")) {
    const artifactKind = statisticalArtifact
      ? statisticalArtifact.kind
      : foundationReference
        ? "foundation_model_reference"
        : "torch_state_dict"
    const artifactSha256 =
      artifactKind === "statsmodels_arima_results"
        ? sha256("arima-results")
        : artifactKind === "statsmodels_statistical_results"
          ? sha256("statistical-results")
          : artifactKind === "foundation_model_reference"
            ? computeSpecHash({
                repo_id: foundationReference,
                revision: FOUNDATION_REVISION,
              })
            : sha256("checkpoint")
    await writeJson(path.join(runDir, "eval.json"), {
      run_id: input.runId,
      evaluator_hash: "e".repeat(64),
      spec_hash: specHash,
      metrics: {
        mse: input.mse,
        mae: input.mse / 2,
        wape: input.mse / 3,
        mase: input.mse / 4,
      },
      eval_split: input.evalSplit,
      timestamp: "2026-04-30T00:00:00.000Z",
      artifact_binding: {
        model_artifact: {
          kind: artifactKind,
          sha256: artifactSha256,
          source_run_id: input.evalSplit === "test" ? (input.reuseSourceRunId ?? "validation-run") : input.runId,
          foundation_revision: foundationReference ? FOUNDATION_REVISION : null,
        },
        prediction_artifacts: {
          predictions_sha256: sha256(predictionFiles["predictions.csv"]),
          actual_sha256: sha256(predictionFiles["actual.npy"]),
          pred_sha256: sha256(predictionFiles["pred.npy"]),
          train_scale_sha256: sha256(predictionFiles["train_scale.json"]),
        },
      },
    })
  }
}

async function writeUserPriorRun(directory: string, input: { runId: string; evalSplit: "val" | "test"; mse: number }) {
  const runDir = path.join(directory, ".forecast", "runs", input.runId)
  const spec = {
    model: "UserPrior",
    dataset_path: task.dataset_path,
    target_col: task.target_col,
    time_col: task.time_col,
    pred_len: task.pred_len,
    seq_len: task.seq_len,
    label_len: task.label_len,
    train_ratio: task.train_ratio,
    val_ratio: task.val_ratio,
    test_ratio: task.test_ratio,
    metrics: task.eval_metrics,
    phase: input.evalSplit === "test" ? "post-forecast" : "forecast",
    eval_split: input.evalSplit,
    input_contract_version: 4,
    source_type: "user_prediction_prior",
  }
  const specHash = computeSpecHash(spec)
  const priorSource = JSON.stringify(
    {
      kind: "external_prediction_prior",
      model: "UserPrior",
      format: "adacast_user_prediction_prior",
      source_dataset_sha256: "f".repeat(64),
    },
    null,
    2,
  )
  const predictionFiles: Record<string, string> = {
    "predictions.csv": "timestamp,actual,predicted\n2026-01-01,1,1.1\n",
    "actual.npy": "actual",
    "pred.npy": "pred",
    "train_scale.json": JSON.stringify({ train_scale: 1 }, null, 2),
    "train.log": "user prior log",
  }
  await writeJson(path.join(runDir, "spec.json"), spec)
  for (const [file, content] of Object.entries(predictionFiles)) {
    await writeText(path.join(runDir, file), content)
  }
  if (input.evalSplit === "val") {
    await writeText(path.join(runDir, "model", "prior-source.json"), priorSource)
  } else {
    await writeText(
      path.join(runDir, "model", "reuse.json"),
      JSON.stringify({
        source_run_id: "validation-run",
        source_artifact_kind: "external_prediction_prior",
        source_artifact_sha256: sha256(priorSource),
        foundation_revision: null,
        training_performed: false,
        evaluated_split: "test",
      }),
    )
  }
  await writeJson(path.join(runDir, "eval.json"), {
    run_id: input.runId,
    evaluator_hash: "e".repeat(64),
    spec_hash: specHash,
    metrics: {
      mse: input.mse,
      mae: input.mse / 2,
      wape: input.mse / 3,
      mase: input.mse / 4,
    },
    eval_split: input.evalSplit,
    timestamp: "2026-04-30T00:00:00.000Z",
    artifact_binding: {
      model_artifact: {
        kind: "external_prediction_prior",
        sha256: sha256(priorSource),
        source_run_id: input.evalSplit === "test" ? "validation-run" : input.runId,
        foundation_revision: null,
      },
      prediction_artifacts: {
        predictions_sha256: sha256(predictionFiles["predictions.csv"]),
        actual_sha256: sha256(predictionFiles["actual.npy"]),
        pred_sha256: sha256(predictionFiles["pred.npy"]),
        train_scale_sha256: sha256(predictionFiles["train_scale.json"]),
      },
    },
  })
}

async function exportBestModel(
  directory: string,
  input: {
    validation_run_id: string
    test_run_id: string
    model_family?: string
    selection_reason?: string
  },
) {
  return Instance.provide({
    directory,
    fn: async () => {
      const tool = await BestModelExportTool.init()
      const result = await tool.execute(input, ctx)
      return { result, output: JSON.parse(result.output) as Record<string, unknown> }
    },
  })
}

afterEach(async () => {
  await Instance.disposeAll()
})

describe("export_best_model", () => {
  test("exports a complete checkpoint package", async () => {
    await using tmp = await tmpdir()

    await writeForecastTask(tmp.path)
    await writeRun(tmp.path, { runId: "validation-run", evalSplit: "val", mse: 2 })
    await writeRun(tmp.path, { runId: "test-run", evalSplit: "test", mse: 1 })

    const { result } = await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      model_family: "deep_learning",
      selection_reason: "lowest test mse",
    })

    const exportRoot = path.join(tmp.path, ".forecast", "best-model")
    const manifest = JSON.parse(await readFile(path.join(exportRoot, "manifest.json"), "utf-8"))

    expect(result.metadata.status).toBe("success")
    expect(await readFile(path.join(exportRoot, "USAGE.md"), "utf-8")).toContain("AdaCast Best Model Package")
    expect(await readFile(path.join(exportRoot, "metadata", "spec.json"), "utf-8")).toContain("DLinear")
    expect(await readFile(path.join(exportRoot, "metadata", "eval.json"), "utf-8")).toContain("test-run")
    expect(await readFile(path.join(exportRoot, "metadata", "train_scale.json"), "utf-8")).toContain("train_scale")
    expect(await readFile(path.join(exportRoot, "metadata", "model-reuse.json"), "utf-8")).toContain(
      "training_performed",
    )
    expect(await readFile(path.join(exportRoot, "predictions", "predictions.csv"), "utf-8")).toContain("predicted")
    expect(await readFile(path.join(exportRoot, "predictions", "actual.npy"), "utf-8")).toBe("actual")
    expect(await readFile(path.join(exportRoot, "predictions", "pred.npy"), "utf-8")).toBe("pred")
    expect(await readFile(path.join(exportRoot, "logs", "train.log"), "utf-8")).toBe("runner log")
    expect(await readFile(path.join(exportRoot, "model", "checkpoint.pth"), "utf-8")).toBe("checkpoint")
    expect(manifest).toMatchObject({
      model: "DLinear",
      model_family: "deep_learning",
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      selection_reason: "lowest test mse",
      source_paths: {
        validation_run: ".forecast/runs/validation-run",
        test_run: ".forecast/runs/test-run",
      },
    })
    expect(manifest.validation_metrics.mse).toBe(2)
    expect(manifest.test_metrics.mse).toBe(1)
    expect(manifest.task.target_col).toBe("value")
    expect(manifest.artifacts).toEqual([
      {
        kind: "checkpoint",
        path: "model/checkpoint.pth",
        format: "torch_state_dict",
        sha256: sha256("checkpoint"),
        model: "DLinear",
        model_family: "deep_learning",
      },
    ])
  })

  test("exports optional adjusted prediction artifacts when present", async () => {
    await using tmp = await tmpdir()

    await writeForecastTask(tmp.path)
    await writeRun(tmp.path, { runId: "validation-run", evalSplit: "val", mse: 2 })
    await writeRun(tmp.path, { runId: "test-run", evalSplit: "test", mse: 1 })
    await writeText(
      path.join(tmp.path, ".forecast", "adjustments", "test", "test-run", "candidate-test", "adjusted_predictions.csv"),
      "timestamp,actual,predicted,raw_predicted\n2026-01-01,1,1.05,1.1\n",
    )
    await writeJson(
      path.join(tmp.path, ".forecast", "adjustments", "test", "test-run", "candidate-test", "adjusted_eval.json"),
      {
        run_id: "test-run",
        candidate_id: "candidate-test",
        split: "test",
        raw_eval_path: ".forecast/runs/test-run/eval.json",
        adjusted_predictions_path: ".forecast/adjustments/test/test-run/candidate-test/adjusted_predictions.csv",
        raw_metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: 0.125 },
        adjusted_metrics: { mse: 0.8, mae: 0.4, wape: 0.2, mase: null },
        delta_vs_raw: { mse: -0.2, mae: -0.1, wape: -0.05, mase: null },
        timestamp: "2026-04-30T00:00:00.000Z",
      },
    )
    await writeText(
      path.join(tmp.path, ".forecast", "adjustments", "test", "test-run", "candidate-test", "leakage-audit.jsonl"),
      `${JSON.stringify({ check: "blind_input_forbidden_keys", passed: true })}\n`,
    )
    await writeJson(path.join(tmp.path, ".forecast", "adjustments", "skills", "active", "manifest.json"), {
      version: "v001",
      status: "locked",
      candidate_id: "candidate-test",
      source_model_set_id: "source-set",
      accepted_at: "2026-04-30T00:00:00.000Z",
      locked_at: "2026-04-30T00:00:01.000Z",
      validation_results: [".forecast/adjustments/validation/validation-run/candidate-test/adjusted_eval.json"],
      policy_path: ".forecast/adjustments/policy.json",
      parent_version: null,
    })
    await writeText(path.join(tmp.path, ".forecast", "adjustments", "skills", "active", "SKILL.md"), "# Skill\n")
    await writeReflectionArtifacts(tmp.path, "test-run", "candidate-test")
    await writeLockedAdjustmentState(tmp.path, "test-run", "candidate-test", true)

    const { output } = await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      model_family: "deep_learning",
    })

    const exportRoot = path.join(tmp.path, ".forecast", "best-model")
    const manifest = JSON.parse(await readFile(path.join(exportRoot, "manifest.json"), "utf-8"))
    expect(manifest.prediction_outputs.adjusted).toMatchObject({
      enabled: true,
      predictions: "adjusted/adjusted_predictions.csv",
      skill_version: "v001",
      leakage_audit_passed: true,
      confirmed_before_test_metrics: true,
      decision_actor: "auto_policy",
      decision_reason: "Auto Mode defaulted to apply for this exact validation-matched model.",
    })
    expect(manifest.prediction_outputs.window_reflection).toMatchObject({
      enabled: true,
      predictions: "reflection/final_predictions.csv",
      metrics: { mse: 0.7, mae: 0.35, wape: 0.18, mase: null },
      baseline_metrics: { mse: 0.8, mae: 0.4, wape: 0.2, mase: null },
      delta_mse: -0.1,
      baseline_kind: "skill_adjusted",
      validation_status: "enabled_improved",
      llm_refinement_enabled: true,
      trigger_count: 2,
      skill_match_count: 2,
      llm_required_count: 2,
      corrected_count: 1,
      fallback_count: 1,
      deterministic_fallback_count: 1,
      raw_fallback_count: 0,
      llm_protocol: "reflection/llm-protocol.json",
      leakage_audit_passed: true,
    })
    expect(output.prediction_outputs).toMatchObject(manifest.prediction_outputs)
    expect(await readFile(path.join(exportRoot, "adjusted", "adjusted_predictions.csv"), "utf-8")).toContain(
      "raw_predicted",
    )
    expect(await readFile(path.join(exportRoot, "adjusted", "skill", "SKILL.md"), "utf-8")).toContain("Skill")
    expect(await readFile(path.join(exportRoot, "reflection", "final_predictions.csv"), "utf-8")).toContain(
      "baseline_predicted",
    )
    expect(await readFile(path.join(exportRoot, "reflection", "leakage-audit.json"), "utf-8")).toContain(
      '"passed": true',
    )
    expect(await readFile(path.join(exportRoot, "reflection", "llm-protocol.json"), "utf-8")).toContain(
      '"skill_guided_test"',
    )
    expect(await readFile(path.join(exportRoot, "USAGE.md"), "utf-8")).toContain("adjusted/adjusted_predictions.csv")
    expect(await readFile(path.join(exportRoot, "USAGE.md"), "utf-8")).toContain("reflection/final_predictions.csv")

    const reflectionRoot = path.join(tmp.path, ".forecast", "reflection", "test", "test-run", "candidate-test")
    const reflectionEvalPath = path.join(reflectionRoot, "reflection_eval.json")
    const failedAudit = { passed: false, forbidden_context_keys: ["actual"] }
    await writeJson(path.join(reflectionRoot, "leakage-audit.json"), failedAudit)
    const reflectionEval = JSON.parse(await readFile(reflectionEvalPath, "utf-8"))
    await writeJson(reflectionEvalPath, {
      ...reflectionEval,
      leakage_audit_sha256: sha256(`${JSON.stringify(failedAudit, null, 2)}\n`),
    })
    const failedAuditExport = await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      model_family: "deep_learning",
    })
    const failedAuditManifest = JSON.parse(await readFile(path.join(exportRoot, "manifest.json"), "utf-8"))
    expect(failedAuditManifest.prediction_outputs.window_reflection).toBeUndefined()
    expect(failedAuditExport.output.warnings).toContain(
      "Window reflection output was not exported because its blind-context leakage audit did not pass",
    )

    const passingAudit = { passed: true, forbidden_context_keys: [] }
    await writeJson(path.join(reflectionRoot, "leakage-audit.json"), passingAudit)
    await writeJson(reflectionEvalPath, {
      ...reflectionEval,
      leakage_audit_sha256: sha256(`${JSON.stringify(passingAudit, null, 2)}\n`),
    })
    await writeText(
      path.join(tmp.path, ".forecast", "adjustments", "test", "test-run", "candidate-test", "leakage-audit.jsonl"),
      `${JSON.stringify({ check: "blind_input_forbidden_keys", passed: false })}\n`,
    )
    const unsafeBaselineExport = await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      model_family: "deep_learning",
    })
    const unsafeBaselineManifest = JSON.parse(await readFile(path.join(exportRoot, "manifest.json"), "utf-8"))
    expect(unsafeBaselineManifest.prediction_outputs.window_reflection).toBeUndefined()
    expect(unsafeBaselineExport.output.warnings).toContain(
      "Window reflection output was not exported because its pre-reflection baseline provenance was incomplete or unsafe",
    )
  })

  test("does not export adjusted artifacts that do not match the locked active skill", async () => {
    await using tmp = await tmpdir()

    await writeForecastTask(tmp.path)
    await writeRun(tmp.path, { runId: "validation-run", evalSplit: "val", mse: 2 })
    await writeRun(tmp.path, { runId: "test-run", evalSplit: "test", mse: 1 })
    await writeText(
      path.join(
        tmp.path,
        ".forecast",
        "adjustments",
        "test",
        "test-run",
        "stale-candidate",
        "adjusted_predictions.csv",
      ),
      "timestamp,actual,predicted,raw_predicted\n2026-01-01,1,1.05,1.1\n",
    )
    await writeJson(
      path.join(tmp.path, ".forecast", "adjustments", "test", "test-run", "stale-candidate", "adjusted_eval.json"),
      {
        run_id: "test-run",
        candidate_id: "stale-candidate",
        split: "test",
        raw_eval_path: ".forecast/runs/test-run/eval.json",
        adjusted_predictions_path: ".forecast/adjustments/test/test-run/stale-candidate/adjusted_predictions.csv",
        raw_metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
        adjusted_metrics: { mse: 0.8, mae: 0.4, wape: 0.2, mase: null },
        delta_vs_raw: { mse: -0.2, mae: -0.1, wape: -0.05, mase: null },
        timestamp: "2026-04-30T00:00:00.000Z",
      },
    )
    await writeJson(path.join(tmp.path, ".forecast", "adjustments", "skills", "active", "manifest.json"), {
      version: "v001",
      status: "locked",
      candidate_id: "active-candidate",
      source_model_set_id: "source-set",
      accepted_at: "2026-04-30T00:00:00.000Z",
      locked_at: "2026-04-30T00:00:01.000Z",
      validation_results: [".forecast/adjustments/validation/validation-run/active-candidate/adjusted_eval.json"],
      policy_path: ".forecast/adjustments/policy.json",
      parent_version: null,
    })
    await writeLockedAdjustmentState(tmp.path, "test-run", "active-candidate")

    await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      model_family: "deep_learning",
    })

    const exportRoot = path.join(tmp.path, ".forecast", "best-model")
    const manifest = JSON.parse(await readFile(path.join(exportRoot, "manifest.json"), "utf-8"))
    expect(manifest.prediction_outputs.adjusted).toBeUndefined()
    await expect(readFile(path.join(exportRoot, "adjusted", "adjusted_predictions.csv"), "utf-8")).rejects.toThrow()
  })

  test("exports reusable fitted ARIMA results from the selected validation run", async () => {
    await using tmp = await tmpdir()

    await writeForecastTask(tmp.path)
    await writeRun(tmp.path, { runId: "validation-run", model: "ARIMA", evalSplit: "val", mse: 2 })
    await writeRun(tmp.path, { runId: "test-run", model: "ARIMA", evalSplit: "test", mse: 1 })

    const { result } = await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      model_family: "statistical",
    })

    const exportRoot = path.join(tmp.path, ".forecast", "best-model")
    expect(result.metadata.status).toBe("success")
    expect(result.metadata.warnings).toEqual([])
    expect(await readFile(path.join(exportRoot, "model", "arima-results.pkl"), "utf-8")).toBe("arima-results")
    const manifest = JSON.parse(await readFile(path.join(exportRoot, "manifest.json"), "utf-8"))
    expect(manifest.artifacts).toEqual([
      {
        kind: "arima_results",
        path: "model/arima-results.pkl",
        format: "statsmodels_arima_results",
        sha256: sha256("arima-results"),
        model: "ARIMA",
        model_family: "statistical",
      },
    ])
  })

  test("exports reusable fitted results for an added statistical model", async () => {
    await using tmp = await tmpdir()

    await writeForecastTask(tmp.path)
    await writeRun(tmp.path, { runId: "validation-run", model: "ETS", evalSplit: "val", mse: 2 })
    await writeRun(tmp.path, { runId: "test-run", model: "ETS", evalSplit: "test", mse: 1 })

    const { result } = await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      model_family: "statistical",
    })

    const exportRoot = path.join(tmp.path, ".forecast", "best-model")
    expect(result.metadata.status).toBe("success")
    expect(result.metadata.warnings).toEqual([])
    expect(await readFile(path.join(exportRoot, "model", "statistical-results.pkl"), "utf-8")).toBe(
      "statistical-results",
    )
    const manifest = JSON.parse(await readFile(path.join(exportRoot, "manifest.json"), "utf-8"))
    expect(manifest.artifacts).toEqual([
      {
        kind: "statistical_results",
        path: "model/statistical-results.pkl",
        format: "statsmodels_results_pickle",
        sha256: sha256("statistical-results"),
        model: "ETS",
        model_family: "statistical",
      },
    ])
  })

  test("exports the selected foundation-model repository reference", async () => {
    await using tmp = await tmpdir()

    await writeForecastTask(tmp.path)
    await writeRun(tmp.path, { runId: "validation-run", model: "Chronos", evalSplit: "val", mse: 2 })
    await writeRun(tmp.path, { runId: "test-run", model: "Chronos", evalSplit: "test", mse: 1 })
    await writeJson(path.join(tmp.path, ".forecast", "runs", "test-run", "model", "reuse.json"), {
      source_run_id: "validation-run",
      source_artifact_kind: "foundation_model_reference",
      source_artifact_sha256: computeSpecHash({
        repo_id: "amazon/chronos-bolt-base",
        revision: FOUNDATION_REVISION,
      }),
      foundation_revision: FOUNDATION_REVISION,
      training_performed: false,
      evaluated_split: "test",
    })

    const { result } = await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      model_family: "foundation",
    })

    expect(result.metadata.status).toBe("success")
    const exportRoot = path.join(tmp.path, ".forecast", "best-model")
    const manifest = JSON.parse(await readFile(path.join(exportRoot, "manifest.json"), "utf-8"))
    const artifact = JSON.parse(await readFile(path.join(exportRoot, "model", "artifact.json"), "utf-8"))
    expect(manifest.artifacts).toEqual([
      {
        kind: "external_reference",
        reference: "amazon/chronos-bolt-base",
        reason: "The selected foundation model is reused from its pretrained repository for zero-shot inference.",
        model: "Chronos",
        model_family: "foundation",
        external_model_id: "amazon/chronos-bolt-base",
        revision: FOUNDATION_REVISION,
        sha256: computeSpecHash({
          repo_id: "amazon/chronos-bolt-base",
          revision: FOUNDATION_REVISION,
        }),
      },
    ])
    expect(artifact).toEqual(manifest.artifacts[0])
  })

  test("exports a user prediction prior as an external reference", async () => {
    await using tmp = await tmpdir()

    await writeForecastTask(tmp.path)
    await writeUserPriorRun(tmp.path, { runId: "validation-run", evalSplit: "val", mse: 2 })
    await writeUserPriorRun(tmp.path, { runId: "test-run", evalSplit: "test", mse: 1 })

    const { result } = await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      model_family: "user_prior",
      selection_reason: "user prediction prior workflow",
    })

    const exportRoot = path.join(tmp.path, ".forecast", "best-model")
    const manifest = JSON.parse(await readFile(path.join(exportRoot, "manifest.json"), "utf-8"))
    const artifact = JSON.parse(await readFile(path.join(exportRoot, "model", "artifact.json"), "utf-8"))

    expect(result.metadata.status).toBe("success")
    expect(manifest.artifacts[0]).toMatchObject({
      kind: "external_reference",
      reference: "user_prediction_prior",
      model: "UserPrior",
      model_family: "user_prior",
      external_model_id: "user_prediction_prior",
    })
    expect(artifact.reference).toBe("user_prediction_prior")
    expect(await readFile(path.join(exportRoot, "USAGE.md"), "utf-8")).toContain("user_prediction_prior")
  })

  test("exports a legacy foundation validation run whose spec omitted phase", async () => {
    await using tmp = await tmpdir()

    await writeForecastTask(tmp.path)
    await writeRun(tmp.path, { runId: "validation-run", model: "Chronos", evalSplit: "val", mse: 2 })
    await writeRun(tmp.path, { runId: "test-run", model: "Chronos", evalSplit: "test", mse: 1 })

    const validationSpecPath = path.join(tmp.path, ".forecast", "runs", "validation-run", "spec.json")
    const validationEvalPath = path.join(tmp.path, ".forecast", "runs", "validation-run", "eval.json")
    const validationSpec = JSON.parse(await readFile(validationSpecPath, "utf-8"))
    delete validationSpec.phase
    await writeJson(validationSpecPath, validationSpec)
    const validationEval = JSON.parse(await readFile(validationEvalPath, "utf-8"))
    validationEval.spec_hash = computeSpecHash(validationSpec)
    await writeJson(validationEvalPath, validationEval)

    const { result } = await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      model_family: "foundation",
    })

    expect(result.metadata.status).toBe("success")
    const manifest = JSON.parse(
      await readFile(path.join(tmp.path, ".forecast", "best-model", "manifest.json"), "utf-8"),
    )
    expect(manifest.validation_run_id).toBe("validation-run")
    expect(manifest.test_run_id).toBe("test-run")
  })

  test("rerun replaces stale best-model files", async () => {
    await using tmp = await tmpdir()

    await writeForecastTask(tmp.path)
    await writeRun(tmp.path, { runId: "validation-run", evalSplit: "val", mse: 2 })
    await writeRun(tmp.path, { runId: "test-run", evalSplit: "test", mse: 1 })
    await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      model_family: "deep_learning",
    })

    const stalePath = path.join(tmp.path, ".forecast", "best-model", "stale.txt")
    await writeText(stalePath, "stale")
    await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      model_family: "deep_learning",
    })

    await expect(readFile(stalePath, "utf-8")).rejects.toThrow()
    expect(await readFile(path.join(tmp.path, ".forecast", "best-model", "manifest.json"), "utf-8")).toContain(
      "test-run",
    )
  })

  test("missing required artifacts block export and preserve previous package", async () => {
    await using tmp = await tmpdir()

    await writeForecastTask(tmp.path)
    await writeRun(tmp.path, { runId: "validation-run", evalSplit: "val", mse: 2 })
    await writeRun(tmp.path, { runId: "test-run", evalSplit: "test", mse: 1 })
    await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      model_family: "deep_learning",
    })

    await writeRun(tmp.path, {
      runId: "broken-test-run",
      evalSplit: "test",
      mse: 0.5,
      missing: ["predictions.csv"],
    })

    const { result } = await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "broken-test-run",
      model_family: "deep_learning",
    })
    const preservedManifest = JSON.parse(
      await readFile(path.join(tmp.path, ".forecast", "best-model", "manifest.json"), "utf-8"),
    )

    expect(result.metadata.status).toBe("blocked")
    expect(result.metadata.missing_artifacts.some((item) => item.endsWith("predictions.csv"))).toBe(true)
    expect(preservedManifest.test_run_id).toBe("test-run")
  })

  test("missing selected validation checkpoint blocks export", async () => {
    await using tmp = await tmpdir()

    await writeForecastTask(tmp.path)
    await writeRun(tmp.path, {
      runId: "validation-run",
      evalSplit: "val",
      mse: 2,
      checkpoint: false,
    })
    await writeRun(tmp.path, { runId: "test-run", evalSplit: "test", mse: 1 })

    const { result, output } = await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      model_family: "deep_learning",
    })

    expect(result.metadata.status).toBe("blocked")
    expect(result.metadata.missing_artifacts.some((item) => item.endsWith("checkpoint.pth"))).toBe(true)
    expect(output.reason).toBe("Selected validation model artifact identity is invalid")
  })

  test("tampered selected validation checkpoint blocks export", async () => {
    await using tmp = await tmpdir()

    await writeForecastTask(tmp.path)
    await writeRun(tmp.path, { runId: "validation-run", evalSplit: "val", mse: 2 })
    await writeRun(tmp.path, { runId: "test-run", evalSplit: "test", mse: 1 })
    await writeText(
      path.join(tmp.path, ".forecast", "runs", "validation-run", "model", "checkpoint.pth"),
      "tampered-checkpoint",
    )

    const { result, output } = await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      model_family: "deep_learning",
    })

    expect(result.metadata.status).toBe("blocked")
    expect(result.metadata.warnings).toContain("Reusable model artifact does not match its validation metadata")
    expect(output.reason).toBe("Selected validation model artifact identity is invalid")
  })

  test("mismatched validation and test artifacts block export", async () => {
    await using tmp = await tmpdir()

    await writeForecastTask(tmp.path)
    await writeRun(tmp.path, { runId: "validation-run", evalSplit: "val", mse: 2 })
    await writeRun(tmp.path, { runId: "test-run", evalSplit: "test", mse: 1 })

    const mismatchedSpec = {
      model: "DLinear",
      dataset_path: task.dataset_path,
      target_col: task.target_col,
      time_col: task.time_col,
      pred_len: task.pred_len,
      seq_len: task.seq_len,
      label_len: task.label_len,
      train_ratio: task.train_ratio,
      val_ratio: task.val_ratio,
      test_ratio: task.test_ratio,
      metrics: task.eval_metrics,
      phase: "forecast",
      eval_split: "val",
    }
    await writeJson(path.join(tmp.path, ".forecast", "runs", "test-run", "spec.json"), mismatchedSpec)
    await writeJson(path.join(tmp.path, ".forecast", "runs", "test-run", "eval.json"), {
      run_id: "test-run",
      evaluator_hash: "e".repeat(64),
      spec_hash: computeSpecHash(mismatchedSpec),
      metrics: {
        mse: 1,
        mae: 0.5,
        wape: 0.25,
        mase: 0.125,
      },
      eval_split: "val",
      timestamp: "2026-04-30T00:00:00.000Z",
    })

    const { result, output } = await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      model_family: "deep_learning",
    })

    expect(result.metadata.status).toBe("blocked")
    expect(output.reason).toBe("Selected test evaluation artifact binding is invalid")
    await expect(readFile(path.join(tmp.path, ".forecast", "best-model", "manifest.json"), "utf-8")).rejects.toThrow()
  })

  test("test metrics with mismatched model reuse provenance block export", async () => {
    await using tmp = await tmpdir()

    await writeForecastTask(tmp.path)
    await writeRun(tmp.path, { runId: "validation-run", evalSplit: "val", mse: 2 })
    await writeRun(tmp.path, {
      runId: "test-run",
      evalSplit: "test",
      mse: 1,
      reuseSourceRunId: "different-validation-run",
    })

    const { result, output } = await exportBestModel(tmp.path, {
      validation_run_id: "validation-run",
      test_run_id: "test-run",
      model_family: "deep_learning",
    })

    expect(result.metadata.status).toBe("blocked")
    expect(output.reason).toBe("Selected test evaluation artifact binding is invalid")
  })
})
