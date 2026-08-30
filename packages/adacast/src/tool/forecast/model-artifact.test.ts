import { expect, test } from "bun:test"
import { mkdir, writeFile } from "fs/promises"
import { createHash } from "node:crypto"
import path from "path"
import { tmpdir } from "../../../test/fixture/fixture"
import { computeSpecHash } from "./hash"
import { readBoundModelArtifactIdentity, readValidationModelArtifactIdentity } from "./model-artifact"
import { EvalResult } from "./types"

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex")
}

async function writeBoundCheckpointRun(
  runDir: string,
  runId: string,
  options: { sourceRunDir?: string; evalSplit?: "val" | "adjustment_fit" } = {},
) {
  const sourceRunDir = options.sourceRunDir ?? runDir
  const sourceRunId = path.basename(sourceRunDir)
  const checkpoint = "original-checkpoint"
  const predictions = "timestamp,actual,predicted\n0,1,1\n"
  const actual = "actual"
  const pred = "pred"
  const trainScale = '{"train_scale":1}'
  await mkdir(path.join(sourceRunDir, "model"), { recursive: true })
  await writeFile(path.join(sourceRunDir, "model", "checkpoint.pth"), checkpoint)
  await writeFile(
    path.join(sourceRunDir, "model", "metadata.json"),
    JSON.stringify({
      kind: "checkpoint",
      path: "model/checkpoint.pth",
      sha256: sha256(checkpoint),
      model: "DLinear",
      eval_split: "val",
      phase: "forecast",
      trained_on: "train",
      format: "torch_state_dict",
    }),
  )
  await mkdir(path.join(runDir, "model"), { recursive: true })
  await writeFile(path.join(runDir, "predictions.csv"), predictions)
  await writeFile(path.join(runDir, "actual.npy"), actual)
  await writeFile(path.join(runDir, "pred.npy"), pred)
  await writeFile(path.join(runDir, "train_scale.json"), trainScale)
  if (sourceRunDir !== runDir) {
    await writeFile(
      path.join(runDir, "model", "reuse.json"),
      JSON.stringify({
        source_run_id: sourceRunId,
        source_artifact_kind: "torch_state_dict",
        source_artifact_sha256: sha256(checkpoint),
        training_performed: false,
        evaluated_split: options.evalSplit ?? "adjustment_fit",
      }),
    )
  }
  const evalResult = EvalResult.parse({
    run_id: runId,
    evaluator_hash: "e".repeat(64),
    spec_hash: runId,
    metrics: { mse: 0, mae: 0, wape: 0, mase: null },
    eval_split: options.evalSplit ?? "val",
    timestamp: "2026-06-10T00:00:00.000Z",
    artifact_binding: {
      model_artifact: {
        kind: "torch_state_dict",
        sha256: sha256(checkpoint),
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
  return evalResult
}

async function writeBoundUserPriorRuns(sourceRunDir: string, testRunDir: string, sourceRunId: string, testRunId: string) {
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
  const predictions = "timestamp,actual,predicted\n0,1,1\n"
  const actual = "actual"
  const pred = "pred"
  const trainScale = '{"train_scale":1}'
  await mkdir(path.join(sourceRunDir, "model"), { recursive: true })
  await writeFile(path.join(sourceRunDir, "model", "prior-source.json"), priorSource)
  await writeFile(path.join(sourceRunDir, "predictions.csv"), predictions)
  await writeFile(path.join(sourceRunDir, "actual.npy"), actual)
  await writeFile(path.join(sourceRunDir, "pred.npy"), pred)
  await writeFile(path.join(sourceRunDir, "train_scale.json"), trainScale)
  await mkdir(path.join(testRunDir, "model"), { recursive: true })
  await writeFile(path.join(testRunDir, "predictions.csv"), predictions)
  await writeFile(path.join(testRunDir, "actual.npy"), actual)
  await writeFile(path.join(testRunDir, "pred.npy"), pred)
  await writeFile(path.join(testRunDir, "train_scale.json"), trainScale)
  await writeFile(
    path.join(testRunDir, "model", "reuse.json"),
    JSON.stringify({
      source_run_id: sourceRunId,
      source_artifact_kind: "external_prediction_prior",
      source_artifact_sha256: sha256(priorSource),
      training_performed: false,
      evaluated_split: "test",
    }),
  )
  return EvalResult.parse({
    run_id: testRunId,
    evaluator_hash: "e".repeat(64),
    spec_hash: testRunId,
    metrics: { mse: 0, mae: 0, wape: 0, mase: null },
    eval_split: "test",
    timestamp: "2026-06-10T00:00:00.000Z",
    artifact_binding: {
      model_artifact: {
        kind: "external_prediction_prior",
        sha256: sha256(priorSource),
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

test("foundation artifact identity requires an immutable commit revision", async () => {
  await using tmp = await tmpdir()
  const modelDir = path.join(tmp.path, "model")
  await mkdir(modelDir, { recursive: true })
  const repoId = "amazon/chronos-bolt-base"
  const revision = "main"
  await writeFile(
    path.join(modelDir, "foundation.json"),
    JSON.stringify({
      kind: "foundation_model_reference",
      model: "Chronos",
      repo_id: repoId,
      revision,
      sha256: computeSpecHash({ repo_id: repoId, revision }),
      format: "huggingface_snapshot",
    }),
    "utf-8",
  )

  await expect(readValidationModelArtifactIdentity(tmp.path, "Chronos")).rejects.toThrow(
    "Foundation source metadata is not self-consistent",
  )
})

test("user prediction prior identity is bound through prior-source metadata", async () => {
  await using tmp = await tmpdir()
  const sourceRunId = "u".repeat(64)
  const testRunId = "v".repeat(64)
  const sourceRunDir = path.join(tmp.path, sourceRunId)
  const testRunDir = path.join(tmp.path, testRunId)
  const evalResult = await writeBoundUserPriorRuns(sourceRunDir, testRunDir, sourceRunId, testRunId)

  await expect(readValidationModelArtifactIdentity(sourceRunDir, "UserPrior")).resolves.toMatchObject({
    kind: "external_prediction_prior",
  })
  await expect(readBoundModelArtifactIdentity(testRunDir, "UserPrior", evalResult)).resolves.toMatchObject({
    kind: "external_prediction_prior",
  })
})

test("bound evaluation rejects a replaced checkpoint even with updated metadata", async () => {
  await using tmp = await tmpdir()
  const runId = "a".repeat(64)
  const runDir = path.join(tmp.path, runId)
  const evalResult = await writeBoundCheckpointRun(runDir, runId)

  await readBoundModelArtifactIdentity(runDir, "DLinear", evalResult)

  const replacement = "replacement-checkpoint"
  await writeFile(path.join(runDir, "model", "checkpoint.pth"), replacement)
  await writeFile(
    path.join(runDir, "model", "metadata.json"),
    JSON.stringify({
      kind: "checkpoint",
      path: "model/checkpoint.pth",
      sha256: sha256(replacement),
      model: "DLinear",
      eval_split: "val",
      phase: "forecast",
      trained_on: "train",
      format: "torch_state_dict",
    }),
  )

  await expect(readBoundModelArtifactIdentity(runDir, "DLinear", evalResult)).rejects.toThrow(
    "Evaluation metrics are not bound to the current model and prediction artifacts",
  )
})

test("bound fit evaluation accepts an explicit validation source directory", async () => {
  await using tmp = await tmpdir()
  const sourceRunId = "c".repeat(64)
  const fitRunId = "d".repeat(64)
  const sourceRunDir = path.join(tmp.path, "forecast-runs", sourceRunId)
  const fitRunDir = path.join(tmp.path, "adjustments", "fit-runs", fitRunId)
  const evalResult = await writeBoundCheckpointRun(fitRunDir, fitRunId, {
    sourceRunDir,
    evalSplit: "adjustment_fit",
  })

  const identity = await readBoundModelArtifactIdentity(fitRunDir, "DLinear", evalResult, sourceRunDir)

  expect(identity).toMatchObject({
    kind: "torch_state_dict",
    sha256: sha256("original-checkpoint"),
  })
})

test("generic statistical results are validated and bound without a checkpoint", async () => {
  await using tmp = await tmpdir()
  const runId = "e".repeat(64)
  const runDir = path.join(tmp.path, runId)
  const results = "fitted-ets-results"
  const predictions = "timestamp,actual,predicted\n0,1,1\n"
  const actual = "actual"
  const pred = "pred"
  const trainScale = '{"train_scale":1}'
  await mkdir(path.join(runDir, "model"), { recursive: true })
  await writeFile(path.join(runDir, "model", "statistical-results.pkl"), results)
  await writeFile(
    path.join(runDir, "model", "metadata.json"),
    JSON.stringify({
      kind: "statistical_results",
      path: "model/statistical-results.pkl",
      sha256: sha256(results),
      model: "ETS",
      eval_split: "val",
      phase: "forecast",
      trained_on: "train",
      format: "statsmodels_results_pickle",
    }),
  )
  await writeFile(path.join(runDir, "predictions.csv"), predictions)
  await writeFile(path.join(runDir, "actual.npy"), actual)
  await writeFile(path.join(runDir, "pred.npy"), pred)
  await writeFile(path.join(runDir, "train_scale.json"), trainScale)
  const evalResult = EvalResult.parse({
    run_id: runId,
    evaluator_hash: "e".repeat(64),
    spec_hash: runId,
    metrics: { mse: 0, mae: 0, wape: 0, mase: null },
    eval_split: "val",
    timestamp: "2026-06-10T00:00:00.000Z",
    artifact_binding: {
      model_artifact: {
        kind: "statsmodels_statistical_results",
        sha256: sha256(results),
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

  await expect(readBoundModelArtifactIdentity(runDir, "ETS", evalResult)).resolves.toMatchObject({
    kind: "statsmodels_statistical_results",
    sha256: sha256(results),
  })
})
