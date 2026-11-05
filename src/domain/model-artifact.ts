import { readFile } from "node:fs/promises"
import { createHash } from "node:crypto"
import path from "node:path"
import { Filesystem } from "./runtime"
import { computeSpecHash } from "./hash"
import { foundationModelReference, statisticalModelArtifact } from "./model-family"
import type { EvalResult } from "./types"

export type ModelArtifactIdentity = {
  kind:
    | "baseline_model"
    | "torch_state_dict"
    | "statsmodels_arima_results"
    | "statsmodels_statistical_results"
    | "foundation_model_reference"
    | "external_prediction_prior"
  sha256: string
  revision?: string
  repoId?: string
  sourcePath?: string
}

function normalizeModelName(model: string): string {
  return model
    .trim()
    .toLocaleLowerCase("en-US")
    .replace(/[\s_-]+/g, "")
}

export async function computeFileSha256(file: string): Promise<string> {
  return createHash("sha256")
    .update(await readFile(file))
    .digest("hex")
}

async function computePredictionArtifactHashes(runDir: string) {
  return {
    predictions_sha256: await computeFileSha256(path.join(runDir, "predictions.csv")),
    actual_sha256: await computeFileSha256(path.join(runDir, "actual.npy")),
    pred_sha256: await computeFileSha256(path.join(runDir, "pred.npy")),
    train_scale_sha256: await computeFileSha256(path.join(runDir, "train_scale.json")),
  }
}

export async function readValidationModelArtifactIdentity(
  runDir: string,
  model: string,
): Promise<ModelArtifactIdentity> {
  if (["naive","mean","seasonalnaive","linear"].includes(normalizeModelName(model))) {
    const file=path.join(runDir,"model/model.json")
    const own=await Filesystem.readJson<any>(file)
    if(own.name!==model || own.family!=="baseline")throw new Error("Baseline identity mismatch")
    return {kind:"baseline_model",sha256:await computeFileSha256(file),sourcePath:file}
  }
  if (normalizeModelName(model) === "userprior") {
    const priorPath = path.join(runDir, "model", "prior-source.json")
    const prior = await Filesystem.readJson<Record<string, unknown>>(priorPath)
    const actualSha256 = await computeFileSha256(priorPath)
    if (
      prior.kind !== "external_prediction_prior" ||
      normalizeModelName(String(prior.model ?? "")) !== "userprior" ||
      prior.format !== "adacast_user_prediction_prior" ||
      typeof prior.source_dataset_sha256 !== "string" ||
      !/^[A-Fa-f0-9]{64}$/.test(prior.source_dataset_sha256)
    ) {
      throw new Error("External prediction prior metadata is not self-consistent")
    }
    return {
      kind: "external_prediction_prior",
      sha256: actualSha256,
      sourcePath: priorPath,
    }
  }

  const foundationReference = foundationModelReference(model)
  if (foundationReference) {
    const foundation = await Filesystem.readJson<Record<string, unknown>>(path.join(runDir, "model", "foundation.json"))
    const revision = foundation.revision
    const expectedSha256 =
      typeof revision === "string" && /^[A-Fa-f0-9]{40,64}$/.test(revision)
        ? computeSpecHash({ repo_id: foundationReference, revision })
        : null
    if (
      foundation.kind !== "foundation_model_reference" ||
      normalizeModelName(String(foundation.model ?? "")) !== normalizeModelName(model) ||
      foundation.repo_id !== foundationReference ||
      foundation.format !== "huggingface_snapshot" ||
      foundation.sha256 !== expectedSha256 ||
      typeof revision !== "string"
    ) {
      throw new Error("Foundation source metadata is not self-consistent")
    }
    const sha256 = expectedSha256 as string
    return {
      kind: "foundation_model_reference",
      sha256,
      revision,
      repoId: foundationReference,
      sourcePath: path.join(runDir, "model", "foundation.json"),
    }
  }

  const metadataPath = path.join(runDir, "model", "metadata.json")
  const metadata = await Filesystem.readJson<Record<string, unknown>>(metadataPath)
  const statisticalArtifact = statisticalModelArtifact(model)
  const artifactPath = path.join(runDir, "model", statisticalArtifact?.filename ?? "checkpoint.pth")
  const actualSha256 = await computeFileSha256(artifactPath)
  if (
    metadata.kind !== (statisticalArtifact?.metadataKind ?? "checkpoint") ||
    metadata.format !== (statisticalArtifact?.format ?? "torch_state_dict") ||
    normalizeModelName(String(metadata.model ?? "")) !== normalizeModelName(model) ||
    metadata.phase !== "forecast" ||
    metadata.eval_split !== "val" ||
    metadata.sha256 !== actualSha256
  ) {
    throw new Error("Reusable model artifact does not match its validation metadata")
  }
  return {
    kind: statisticalArtifact?.kind ?? "torch_state_dict",
    sha256: actualSha256,
    sourcePath: artifactPath,
  }
}

export async function evalArtifactBindingMatches(
  runDir: string,
  evalResult: EvalResult,
  identity: ModelArtifactIdentity,
  sourceRunId: string,
): Promise<boolean> {
  const binding = evalResult.artifact_binding
  if (!binding) return false
  let predictionArtifacts
  try {
    predictionArtifacts = await computePredictionArtifactHashes(runDir)
  } catch {
    return false
  }
  return (
    binding.model_artifact.kind === identity.kind &&
    binding.model_artifact.sha256 === identity.sha256 &&
    binding.model_artifact.source_run_id === sourceRunId &&
    binding.model_artifact.foundation_revision === (identity.revision ?? null) &&
    binding.prediction_artifacts.predictions_sha256 === predictionArtifacts.predictions_sha256 &&
    binding.prediction_artifacts.actual_sha256 === predictionArtifacts.actual_sha256 &&
    binding.prediction_artifacts.pred_sha256 === predictionArtifacts.pred_sha256 &&
    binding.prediction_artifacts.train_scale_sha256 === predictionArtifacts.train_scale_sha256
  )
}

export async function readBoundModelArtifactIdentity(
  runDir: string,
  model: string,
  evalResult: EvalResult,
  sourceRunDirOverride?: string,
): Promise<ModelArtifactIdentity> {
  const sourceRunId = evalResult.artifact_binding?.model_artifact.source_run_id
  if (!sourceRunId || !/^[A-Za-z0-9_-]+$/.test(sourceRunId)) {
    throw new Error("Evaluation result is missing a safe model artifact binding")
  }
  const currentRunId = path.basename(runDir)
  if (evalResult.run_id !== currentRunId) {
    throw new Error("Evaluation result run_id does not match its artifact directory")
  }
  const sourceRunDir = sourceRunDirOverride
    ? path.resolve(sourceRunDirOverride)
    : sourceRunId === currentRunId
      ? runDir
      : path.join(path.dirname(runDir), sourceRunId)
  if (path.basename(sourceRunDir) !== sourceRunId) {
    throw new Error("Evaluation model binding does not match the supplied source run directory")
  }
  const identity = await readValidationModelArtifactIdentity(sourceRunDir, model)
  if (!(await evalArtifactBindingMatches(runDir, evalResult, identity, sourceRunId))) {
    throw new Error("Evaluation metrics are not bound to the current model and prediction artifacts")
  }
  if (sourceRunId !== currentRunId) {
    const reuse = await Filesystem.readJson<Record<string, unknown>>(path.join(runDir, "model", "reuse.json"))
    if (
      reuse.source_run_id !== sourceRunId ||
      reuse.source_artifact_kind !== identity.kind ||
      reuse.source_artifact_sha256 !== identity.sha256 ||
      reuse.training_performed !== false ||
      reuse.evaluated_split !== evalResult.eval_split
    ) {
      throw new Error("Evaluation model binding does not match its reuse provenance")
    }
  }
  return identity
}
