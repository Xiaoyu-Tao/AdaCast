import { createHash } from "node:crypto"
import path from "node:path"
import z from "zod"
import { readJsonFile, writeJsonFile } from "./forecast-adjustment-common"
import { computeFileSha256 } from "./model-artifact"
import { CriticPredictionEvidence, CriticPredictionPoint, CriticPredictionState } from "./types"

export const CriticStateArtifact = z.object({
  path: z.string().min(1),
  sha256: z.string().length(64),
})
export type CriticStateArtifact = z.infer<typeof CriticStateArtifact>

function relocatedCheckpointPath(value: string): string {
  const relocationRoot = process.env.ADACAST_LITE_REFLECTION_CHECKPOINT_RELOCATE_ROOT
  const normalized = value.replaceAll("\\", "/")
  if (relocationRoot) {
    const resolvedRoot = path.resolve(relocationRoot)
    const sourceRoot = process.env.ADACAST_LITE_REFLECTION_CHECKPOINT_SOURCE_ROOT?.replaceAll("\\", "/").replace(/\/$/, "")
    if (sourceRoot && (normalized === sourceRoot || normalized.startsWith(`${sourceRoot}/`))) {
      return path.join(resolvedRoot, normalized.slice(sourceRoot.length + 1))
    }
    const marker = `/${path.basename(resolvedRoot)}/`
    const markerIndex = normalized.lastIndexOf(marker)
    if (markerIndex >= 0) return path.join(resolvedRoot, normalized.slice(markerIndex + marker.length))
  }
  return value
}

function assertGapContract(evidenceGap: 0 | 1, neededEvidence: string): void {
  if (evidenceGap === 0 && neededEvidence.trim()) throw new Error("A closed evidence gap cannot request evidence")
  if (evidenceGap === 1 && !neededEvidence.trim()) throw new Error("An open evidence gap must describe needed evidence")
}

function assertPredictionLayout(
  previous: z.infer<typeof CriticPredictionState>,
  predictions: z.infer<typeof CriticPredictionPoint>[],
): void {
  if (predictions.length !== previous.predictions.length)
    throw new Error("A state transition cannot change horizon length")
  for (let index = 0; index < predictions.length; index++) {
    const before = previous.predictions[index]
    const after = predictions[index]
    if (!before || !after || before.horizon_index !== after.horizon_index || before.timestamp !== after.timestamp) {
      throw new Error("A state transition cannot change horizon indices or timestamps")
    }
  }
}

function stateBytes(state: z.infer<typeof CriticPredictionState>): string {
  return JSON.stringify(state, null, 2)
}

async function persistState(
  statesDir: string,
  state: z.infer<typeof CriticPredictionState>,
): Promise<CriticStateArtifact> {
  const bytes = stateBytes(state)
  const sha256 = createHash("sha256").update(bytes).digest("hex")
  const statePath = path.join(statesDir, `state-${sha256}.json`)
  await writeJsonFile(statePath, state)
  const actual = await computeFileSha256(statePath)
  if (actual !== sha256) throw new Error("Persisted Critic state hash does not match its content address")
  const artifact = { path: statePath, sha256 }
  await writeJsonFile(path.join(statesDir, "current.json"), artifact)
  return artifact
}

async function assertPreviousStateArtifact(
  state: z.infer<typeof CriticPredictionState>,
  artifact: CriticStateArtifact,
): Promise<void> {
  const artifactPath = relocatedCheckpointPath(artifact.path)
  if ((await computeFileSha256(artifactPath)) !== artifact.sha256) throw new Error("Previous Critic state hash changed")
  const persisted = CriticPredictionState.parse(await readJsonFile<unknown>(artifactPath))
  if (JSON.stringify(persisted) !== JSON.stringify(state)) {
    throw new Error("Previous Critic state does not match its artifact")
  }
}

export async function commitCriticGap(input: {
  statesDir: string
  previousState: z.infer<typeof CriticPredictionState>
  previousArtifact: CriticStateArtifact
  evidenceGap: 0 | 1
  neededEvidence: string
}): Promise<{ state: z.infer<typeof CriticPredictionState>; artifact: CriticStateArtifact }> {
  const previousState = CriticPredictionState.parse(input.previousState)
  const artifact = CriticStateArtifact.parse(input.previousArtifact)
  await assertPreviousStateArtifact(previousState, artifact)
  assertGapContract(input.evidenceGap, input.neededEvidence)
  const state = CriticPredictionState.parse({
    ...previousState,
    evidence_gap: input.evidenceGap,
    needed_evidence: input.neededEvidence.trim(),
  })
  return { state, artifact: await persistState(input.statesDir, state) }
}

export async function commitCriticSafetyTransition(input: {
  statesDir: string
  previousState: z.infer<typeof CriticPredictionState>
  previousArtifact: CriticStateArtifact
  predictions: z.infer<typeof CriticPredictionPoint>[]
}): Promise<{ state: z.infer<typeof CriticPredictionState>; artifact: CriticStateArtifact }> {
  const previousState = CriticPredictionState.parse(input.previousState)
  const previousArtifact = CriticStateArtifact.parse(input.previousArtifact)
  await assertPreviousStateArtifact(previousState, previousArtifact)
  const predictions = z.array(CriticPredictionPoint).min(1).parse(input.predictions)
  assertPredictionLayout(previousState, predictions)
  const state = CriticPredictionState.parse({
    ...previousState,
    predictions,
  })
  return { state, artifact: await persistState(input.statesDir, state) }
}

export async function commitCriticEvidenceTransition(input: {
  statesDir: string
  previousState: z.infer<typeof CriticPredictionState>
  previousArtifact: CriticStateArtifact
  selectedEvidence: z.infer<typeof CriticPredictionEvidence>
  predictions: z.infer<typeof CriticPredictionPoint>[]
  evidenceGap: 0 | 1
  neededEvidence: string
  expectedCandidateArtifact: CriticStateArtifact
}): Promise<{ state: z.infer<typeof CriticPredictionState>; artifact: CriticStateArtifact }> {
  const previousState = CriticPredictionState.parse(input.previousState)
  const previousArtifact = CriticStateArtifact.parse(input.previousArtifact)
  const selected = CriticPredictionEvidence.parse(input.selectedEvidence)
  const candidateArtifact = CriticStateArtifact.parse(input.expectedCandidateArtifact)
  await assertPreviousStateArtifact(previousState, previousArtifact)
  if (selected.artifact_path !== candidateArtifact.path || selected.artifact_sha256 !== candidateArtifact.sha256) {
    throw new Error("Selected evidence does not match the prepared candidate artifact")
  }
  const candidateArtifactPath = relocatedCheckpointPath(candidateArtifact.path)
  if ((await computeFileSha256(candidateArtifactPath)) !== candidateArtifact.sha256) {
    throw new Error("Selected evidence artifact changed")
  }
  const candidate = await readJsonFile<unknown>(candidateArtifactPath)
  if (
    typeof candidate !== "object" ||
    candidate === null ||
    Array.isArray(candidate) ||
    (candidate as Record<string, unknown>).evidence_candidate_id !== selected.evidence_candidate_id ||
    (candidate as Record<string, unknown>).candidate_type !== selected.candidate_type
  ) {
    throw new Error("Selected evidence identity does not match its artifact content")
  }
  if (previousState.evidences.some((item) => item.evidence_candidate_id === selected.evidence_candidate_id)) {
    throw new Error("Evidence cannot be applied more than once")
  }
  const predictions = z.array(CriticPredictionPoint).min(1).parse(input.predictions)
  assertPredictionLayout(previousState, predictions)
  assertGapContract(input.evidenceGap, input.neededEvidence)
  const state = CriticPredictionState.parse({
    ...previousState,
    evidences: [...previousState.evidences, selected],
    predictions,
    evidence_gap: input.evidenceGap,
    needed_evidence: input.neededEvidence.trim(),
  })
  return { state, artifact: await persistState(input.statesDir, state) }
}
