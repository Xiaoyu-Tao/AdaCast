import { readFile, readdir } from "fs/promises"
import path from "path"
import { Filesystem } from "@/util/filesystem"
import { AdjustmentCandidate, AdjustmentEvalResult, AdjustmentSourceModelSet } from "./types"
import {
  adjustmentRoot,
  forecastRoot,
  projectRelative,
  readJsonFile,
  readPhaseState,
  runIdIsSafe,
} from "./forecast-adjustment-common"

export function validationEvalPath(runId: string, candidateId: string): string {
  return path.join(adjustmentRoot(), "validation", runId, candidateId, "adjusted_eval.json")
}

export function validationAuditPath(runId: string, candidateId: string): string {
  return path.join(adjustmentRoot(), "validation", runId, candidateId, "leakage-audit.jsonl")
}

export async function leakageAuditPassed(file: string): Promise<boolean> {
  const raw = await readFile(file, "utf-8")
  const events = raw
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as { passed?: unknown })
  return events.length > 0 && events.every((event) => event.passed === true)
}

export async function collectAcceptedValidationResultPaths(
  candidateId: string,
  sourceSet: AdjustmentSourceModelSet,
): Promise<string[]> {
  const completed = await collectCompletedValidationResults(candidateId, sourceSet)
  return completed
    .filter((item) => item.result.acceptance_decision?.accepted)
    .map((item) => item.path)
}

export async function collectCompletedValidationResults(
  candidateId: string,
  sourceSet: AdjustmentSourceModelSet,
): Promise<Array<{ path: string; result: AdjustmentEvalResult }>> {
  const completed: Array<{ path: string; result: AdjustmentEvalResult }> = []
  for (const member of sourceSet.members) {
    const runId = member.validation_run_id
    if (!runIdIsSafe(runId)) continue
    const evalPath = validationEvalPath(runId, candidateId)
    const auditPath = validationAuditPath(runId, candidateId)
    if (!(await Filesystem.exists(evalPath)) || !(await Filesystem.exists(auditPath))) continue

    const validationEval = AdjustmentEvalResult.parse(await readJsonFile(evalPath))
    if (
      validationEval.run_id !== runId ||
      validationEval.candidate_id !== candidateId ||
      validationEval.split !== "val" ||
      validationEval.acceptance_decision === undefined
    ) {
      continue
    }
    if (!(await leakageAuditPassed(auditPath))) continue
    const resultPath = projectRelative(evalPath)
    completed.push({ path: resultPath, result: validationEval })
  }
  return completed
}

export function isUserPriorMirrorCandidate(candidate: AdjustmentCandidate): boolean {
  return candidate.origin === "user_prior_mirror" || candidate.llm_provenance.user_prior_mirror === true
}

export async function hasResolvedUserPriorMirror(root = forecastRoot()): Promise<boolean> {
  const sourceSet = AdjustmentSourceModelSet.safeParse(
    await readJsonFile<Record<string, unknown>>(path.join(root, "adjustments", "source-model-set.json")).catch(
      () => null,
    ),
  )
  if (!sourceSet.success) return false
  const candidatesRoot = path.join(root, "adjustments", "candidates")
  const entries = await readdir(candidatesRoot, { withFileTypes: true }).catch(() => [])
  for (const entry of entries) {
    if (!entry.isDirectory() || !runIdIsSafe(entry.name)) continue
    const candidate = AdjustmentCandidate.safeParse(
      await readJsonFile<Record<string, unknown>>(path.join(candidatesRoot, entry.name, "candidate.json")).catch(
        () => null,
      ),
    )
    if (!candidate.success || !isUserPriorMirrorCandidate(candidate.data)) continue
    if (candidate.data.origin === "user_prior_mirror" && candidate.data.user_prior_contract === null) continue
    if (candidate.data.source_model_set_id !== sourceSet.data.source_model_set_id) continue
    const completed = await collectCompletedValidationResults(candidate.data.candidate_id, sourceSet.data)
    const hasAcceptedResult = completed.some((item) => item.result.acceptance_decision?.accepted === true)
    if (candidate.data.status === "accepted" && hasAcceptedResult) return true
    if (candidate.data.status === "rejected" && completed.length > 0 && !hasAcceptedResult) return true
  }
  return false
}

export async function taskRequiresUserPriorMirror(root = forecastRoot()): Promise<boolean> {
  const task = await readJsonFile<Record<string, unknown>>(path.join(root, "task.json")).catch(() => null)
  return typeof task?.adjustment_skill_priors === "string" && task.adjustment_skill_priors.trim().length > 0
}

export async function assertUserPriorMirrorResolved(operation: string): Promise<void> {
  const state = await readPhaseState()
  if (!state.adjustment.source_model_set_locked || !state.adjustment.fit_runs_ready) return
  if ((await taskRequiresUserPriorMirror()) && !(await hasResolvedUserPriorMirror())) {
    throw new Error(
      `${operation} cannot finish adjustment while adjustment_skill_priors remains unresolved. ` +
        "Independently inspect, write, evaluate on validation-back, and accept or reject at least one user_prior_mirror candidate first.",
    )
  }
}

export async function retainableValidationCandidateIds(): Promise<string[]> {
  const sourceSetPath = path.join(adjustmentRoot(), "source-model-set.json")
  if (!(await Filesystem.exists(sourceSetPath))) return []
  const sourceSet = AdjustmentSourceModelSet.parse(await readJsonFile(sourceSetPath))
  const candidatesRoot = path.join(adjustmentRoot(), "candidates")
  const entries = await readdir(candidatesRoot, { withFileTypes: true }).catch(() => [])
  const accepted: string[] = []
  for (const entry of entries) {
    const candidateId = entry.name
    if (!entry.isDirectory() || !runIdIsSafe(candidateId)) continue
    const candidatePayload = await readJsonFile<Record<string, unknown>>(
      path.join(candidatesRoot, candidateId, "candidate.json"),
    ).catch(() => null)
    const candidate = AdjustmentCandidate.safeParse(candidatePayload)
    if (!candidate.success || candidate.data.source_model_set_id !== sourceSet.source_model_set_id) continue
    if ((await collectAcceptedValidationResultPaths(candidateId, sourceSet)).length > 0) {
      accepted.push(candidateId)
    }
  }
  return accepted.sort()
}

export async function assertNoRetainableValidationCandidate(operation: string): Promise<void> {
  const candidateIds = await retainableValidationCandidateIds()
  if (candidateIds.length > 0) {
    throw new Error(
      `${operation} cannot discard validation-accepted candidate(s): ${candidateIds.join(", ")}. ` +
        "Accept and lock a candidate instead; a zero-trigger accepted candidate is a retainable dormant skill.",
    )
  }
}
