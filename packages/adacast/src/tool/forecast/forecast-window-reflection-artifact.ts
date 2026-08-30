import path from "path"
import { stat } from "fs/promises"
import { Instance } from "@/project/instance"
import { Filesystem } from "@/util/filesystem"
import {
  ForecastReflectionTestRecord,
  ForecastReflectionValidationRecord,
  ForecastWindowReflectionArtifact,
  ForecastWindowReflectionPolicy,
} from "./types"
import { computeFileSha256 } from "./model-artifact"

type ValidationRecord = ForecastReflectionValidationRecord
type TestRecord = ForecastReflectionTestRecord

function resolveReflectionArtifactPath(value: string): string {
  const relocationRoot = process.env.ADACAST_REFLECTION_CHECKPOINT_RELOCATE_ROOT
  const normalized = value.replaceAll("\\", "/")
  if (relocationRoot) {
    const resolvedRoot = path.resolve(relocationRoot)
    const sourceRoot = process.env.ADACAST_REFLECTION_CHECKPOINT_SOURCE_ROOT?.replaceAll("\\", "/").replace(/\/$/, "")
    if (sourceRoot && (normalized === sourceRoot || normalized.startsWith(`${sourceRoot}/`))) {
      return path.join(resolvedRoot, normalized.slice(sourceRoot.length + 1))
    }
    const marker = `/${path.basename(resolvedRoot)}/`
    const markerIndex = normalized.lastIndexOf(marker)
    if (markerIndex >= 0) return path.join(resolvedRoot, normalized.slice(markerIndex + marker.length))
  }
  if (path.isAbsolute(value)) return value
  return path.resolve(Instance.directory, value)
}

async function pathsReferToSameFile(actualPath: string, expectedPath: string): Promise<boolean> {
  if (actualPath === expectedPath) return true
  try {
    const [actual, expected] = await Promise.all([stat(actualPath), stat(expectedPath)])
    return actual.dev === expected.dev && actual.ino === expected.ino
  } catch {
    return false
  }
}

export async function validateWindowReflectionPolicyArtifact(
  forecastRoot: string,
  policyPath: string | null,
  expected: {
    enabled: boolean
    controllerVersion: string
    maxReflectionRounds: number
    maxFormatRetries: number
    policySha256: string | null
  },
): Promise<ForecastWindowReflectionPolicy | null> {
  if (!policyPath) return null
  const resolvedPolicyPath = path.resolve(Instance.directory, policyPath)
  const expectedPolicyPath = path.resolve(forecastRoot, "reflection", "policy.json")
  if (!(await pathsReferToSameFile(resolvedPolicyPath, expectedPolicyPath))) return null
  const parsed = ForecastWindowReflectionPolicy.safeParse(
    await Filesystem.readJson<unknown>(resolvedPolicyPath).catch(() => null),
  )
  if (
    !parsed.success ||
    parsed.data.enabled !== expected.enabled ||
    parsed.data.controller_version !== expected.controllerVersion ||
    parsed.data.max_reflection_rounds !== expected.maxReflectionRounds ||
    parsed.data.max_format_retries !== expected.maxFormatRetries ||
    !expected.policySha256 ||
    (await computeFileSha256(resolvedPolicyPath)) !== expected.policySha256
  ) {
    return null
  }
  return parsed.data
}

function metricsMatch(
  actual: ForecastWindowReflectionArtifact["baseline_metrics"],
  expected: ValidationRecord["baseline_metrics"] | TestRecord["baseline_metrics"],
): boolean {
  if (!expected) return false
  return ["mse", "mae", "wape", "mase"].every(
    (key) => actual[key as keyof typeof actual] === expected[key as keyof typeof expected],
  )
}

export async function validateWindowReflectionRecordArtifact(
  forecastRoot: string,
  record: ValidationRecord | TestRecord,
  split: "val" | "test",
): Promise<{
  artifact: ForecastWindowReflectionArtifact
  evalPath: string
  predictionsPath: string
  auditPath: string
  auditPassed: boolean
  protocolPath: string | null
} | null> {
  if (!record.result_path || !record.baseline_metrics || !record.final_metrics) return null
  const expectedRunId =
    split === "val" ? (record as ValidationRecord).validation_run_id : (record as TestRecord).test_run_id
  const expectedRoot = path.resolve(forecastRoot, "reflection", split === "val" ? "validation" : "test")
  const candidateSegment = record.adjustment_candidate_id ?? "raw"
  if (!/^[A-Za-z0-9_-]+$/.test(expectedRunId) || !/^[A-Za-z0-9_-]+$/.test(candidateSegment)) return null
  const expectedEvalPath = path.join(expectedRoot, expectedRunId, candidateSegment, "reflection_eval.json")
  const evalPath = path.resolve(Instance.directory, record.result_path)
  if (!(await pathsReferToSameFile(evalPath, expectedEvalPath))) return null

  const parsed = ForecastWindowReflectionArtifact.safeParse(
    await Filesystem.readJson<unknown>(evalPath).catch(() => null),
  )
  if (!parsed.success) return null
  const artifact = parsed.data
  const expectedStatus = split === "val" ? (record as ValidationRecord).status : "completed"
  if (
    artifact.source_run_id !== expectedRunId ||
    artifact.split !== split ||
    artifact.baseline_kind !== record.baseline_kind ||
    artifact.adjustment_candidate_id !== record.adjustment_candidate_id ||
    artifact.status !== expectedStatus ||
    artifact.trigger_count !== record.trigger_count ||
    artifact.corrected_count !== record.corrected_count ||
    artifact.fallback_count !== record.fallback_count ||
    !metricsMatch(artifact.baseline_metrics, record.baseline_metrics) ||
    !metricsMatch(artifact.final_metrics, record.final_metrics)
  ) {
    return null
  }
  if (split === "val" && artifact.delta_mse !== (record as ValidationRecord).delta_mse) return null
  if (split === "test") {
    const testRecord = record as TestRecord
    if (
      artifact.skill_match_count !== testRecord.skill_match_count ||
      artifact.llm_required_count !== testRecord.llm_required_count ||
      artifact.deterministic_fallback_count !== testRecord.deterministic_fallback_count ||
      artifact.raw_fallback_count !== testRecord.raw_fallback_count
    ) {
      return null
    }
  }

  const predictionsPath = resolveReflectionArtifactPath(artifact.final_predictions_path)
  if (!(await pathsReferToSameFile(predictionsPath, path.join(path.dirname(evalPath), "final_predictions.csv")))) {
    return null
  }
  if (!(await Filesystem.exists(predictionsPath))) return null
  if ((await computeFileSha256(predictionsPath)) !== artifact.final_predictions_sha256) return null

  const auditPath = path.join(path.dirname(evalPath), "leakage-audit.json")
  if (!(await Filesystem.exists(auditPath))) return null
  if ((await computeFileSha256(auditPath)) !== artifact.leakage_audit_sha256) return null
  if ((artifact.llm_protocol_path === null) !== (artifact.llm_protocol_sha256 === null)) return null
  let protocolPath: string | null = null
  if (artifact.llm_protocol_path && artifact.llm_protocol_sha256) {
    protocolPath = resolveReflectionArtifactPath(artifact.llm_protocol_path)
    if (!(await pathsReferToSameFile(protocolPath, path.join(path.dirname(evalPath), "llm-protocol.json")))) return null
    if (!(await Filesystem.exists(protocolPath))) return null
    if ((await computeFileSha256(protocolPath)) !== artifact.llm_protocol_sha256) return null
  }
  if (
    split === "test" &&
    artifact.baseline_kind === "skill_adjusted" &&
    artifact.llm_required_count > 0 &&
    !protocolPath
  ) {
    return null
  }
  if (artifact.gap_predictions_sha256) {
    const gapPredictionsPath = path.join(forecastRoot, "runs", expectedRunId, "gap_predictions.csv")
    if (!(await Filesystem.exists(gapPredictionsPath))) return null
    if ((await computeFileSha256(gapPredictionsPath)) !== artifact.gap_predictions_sha256) return null
  }
  const audit = await Filesystem.readJson<{ passed?: boolean }>(auditPath).catch(() => null)
  return {
    artifact,
    evalPath,
    predictionsPath,
    auditPath,
    auditPassed: audit?.passed === true,
    protocolPath,
  }
}
