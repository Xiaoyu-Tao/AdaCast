import { createHash } from "node:crypto"
import { mkdir, readFile, readdir, rm } from "node:fs/promises"
import path from "node:path"
import z from "zod"
import { configuredWindowConcurrency, configuredWindowLlmTimeoutMs, Instance } from "./runtime"
import { Permission } from "./runtime"
import { ModelID, ProviderID } from "./runtime"
import { ProviderTransform } from "./runtime"
import { Session } from "./runtime"
import { MessageV2 } from "./runtime"
import { MessageID, SessionID } from "./runtime"
import { SessionPrompt } from "./runtime"
import { Filesystem } from "./runtime"
import { Tool } from "./runtime"
import DESCRIPTION from "./forecast-window-reflection.txt"
import { normalizeValidationExecutionContract, validationSpecMatchesPostForecastTestSpec } from "./execution-contract"
import {
  forecastPath,
  forecastRoot,
  projectRelative,
  readJsonFile,
  readPhaseState,
  runAdjustmentPython,
  runIdIsSafe,
  writeJsonFile,
  writePhaseState,
} from "./forecast-adjustment-common"
import {
  validateWindowReflectionPolicyArtifact,
  validateWindowReflectionRecordArtifact,
} from "./forecast-window-reflection-artifact"
import { computeFileSha256, readBoundModelArtifactIdentity } from "./model-artifact"
import {
  CriticPredictionPoint,
  CriticPredictionState,
  EvalResult,
  ForecastWindowReflectionArtifact,
  ForecastWindowReflectionPolicy,
  TaskConfig,
} from "./types"
import { assertFrozenTaskIntegrity } from "./forecast-task-integrity"
import {
  commitCriticEvidenceTransition,
  commitCriticGap,
  commitCriticSafetyTransition,
  CriticStateArtifact,
} from "./forecast-critic-state-update"

const plannerConstraints = z
  .object({
    nonnegative_target: z.boolean().optional(),
    absolute_min: z.number().finite().optional(),
    absolute_max: z.number().finite().optional(),
    max_abs_step: z.number().positive().optional(),
    window_mean_abs_delta_limit: z.number().positive().optional(),
    point_emergency_delta_limit: z.number().positive().optional(),
    rationale: z.string().max(1000).optional(),
  })
  .optional()

const parameters = z.object({
  subcommand: z.enum(["configure", "read", "status", "run_validation", "run_test", "resume_test"]),
  run_id: z.string().optional().describe("Validation or test source run id"),
  validation_run_id: z.string().optional().describe("Matching validation source for run_test"),
  adjustment_candidate_id: z.string().optional().describe("Locked deterministic skill candidate, when applied"),
  status_split: z.enum(["val", "test"]).optional().describe("Reflection split inspected by status"),
  planner_constraints: plannerConstraints,
})

const LlmResponse = z.object({
  window_index: z.number().int().nonnegative(),
  points: z.array(
    z.object({
      horizon_index: z.number().int().nonnegative(),
      timestamp: z.string(),
      value: z.number().finite(),
    }),
  ),
  rationale: z.string().min(1),
  evidence_ids: z.array(z.string()).default([]),
})

const EvidenceGapResponse = z
  .object({
    evidence_gap: z.union([z.literal(0), z.literal(1)]),
    needed_evidence: z.string(),
    evidence_coverage: z.number().min(0).max(1),
    missing_evidence: z.array(z.string().min(1)),
    evidence_consistency: z.number().min(0).max(1),
    evidence_conflicts: z.array(z.string().min(1)),
    evidence_confidence: z.number().min(0).max(1),
    rationale: z.string().min(1),
  })
  .strict()
  .superRefine((value, ctx) => {
    if (value.evidence_gap === 0 && value.needed_evidence.trim().length > 0) {
      ctx.addIssue({ code: "custom", message: "A closed evidence gap cannot request more evidence" })
    }
    if (value.evidence_gap === 1 && value.needed_evidence.trim().length === 0) {
      ctx.addIssue({ code: "custom", message: "An open evidence gap must describe the evidence still needed" })
    }
    if (value.evidence_gap === 0 && value.missing_evidence.length > 0) {
      ctx.addIssue({ code: "custom", message: "A closed evidence gap cannot list missing evidence" })
    }
    if (value.evidence_gap === 1 && value.missing_evidence.length === 0 && value.evidence_conflicts.length === 0) {
      ctx.addIssue({
        code: "custom",
        message: "An open evidence gap must list missing evidence or a material conflict",
      })
    }
    if (value.evidence_gap === 0 && value.evidence_conflicts.length > 0) {
      ctx.addIssue({ code: "custom", message: "A closed evidence gap cannot contain material evidence conflicts" })
    }
  })

/** @internal Exported for structured evidence-gap submission tests. */
export function evidenceGapStructuredOutputFormat(): z.infer<typeof MessageV2.Format> {
  return {
    type: "json_schema",
    schema: z.toJSONSchema(EvidenceGapResponse),
    // Whole assessor calls are retried below so the total remains bounded at two retries.
    retryCount: 0,
  }
}

/** @internal Exported for structured evidence-gap submission tests. */
export function parseEvidenceGapStructuredOutput(value: unknown): z.infer<typeof EvidenceGapResponse> {
  return EvidenceGapResponse.parse(value)
}

const EvidenceSeekingResponse = z
  .object({
    selected_evidence_candidate_ids: z.array(z.string()).max(1),
    rationale: z.string().min(1),
    exhausted: z.boolean(),
  })
  .strict()

const EvidenceTypeSelectionResponse = z
  .object({
    candidate_type: z.enum(["experience", "context", "knowledge"]),
    rationale: z.string().min(1),
  })
  .strict()

// Kept only for the unreachable pre-loop controller body below; skill-guided windows
// return through processEvidenceLoopWindow before that compatibility code executes.
const EvidenceStageNeedResponse = z.object({ use_current_evidence: z.boolean(), rationale: z.string().min(1) })

const EvidenceLoopCheckpoint = z.object({
  version: z.literal(1),
  window_index: z.number().int().nonnegative(),
  state_sha256: z.string().length(64),
  phase: z.enum([
    "ready",
    "seeking",
    "fusing_evidence",
    "generating_prediction",
    "updating",
    "checking",
    "reassessing",
    "safety_reflection",
    "terminated",
  ]),
  selected_candidate_id: z.string().nullable(),
  attempted_candidate_ids: z.array(z.string()),
  seeking_trace: z.array(z.record(z.string(), z.unknown())),
  rejected_transitions: z.array(z.record(z.string(), z.unknown())),
  accepted_transitions: z.number().int().nonnegative(),
  termination_reason: z.string().nullable(),
  predictability: z.enum(["predictable", "unpredictable"]).nullable().default(null),
  updated_at: z.string(),
})

const REOPENABLE_EVIDENCE_LOOP_FAILURES = new Set([
  "provider_failed",
  "schema_failed",
  "response_empty",
  "output_truncated",
])
const INITIAL_EVIDENCE_ASSESSMENT_FAILURE = "The initial evidence-gap assessment failed; additional relevant evidence may still be needed."

/** A cancelled/failed first assessment is pending work, not evidence for an open gap. */
export function needsInitialEvidenceAssessmentRetry(state: z.infer<typeof CriticPredictionState>): boolean {
  return state.evidences.length === 0 && state.evidence_gap === 1 && state.needed_evidence === INITIAL_EVIDENCE_ASSESSMENT_FAILURE
}

function forcedEvidenceLoopDepth(): number | null {
  const raw = process.env.ADACAST_LITE_FORCE_EVIDENCE_LOOP_DEPTH
  if (!raw) return null
  const parsed = Number.parseInt(raw, 10)
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null
}

type EvidenceSelectionMode = "adaptive" | "random" | "fixed_order"

function evidenceSelectionMode(): EvidenceSelectionMode {
  const mode = process.env.ADACAST_LITE_EVIDENCE_SELECTION_MODE
  return mode === "random" || mode === "fixed_order" ? mode : "adaptive"
}

function shouldReopenForcedEvidenceLoopCheckpoint(value: unknown): boolean {
  const target = forcedEvidenceLoopDepth()
  if (!target) return false
  const parsed = EvidenceLoopCheckpoint.safeParse(value)
  if (!parsed.success) return false
  return ["safety_reflection", "terminated"].includes(parsed.data.phase) && parsed.data.accepted_transitions < target
}

/** @internal Exported for durable abnormal-checkpoint resume tests. */
export function shouldReopenAbnormalEvidenceLoopCheckpoint(value: unknown): boolean {
  const parsed = EvidenceLoopCheckpoint.safeParse(value)
  if (!parsed.success) return false
  const checkpoint = parsed.data
  if (!["safety_reflection", "terminated"].includes(checkpoint.phase)) return false
  if (checkpoint.termination_reason && REOPENABLE_EVIDENCE_LOOP_FAILURES.has(checkpoint.termination_reason)) return true
  // Older runs treated failed requests as exhausted evidence. Only reopen the
  // failed candidates still pending in the current committed state; historical
  // failures followed by a successful transition remain valid audit history.
  return ["evidence_exhausted", "no_valid_transition"].includes(checkpoint.termination_reason ?? "") &&
    checkpoint.rejected_transitions.some((record) =>
      typeof record.candidate_id === "string" &&
      checkpoint.attempted_candidate_ids.includes(record.candidate_id) &&
      typeof record.reason === "string" && REOPENABLE_EVIDENCE_LOOP_FAILURES.has(record.reason))
}

async function hasReopenableEvidenceLoopCheckpoint(outDir: string): Promise<boolean> {
  const statesRoot = path.join(outDir, "corrections", "states")
  for (const name of await readdir(statesRoot).catch(() => [])) {
    if (!/^window-\d{6}$/.test(name)) continue
    const statesDir = path.join(statesRoot, name)
    const [pointer, checkpoint] = await Promise.all([
      readJsonFile<unknown>(path.join(statesDir, "current.json")).catch(() => null),
      readJsonFile<unknown>(path.join(statesDir, "loop-checkpoint.json")).catch(() => null),
    ])
    const parsedPointer = CriticStateArtifact.safeParse(pointer)
    const parsedCheckpoint = EvidenceLoopCheckpoint.safeParse(checkpoint)
    if (
      parsedPointer.success &&
      parsedCheckpoint.success &&
      parsedPointer.data.sha256 === parsedCheckpoint.data.state_sha256 &&
      (shouldReopenAbnormalEvidenceLoopCheckpoint(parsedCheckpoint.data) ||
        shouldReopenForcedEvidenceLoopCheckpoint(parsedCheckpoint.data))
    ) {
      return true
    }
  }
  return false
}

const LooseLlmPoint = z.object({
  horizon_index: z.number().int().nonnegative(),
  timestamp: z.string(),
  value: z.number().finite(),
  evidence_ids: z.array(z.string()).optional(),
})

type StructuredResponseSource = "text" | "reasoning" | "tool"

type ParsedStructuredResponse<T> = {
  value: T
  source: StructuredResponseSource
  text: string
}

function jsonCandidates(text: string): unknown[] {
  const sources = [text.trim()]
  for (const match of text.matchAll(/```(?:json)?\s*([\s\S]*?)```/gi)) {
    if (match[1]) sources.push(match[1].trim())
  }

  const stack: string[] = []
  let start = -1
  let inString = false
  let escaped = false
  for (let index = 0; index < text.length; index++) {
    const character = text[index]
    if (start < 0) {
      if (character === "{" || character === "[") {
        start = index
        stack.push(character)
      }
      continue
    }
    if (inString) {
      if (escaped) escaped = false
      else if (character === "\\") escaped = true
      else if (character === '"') inString = false
      continue
    }
    if (character === '"') {
      inString = true
      continue
    }
    if (character === "{" || character === "[") stack.push(character)
    else if (character === "}" || character === "]") {
      const opening = stack.at(-1)
      if ((opening === "{" && character === "}") || (opening === "[" && character === "]")) {
        stack.pop()
        if (stack.length === 0) {
          sources.push(text.slice(start, index + 1))
          start = -1
        }
      } else {
        stack.length = 0
        start = -1
      }
    }
  }

  const parsed: unknown[] = []
  const seen = new Set<string>()
  for (const source of sources) {
    if (!source || seen.has(source)) continue
    seen.add(source)
    try {
      parsed.push(JSON.parse(source))
    } catch {
      // Provider adapters may wrap valid JSON in prose or DSML.
    }
  }
  return parsed
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function normalizeLlmResponse(value: unknown, expectedWindowIndex: number): z.infer<typeof LlmResponse> | null {
  const record = isRecord(value) ? value : null
  const explicitWindowIndex = record?.window_index
  if (
    explicitWindowIndex !== undefined &&
    (!Number.isInteger(explicitWindowIndex) || explicitWindowIndex !== expectedWindowIndex)
  ) {
    return null
  }

  let rawPoints: unknown = Array.isArray(value) ? value : (record?.points ?? record?.predictions)
  const horizonIndices = record?.horizon_index
  const timestamps = record?.timestamp
  const values = record?.value
  if (
    rawPoints === undefined &&
    Array.isArray(horizonIndices) &&
    Array.isArray(timestamps) &&
    Array.isArray(values) &&
    horizonIndices.length === timestamps.length &&
    horizonIndices.length === values.length
  ) {
    rawPoints = horizonIndices.map((horizonIndex, index) => ({
      horizon_index: horizonIndex,
      timestamp: timestamps[index],
      value: values[index],
    }))
  }
  const points = z.array(LooseLlmPoint).safeParse(rawPoints)
  if (!points.success) return null

  const rootEvidence = z.array(z.string()).safeParse(record?.evidence_ids)
  const evidenceIds = new Set(rootEvidence.success ? rootEvidence.data : [])
  for (const point of points.data) {
    for (const evidenceId of point.evidence_ids ?? []) evidenceIds.add(evidenceId)
  }
  const rationale =
    typeof record?.rationale === "string" && record.rationale.trim()
      ? record.rationale.trim()
      : "Normalized from the model's sample-specific forecast response."
  const normalized = LlmResponse.safeParse({
    window_index: expectedWindowIndex,
    points: points.data.map((point) => ({
      horizon_index: point.horizon_index,
      timestamp: point.timestamp,
      value: point.value,
    })),
    rationale,
    evidence_ids: [...evidenceIds],
  })
  return normalized.success ? normalized.data : null
}

/** @internal Exported for provider-response compatibility regression tests. */
export function parseWindowReflectionText(
  text: string,
  expectedWindowIndex: number,
  expectedPointCount?: number,
): z.infer<typeof LlmResponse> {
  let best: z.infer<typeof LlmResponse> | null = null
  for (const candidate of jsonCandidates(text)) {
    const normalized = normalizeLlmResponse(candidate, expectedWindowIndex)
    if (normalized && (!best || normalized.points.length > best.points.length)) best = normalized
  }
  if (best) {
    if (expectedPointCount !== undefined) {
      if (!Number.isInteger(expectedPointCount) || expectedPointCount < 1) {
        throw new Error("Expected prediction point count must be a positive integer")
      }
      if (best.points.length < expectedPointCount) {
        throw new Error(`Prediction response contained ${best.points.length}/${expectedPointCount} required points`)
      }
      if (best.points.length > expectedPointCount) {
        return { ...best, points: best.points.slice(0, expectedPointCount) }
      }
    }
    return best
  }
  throw new Error("Model response did not contain a valid sample-specific forecast JSON object")
}

/** @internal Exported for format-only proposal admission tests. */
export function normalizeProposalLayout(
  points: z.infer<typeof CriticPredictionPoint>[],
  expected: z.infer<typeof CriticPredictionPoint>[],
): z.infer<typeof CriticPredictionPoint>[] {
  if (points.length < expected.length) {
    throw new Error(`Prediction proposal contained ${points.length}/${expected.length} required points`)
  }
  const normalized = points.slice(0, expected.length)
  for (let index = 0; index < expected.length; index++) {
    const actual = normalized[index]
    const layout = expected[index]
    if (!actual || !layout || actual.horizon_index !== layout.horizon_index || actual.timestamp !== layout.timestamp) {
      throw new Error(`Prediction proposal layout differs from the required horizon at position ${index}`)
    }
  }
  return normalized
}

/** @internal Exported for provider response-part compatibility tests. */
export function providerStructuredResponseTexts(
  responseText: string | null,
  responseParts: unknown[] | null,
): Array<{ source: StructuredResponseSource; text: string }> {
  const candidates: Array<{ source: StructuredResponseSource; text: string }> = []
  const seen = new Set<string>()
  const append = (source: StructuredResponseSource, value: unknown) => {
    if (typeof value !== "string") return
    const text = value.trim()
    if (!text || seen.has(text)) return
    seen.add(text)
    candidates.push({ source, text })
  }
  append("text", responseText)
  for (const part of responseParts ?? []) {
    if (isRecord(part) && part.type === "reasoning") append("reasoning", part.text)
  }
  return candidates
}

function parseStructuredResponse<T>(
  responseText: string | null,
  responseParts: unknown[] | null,
  parser: (text: string) => T,
): ParsedStructuredResponse<T> {
  let lastError: unknown = null
  for (const candidate of providerStructuredResponseTexts(responseText, responseParts)) {
    try {
      return { value: parser(candidate.text), ...candidate }
    } catch (error) {
      lastError = error
    }
  }
  if (!responseText?.trim() && providerStructuredResponseTexts(responseText, responseParts).length === 0) {
    throw new WindowReflectionRequestError(
      "Provider returned no text or reasoning response",
      "format",
      true,
      "response_empty",
    )
  }
  throw lastError ?? new Error("Provider response did not contain parseable structured output")
}

/** @internal Exported for evidence-gap contract tests. */
export function parseEvidenceGapText(text: string): z.infer<typeof EvidenceGapResponse> {
  for (const candidate of jsonCandidates(text)) {
    const parsed = EvidenceGapResponse.safeParse(candidate)
    if (parsed.success) return parsed.data
  }
  throw new Error(
    "Evidence-gap response must contain a consistent gap decision, coverage, consistency, confidence, and non-empty rationale",
  )
}

function parseExactEvidenceGapText(text: string): z.infer<typeof EvidenceGapResponse> {
  const value: unknown = JSON.parse(text.trim())
  if (!isRecord(value)) throw new Error("Evidence-gap response must be one JSON object")
  if (value.trace !== undefined && typeof value.trace !== "string") {
    throw new Error("Evidence-gap compatibility trace must be a string")
  }
  const { trace: _compatibilityTrace, ...submission } = value
  return EvidenceGapResponse.parse(submission)
}

/** @internal Exported for structured-output transport compatibility tests. */
export function recoverEvidenceGapSubmission(
  structured: unknown,
  responseText: string | null,
): ParsedStructuredResponse<z.infer<typeof EvidenceGapResponse>> {
  if (structured !== undefined && structured !== null) {
    const value = parseEvidenceGapStructuredOutput(structured)
    return { value, source: "tool", text: JSON.stringify(value) }
  }
  if (!responseText?.trim()) {
    throw new WindowReflectionRequestError(
      "Provider returned no evidence-gap response",
      "format",
      true,
      "response_empty",
    )
  }
  return { value: parseExactEvidenceGapText(responseText), source: "text", text: responseText }
}

/** @internal Exported for evidence-seeking contract tests. */
export function parseEvidenceSeekingText(text: string): z.infer<typeof EvidenceSeekingResponse> {
  for (const candidate of jsonCandidates(text)) {
    const normalized =
      isRecord(candidate) && candidate.exhausted === undefined
        ? {
            ...candidate,
            exhausted:
              Array.isArray(candidate.selected_evidence_candidate_ids) &&
              candidate.selected_evidence_candidate_ids.length === 0,
          }
        : candidate
    const parsed = EvidenceSeekingResponse.safeParse(normalized)
    if (parsed.success) return parsed.data
  }
  throw new Error(
    'Evidence-seeking response must contain only "selected_evidence_candidate_ids", "rationale", and "exhausted"',
  )
}

function parseEvidenceTypeSelectionText(text: string): z.infer<typeof EvidenceTypeSelectionResponse> {
  for (const candidate of jsonCandidates(text)) {
    const parsed = EvidenceTypeSelectionResponse.safeParse(candidate)
    if (parsed.success) return parsed.data
  }
  throw new Error('Evidence type selection must contain only "candidate_type" and "rationale"')
}

type EvidenceCandidateType = "experience" | "context" | "knowledge"
const EVIDENCE_CANDIDATE_PRIORITY: readonly EvidenceCandidateType[] = ["context", "experience", "knowledge"]

/** @internal Exported for adaptive-loop tests. */
export function unusedEvidenceCandidates(
  state: z.infer<typeof CriticPredictionState>,
  evidenceCandidates: unknown,
  attemptedCandidateIds: ReadonlySet<string> = new Set(),
): Partial<Record<EvidenceCandidateType, Record<string, unknown>>> {
  if (!isRecord(evidenceCandidates)) return {}
  const used = new Set(state.evidences.map((item) => item.evidence_candidate_id))
  const result: Partial<Record<EvidenceCandidateType, Record<string, unknown>>> = {}
  for (const candidateType of ["experience", "context", "knowledge"] as const) {
    const candidate = evidenceCandidates[candidateType]
    if (!isRecord(candidate) || candidate.candidate_type !== candidateType) continue
    const id = candidate.evidence_candidate_id
    if (typeof id !== "string" || used.has(id) || attemptedCandidateIds.has(id)) continue
    result[candidateType] = candidate
  }
  return result
}

const EVIDENCE_NEED_TERMS: Record<EvidenceCandidateType, string[]> = {
  experience: [
    "experience",
    "skill",
    "rule",
    "adjustment",
    "residual",
    "validated guidance",
    "经验",
    "技能",
    "规则",
    "调整",
  ],
  context: [
    "context",
    "current-window",
    "time-local",
    "target-day",
    "exogenous",
    "calendar",
    "event",
    "weather",
    "上下文",
    "当前窗口",
    "外生",
    "日历",
    "事件",
    "天气",
  ],
  knowledge: [
    "knowledge",
    "task-level",
    "domain",
    "seasonality",
    "distribution",
    "physical",
    "uncertainty",
    "知识",
    "任务级",
    "领域",
    "季节",
    "分布",
    "物理",
    "不确定性",
  ],
}

/** @internal Exported for deterministic seeker routing tests. */
export function deterministicEvidenceTypeFromNeed(
  neededEvidence: string,
  unused: Partial<Record<EvidenceCandidateType, Record<string, unknown>>>,
): EvidenceCandidateType | null {
  const available = (Object.keys(unused) as EvidenceCandidateType[]).filter((type) => isRecord(unused[type]))
  if (available.length === 0) return null
  if (available.length === 1) return available[0] ?? null
  const need = neededEvidence.toLowerCase()
  const scores = available.map((candidateType) => {
    const candidate = unused[candidateType]
    const id =
      isRecord(candidate) && typeof candidate.evidence_candidate_id === "string"
        ? candidate.evidence_candidate_id.toLowerCase()
        : ""
    const explicitIdMatch = Boolean(id && need.includes(id))
    const termScore = EVIDENCE_NEED_TERMS[candidateType].reduce(
      (score, term) => score + (need.includes(term.toLowerCase()) ? 1 : 0),
      0,
    )
    return { candidateType, explicitIdMatch, termScore }
  })
  const explicitMatches = scores.filter((item) => item.explicitIdMatch)
  const termMatches = scores.filter((item) => item.termScore > 0)
  const eligible = new Set(
    (explicitMatches.length > 0 ? explicitMatches : termMatches).map((item) => item.candidateType),
  )
  if (eligible.size === 0) return null
  return EVIDENCE_CANDIDATE_PRIORITY.find((candidateType) => eligible.has(candidateType)) ?? null
}

const FIXED_EVIDENCE_ORDER: readonly EvidenceCandidateType[] = ["experience", "context", "knowledge"]

/** @internal Exported for evidence selection mode tests. */
export function selectEvidenceTypeByMode(
  unused: Partial<Record<EvidenceCandidateType, Record<string, unknown>>>,
  mode: Exclude<EvidenceSelectionMode, "adaptive">,
  random: () => number = Math.random,
): EvidenceCandidateType | null {
  const available = FIXED_EVIDENCE_ORDER.filter((candidateType) => isRecord(unused[candidateType]))
  if (available.length === 0) return null
  if (mode === "fixed_order") return available[0] ?? null
  const index = Math.min(available.length - 1, Math.max(0, Math.floor(random() * available.length)))
  return available[index] ?? null
}

/** @internal Exported for deterministic seeker-controller tests. */
export function finalizeEvidenceSeekingResponse(
  unused: Partial<Record<EvidenceCandidateType, Record<string, unknown>>>,
  selectedType: EvidenceCandidateType,
  rationale: string,
): z.infer<typeof EvidenceSeekingResponse> {
  const selected = unused[selectedType]
  if (!isRecord(selected) || typeof selected.evidence_candidate_id !== "string") {
    throw new WindowReflectionRequestError(
      `Evidence seeker selected unavailable candidate type ${selectedType}`,
      "format",
      true,
      "schema_failed",
    )
  }
  const remainingCandidateCount = EVIDENCE_CANDIDATE_PRIORITY.filter(
    (candidateType) => candidateType !== selectedType && isRecord(unused[candidateType]),
  ).length
  return {
    selected_evidence_candidate_ids: [selected.evidence_candidate_id],
    rationale,
    exhausted: remainingCandidateCount === 0,
  }
}

/** @internal Exported for adaptive-loop tests. */
export function buildEvidenceSeekingInput(
  state: z.infer<typeof CriticPredictionState>,
  evidenceCandidates: unknown,
  attemptedCandidateIds: ReadonlySet<string> = new Set(),
): { state: z.infer<typeof CriticPredictionState>; unused_evidence_candidates: unknown[] } {
  if (state.evidence_gap === 0) throw new Error("Evidence seeking is forbidden when the gap is closed")
  return {
    state,
    unused_evidence_candidates: Object.values(
      unusedEvidenceCandidates(state, evidenceCandidates, attemptedCandidateIds),
    ),
  }
}

/** @internal Exported for evidence-gap contract tests. */
export function buildEvidenceGapAssessmentInput(
  state: z.infer<typeof CriticPredictionState>,
  evidenceCandidates: unknown,
  lookbackTargetWindow: unknown,
): {
  state: Omit<z.infer<typeof CriticPredictionState>, "evidence_gap" | "needed_evidence">
  committed_evidence_candidates: Record<string, unknown>[]
  unused_evidence_candidate_summaries: Array<{
    evidence_candidate_id: string
    candidate_type: EvidenceCandidateType
    capability_summary: string
  }>
  lookback_window_target: unknown
} {
  const candidates = isRecord(evidenceCandidates) ? evidenceCandidates : {}
  const candidateList = Object.values(candidates).filter(
    (candidate): candidate is Record<string, unknown> =>
      isRecord(candidate) &&
      typeof candidate.evidence_candidate_id === "string" &&
      ["experience", "context", "knowledge"].includes(String(candidate.candidate_type)),
  )
  const committedIds = new Set(state.evidences.map((evidence) => evidence.evidence_candidate_id))
  const committedEvidenceCandidates = candidateList.filter((candidate) =>
    committedIds.has(String(candidate.evidence_candidate_id)),
  )
  if (committedEvidenceCandidates.length !== committedIds.size) {
    throw new Error("Evidence-gap assessment cannot resolve every committed evidence candidate")
  }
  const defaultCapabilities: Record<EvidenceCandidateType, string> = {
    experience: "Historically or validation-derived guidance that may transfer under its stated conditions.",
    context: "Time-local facts and future-known context for the current forecast window.",
    knowledge: "Stable domain or task semantics and scoped empirical structure.",
  }
  const unusedEvidenceCandidateSummaries = candidateList
    .filter((candidate) => !committedIds.has(String(candidate.evidence_candidate_id)))
    .map((candidate) => {
      const candidateType = candidate.candidate_type as EvidenceCandidateType
      const definition =
        typeof candidate.definition === "string" && candidate.definition.trim()
          ? candidate.definition.trim()
          : defaultCapabilities[candidateType]
      const scope = typeof candidate.scope === "string" && candidate.scope.trim() ? candidate.scope.trim() : null
      return {
        evidence_candidate_id: String(candidate.evidence_candidate_id),
        candidate_type: candidateType,
        capability_summary: scope ? `${definition} Declared scope: ${scope}` : definition,
      }
    })
  return {
    state: {
      version: state.version,
      window_index: state.window_index,
      evidences: state.evidences,
      predictions: state.predictions,
    },
    committed_evidence_candidates: committedEvidenceCandidates,
    unused_evidence_candidate_summaries: unusedEvidenceCandidateSummaries,
    lookback_window_target: lookbackTargetWindow,
  }
}

/** @internal Exported for full-lookback contract tests. */
export function requireCompleteLookbackTargetWindow(blindContext: Record<string, unknown>): unknown {
  const lookback = blindContext.visible_history_target_table
  const task = blindContext.task_and_scenario
  if (!isRecord(lookback) || !Array.isArray(lookback.columns) || !Array.isArray(lookback.rows)) {
    throw new Error("Blind context is missing the full lookback target window")
  }
  const expectedSteps = isRecord(task) ? task.history_window_steps : undefined
  if (!Number.isInteger(expectedSteps) || Number(expectedSteps) <= 0) {
    throw new Error("Blind context is missing the declared lookback window length")
  }
  if (lookback.rows.length !== Number(expectedSteps)) {
    throw new Error(
      `Lookback target window is incomplete: expected ${expectedSteps} rows, received ${lookback.rows.length}`,
    )
  }
  return lookback
}

/** @internal Exported for Critic prediction-state tests. */
export function buildContextualCriticPredictionState(
  blindContext: string,
  predictions: z.infer<typeof LlmResponse>["points"],
  previousState?: z.infer<typeof CriticPredictionState>,
): z.infer<typeof CriticPredictionState> {
  const context = JSON.parse(blindContext) as Record<string, unknown>
  const current = previousState ?? CriticPredictionState.parse(context.critic_prediction_state)
  const candidates = isRecord(context.evidence_candidates) ? context.evidence_candidates : null
  const storage = isRecord(context.evidence_candidate_storage) ? context.evidence_candidate_storage : null
  const contextCandidate = candidates && isRecord(candidates.context) ? candidates.context : null
  const contextArtifact = storage && isRecord(storage.context) ? storage.context : null
  if (
    !contextCandidate ||
    contextCandidate.candidate_type !== "context" ||
    typeof contextCandidate.evidence_candidate_id !== "string" ||
    !contextArtifact ||
    typeof contextArtifact.path !== "string" ||
    typeof contextArtifact.sha256 !== "string"
  ) {
    throw new Error("Blind context is missing its classified context evidence candidate")
  }
  const selectedContext = {
    evidence_candidate_id: contextCandidate.evidence_candidate_id,
    candidate_type: "context" as const,
    artifact_path: contextArtifact.path,
    artifact_sha256: contextArtifact.sha256,
  }
  const evidences = current.evidences.filter(
    (item) => item.evidence_candidate_id !== selectedContext.evidence_candidate_id,
  )
  evidences.push(selectedContext)
  return CriticPredictionState.parse({
    ...current,
    evidences,
    predictions,
    evidence_gap: 1,
    needed_evidence: "",
  })
}

/** @internal Exported for state-bound prompt tests. */
export function bindCriticStateToBlindContext(
  blindContext: string,
  state: z.infer<typeof CriticPredictionState>,
  stateArtifact: { path: string; sha256: string },
): string {
  const context = JSON.parse(blindContext) as Record<string, unknown>
  const byHorizon = new Map(state.predictions.map((point) => [point.horizon_index, point]))
  if (Array.isArray(context.prediction_points)) {
    context.prediction_points = context.prediction_points.map((item) => {
      if (!isRecord(item) || typeof item.horizon_index !== "number") return item
      const prediction = byHorizon.get(item.horizon_index)
      return prediction ? { ...item, pre_reflection_baseline: prediction.value } : item
    })
  }
  context.critic_prediction_state = state
  context.critic_prediction_state_storage = stateArtifact
  const experienceUsed = state.evidences.some((evidence) => evidence.candidate_type === "experience")
  context.evidence_stage_status = {
    experience_used: experienceUsed,
    current_prediction_semantics: experienceUsed
      ? "Current predictions include selected experience evidence."
      : "Current predictions do not include experience evidence; do not use deterministic experience or its proposal.",
  }
  if (isRecord(context.instructions)) {
    context.instructions = {
      ...context.instructions,
      baseline_semantics: experienceUsed
        ? "The current state prediction already includes selected experience evidence. Do not apply it again."
        : "The current state prediction is the raw prior. Deterministic experience is an unselected candidate and must not influence this stage.",
    }
  }
  return JSON.stringify(context)
}

const PreparedResult = z.object({
  prepared_path: z.string().min(1),
  trigger_count: z.number().int().nonnegative(),
  skill_match_count: z.number().int().nonnegative(),
  llm_required_count: z.number().int().nonnegative(),
  windows: z.array(
    z.object({
      window_index: z.number().int().nonnegative(),
      detection: z.object({
        trigger: z.boolean(),
        hard_violations: z.array(z.unknown()),
        soft_warnings: z.array(z.unknown()),
      }),
      skill_matched: z.boolean(),
      llm_required: z.boolean(),
      active_recalibration_priority: z.boolean().default(false),
      matched_window_ordinal: z.number().int().positive().nullable().default(null),
      minimum_meaningful_delta_original: z.number().nonnegative().default(0),
      minimum_meaningful_points: z.number().int().nonnegative().default(0),
      reasoning_mode: z.enum(["safety_reflection", "skill_guided_test", "evidence_loop_test"]),
      failure_fallback: z.enum(["pre_reflection_baseline", "raw_anchor"]),
      horizon_indices: z.array(z.number().int().nonnegative()),
      timestamps: z.array(z.string()),
      blind_context_path: z.string().nullable(),
      blind_context_sha256: z.string().length(64).nullable().default(null),
      evidence_candidate_artifacts: z
        .object({
          experience: z.object({ path: z.string().min(1), sha256: z.string().length(64) }).optional(),
          context: z.object({ path: z.string().min(1), sha256: z.string().length(64) }).optional(),
          knowledge: z.object({ path: z.string().min(1), sha256: z.string().length(64) }).optional(),
        })
        .default({}),
      evidence_candidate_ids: z
        .object({
          experience: z.string().min(1).optional(),
          context: z.string().min(1).optional(),
          knowledge: z.string().min(1).optional(),
        })
        .default({}),
      critic_prediction_state_artifact: z
        .object({ path: z.string().min(1), sha256: z.string().length(64) })
        .nullable()
        .default(null),
      critic_prediction_state_history: z
        .object({
          initial: z.object({ path: z.string().min(1), sha256: z.string().length(64) }).optional(),
          deterministic: z.object({ path: z.string().min(1), sha256: z.string().length(64) }).optional(),
        })
        .default({}),
      experience_rule_reference_artifact: z
        .object({ path: z.string().min(1), sha256: z.string().length(64) })
        .nullable()
        .default(null),
    }),
  ),
})
type PreparedResult = z.infer<typeof PreparedResult>

const EvidenceLoopTerminationReason = z.enum([
  "evidence_sufficient",
  "evidence_exhausted",
  "max_iterations",
  "no_valid_transition",
  "provider_failed",
  "schema_failed",
  "response_empty",
  "output_truncated",
  "safety_rejected",
])
type EvidenceLoopTerminationReason = z.infer<typeof EvidenceLoopTerminationReason>
type EvidenceLoopPredictability = "predictable" | "unpredictable"

const CheckResult = z.object({
  accepted: z.boolean(),
  format_retry: z.boolean(),
  format_errors: z.array(z.string()),
  detector_report: z.unknown(),
  points: z
    .array(
      z.object({
        horizon_index: z.number().int().nonnegative(),
        timestamp: z.string(),
        value: z.number().finite(),
      }),
    )
    .optional(),
})
const ReflectionExecutionManifest = z.object({
  version: z.literal(1),
  signature_sha256: z.string().length(64),
  prepared_sha256: z.string().length(64),
  status: z.enum(["running", "completed", "failed"]),
  started_at: z.string(),
  updated_at: z.string(),
})
type ReflectionExecutionManifest = z.infer<typeof ReflectionExecutionManifest>

export const WINDOW_REFLECTION_REASONING_TOKEN_GUIDANCE = 20_000
export const WINDOW_REFLECTION_PREDICTION_OUTPUT_TOKEN_LIMIT = 65_536

export function windowReflectionPredictionOutputTokenLimit(): number {
  return Math.min(WINDOW_REFLECTION_PREDICTION_OUTPUT_TOKEN_LIMIT, ProviderTransform.OUTPUT_TOKEN_MAX)
}

const PREDICTION_REASONING_LENGTH_INSTRUCTION =
  "Keep provider-visible reasoning under approximately 20,000 tokens and reserve enough output space for the complete prediction JSON."
const ROW_ORIENTED_PREDICTION_OUTPUT_INSTRUCTION =
  'Return only one JSON object with keys "window_index", "points", "rationale", and "evidence_ids"; each point must contain only "horizon_index", "timestamp", and "value".'

const ReflectionResult = ForecastWindowReflectionArtifact.extend({
  result_path: z.string().min(1),
})
type ReflectionResult = z.infer<typeof ReflectionResult>

const ISOLATED_SYSTEM = [
  "This is a blind, isolated final safety repair.",
  "Never request or use future observed values, residuals, metrics, later windows, or parent conversation.",
  "The supplied pre-reflection baseline is the immutable fallback.",
  "On later rounds, conservatively repair the supplied previous rejected proposal using only its deterministic report.",
  "Make the smallest evidence-supported change that clears the deterministic report.",
  "Return every required indexed point exactly once in original dataset units.",
  PREDICTION_REASONING_LENGTH_INSTRUCTION,
  ROW_ORIENTED_PREDICTION_OUTPUT_INSTRUCTION,
  "Do not use Markdown fences, XML, DSML, tool-call syntax, commentary, a predictions alias, or column-oriented arrays.",
].join("\n")

// Retained only to validate signatures created by pre-removal checkpoints.
const LEGACY_POST_LOOP_SAFETY_SYSTEM = [
  "This safety-reflection stage runs only after the adaptive evidence loop has terminated.",
  "Treat critic_prediction_state.predictions as the only prediction baseline and immutable fallback.",
  "Use only evidence already listed in critic_prediction_state.evidences and committed_evidence_candidates.",
  "Never introduce, infer, cite, or apply an unselected evidence candidate.",
  "Use the supplied frozen detector report only to make the smallest repair that clears its violations.",
  "Do not change evidence membership, evidence_gap, or needed_evidence; return prediction points only.",
  "Never request or use future observed values, residuals, metrics, later windows, or parent conversation.",
  "Return every required indexed point exactly once in original dataset units.",
  PREDICTION_REASONING_LENGTH_INSTRUCTION,
  ROW_ORIENTED_PREDICTION_OUTPUT_INSTRUCTION,
  "Do not use Markdown fences, XML, DSML, tool-call syntax, commentary, a predictions alias, or column-oriented arrays.",
].join("\n")

/** @internal Exported for prompt-contract tests. */
export const SKILL_GUIDED_TEST_SYSTEM = [
  "This is blind, isolated, sample-specific generative forecast reasoning for a test window matched by an accepted skill.",
  "Never request or use observed target values in the forecast gap or target horizon, residuals, metrics, later windows, or parent conversation.",
  "Tabular context objects declare columns once and encode rows positionally; map every row value to the same-index column and never reorder columns.",
  "Use the full-resolution exogenous table for yesterday, forecast issue day today, and target day tomorrow, together with full-sample summaries and the losslessly deduplicated daily-constant table.",
  "Your objective is to improve the expected forecast accuracy of this sample from visible evidence, not merely to clear a detector threshold or demonstrate that a change was made.",
  "Treat critic_prediction_state.predictions as the only current prediction. Read evidence_stage_status before using experience: when experience_used=false, deterministic experience and its proposal are unselected candidates and must not influence the update; when true, never reapply, sum, net, or stack their actions.",
  "Treat the supplied first expert-derived skill as an important reason to inspect this sample and as the primary hypothesis, not as a numerical template or the whole context. Consider it explicitly and cite only matched rules, but independently re-estimate direction, magnitude, timing, and shape from this sample's visible evidence.",
  "Treat evidence_candidates.experience, evidence_candidates.context, and evidence_candidates.knowledge as distinct evidence families with different scopes; never promote current context into knowledge or treat validation experience as a current observation.",
  "Knowledge may include scoped qualitative and quantitative data analysis such as trend, seasonality, stationarity, volatility, missingness, anomalies, and distribution shifts. Preserve its analysis scope and uncertainty; model residuals, rankings, validation gains, and correction magnitudes remain experience.",
  "Start with current_sample_exogenous_context. Review the target-day joint context, changes from yesterday and forecast-issue day, forecast-gap dynamics, and every secondary_context_column. Group secondary factors as supporting, opposing, interacting, or neutral; do not ignore them because the skill does not mention them.",
  "Use all supplied exogenous families together when present: dynamic weather trajectories, temperature levels and changes, humidity, radiation, daily weather, wind, sunshine, extreme and overlap flags, environmental control, policy, and calendar information. Correlated, duplicated, or derived measurements of one physical driver count as one evidence family, not multiple votes.",
  "Compare the current state prediction with a context-derived profile. A deterministic proposal may be considered only when experience_used=true. Calibrate morning-ramp, midday, evening-peak, and late-day segments separately before composing one smooth curve.",
  "Before changing a segment, require at least two independent evidence families, and at least one must come from the current sample's target-day exogenous changes or forecast-gap dynamics. Skill tags and their matching rules count as one family. Historical observations may only corroborate a conclusion already supported by the current sample; they cannot supply correction direction or magnitude by themselves.",
  "Do not search for, nominate, or copy one closest historical day. Past observed days are secondary distribution or shape cross-checks, not historical model residuals. Never subtract a past observed value from the target-window forecast and interpret that difference as forecast bias, and never calibrate magnitude from one convenient day.",
  "Do not infer correction direction from labels such as holiday, restriction, release, rain, snow, extreme, warm, or cold alone. In load and energy tasks weather effects can be nonlinear or oppose a calendar effect; use the supplied numeric trajectory and genuinely comparable days to resolve direction.",
  "The raw anchor may already encode ordinary trend, calendar, and exogenous effects, so never apply the same effect a second time. For electric-load targets, cold, rain, or snow is not evidence for lower demand by itself and may sustain or increase heating and lighting load; absent quantitative task evidence, never use those labels to amplify a downward correction. Use exogenous conditions only to estimate the residual effect not already represented by the raw anchor.",
  "Rule bounds are safety limits, not recommended adjustment magnitudes. Never move to a minimum or maximum bound merely because more qualitative drivers point in the same direction. A full coherent correction means consistent temporal coverage over an evidenced segment, not a larger amplitude.",
  "The deterministic baseline may legitimately include the combined result of multiple matched rules. Never undo, halve, or replace that combined effect merely because tags co-occur or sound related. Treat tags as duplicate descriptions only when the supplied deterministic evidence establishes the same underlying event and independent quantitative sample evidence supports reducing the combined magnitude.",
  "When rules overlap or point in different directions, compare the already-combined deterministic baseline with the raw and context-derived candidates; do not reconstruct it by arithmetic. Explain which quantitative evidence makes the selected profile more plausible than both anchors.",
  "Read active_recalibration from the blind context. When priority=true, this window belongs to the leakage-free minimum 20% active-recalibration subset: produce a genuinely sample-specific curve meeting its relative delta and point-count contract instead of copying the deterministic baseline or making a token-sized edit. Use the smallest coherent localized adjustment that satisfies the contract and is supported by the current joint exogenous context; do not enlarge it merely to appear active.",
  "Detector limits are safety diagnostics, not desired curve statistics. A soft warning alone is not sufficient reason to change values; never create an isolated spike or trough merely to cross a flat-window threshold. If a hard violation must be cleared, use an evidence-supported smooth shape rather than targeting the boundary numerically.",
  "Prefer coherent changes over contiguous load periods such as ramps or peaks. Use isolated point changes only when the supplied evidence is itself localized to that timestamp.",
  "Only after current-sample analysis may multiple independently comparable historical periods be used as a weak consistency check. Do not assume the immediately preceding day is an analogue, and do not rank or name a single closest day.",
  "In the rationale, first state the joint target-day context, then list skill-primary evidence, secondary supporting and opposing or neutral evidence, and the residual effect not already encoded by the raw anchor. Finish with each segment's chosen delta relative to both raw and deterministic anchors. If evidence is insufficient or contradictory, keep the better-supported anchor; never make a token-sized activity edit.",
  "On later rounds, repair the previous rejected proposal using only its deterministic report and the same frozen context.",
  "Return every required indexed point exactly once in original dataset units and cite the exact evidence candidate ids used in evidence_ids.",
  PREDICTION_REASONING_LENGTH_INSTRUCTION,
  ROW_ORIENTED_PREDICTION_OUTPUT_INSTRUCTION,
  "Do not use Markdown fences, XML, DSML, tool-call syntax, commentary, a predictions alias, or column-oriented arrays.",
].join("\n")

export const EVIDENCE_GAP_SYSTEM = [
  "Judge whether the current prediction needs further evidence-guided correction or is already sufficiently credible to keep unchanged.",
  "Use only the supplied state, committed_evidence_candidates, unused_evidence_candidate_summaries, and the complete observed-target lookback window.",
  "committed_evidence_candidates contains the full content of evidence already used in the prediction. Only this content may support claims introduced by an evidence-guided correction; an empty committed set does not disqualify an unchanged prior that passes the initial-prior check.",
  "unused_evidence_candidate_summaries contains only candidate identity, type, and capability. Use it only to name what could address a gap; never treat a summary as evidence, infer its hidden payload, or let it increase coverage, consistency, or confidence.",
  "Use lookback_window_target as historical shape and level context only; never treat it as forecast-horizon truth or residual evidence.",
  "Treat current-window context as optional rather than presumptively relevant. Future-known exogenous variables can address a prediction gap only when committed evidence or scoped knowledge establishes a credible, sufficiently strong, and stable relationship between those variables and the target decision under review.",
  "The availability of exogenous variables does not itself create an evidence gap or make context the preferred next candidate. If their target association is known to be weak, unstable, indirect, or unsupported by a quantitative mapping, do not list missing context merely to inspect them and do not require context evidence for closure.",
  "When the exogenous-target relationship is unknown, request evidence capable of establishing that relationship only if it could materially change a specific forecast decision; never assume that selecting context will be informative. State the target segment, proposed relationship, and expected decision impact whenever needed_evidence names context or an exogenous variable.",
  "The prediction prior is a candidate no-change baseline, not the default accepted answer. Some samples have a strongly supported prior and need no evidence-guided repair, but the assessor must establish that support explicitly.",
  "When committed_evidence_candidates is empty, perform an explicit initial-prior check across level, temporal shape, direction, magnitude, timing, boundary continuity, regime behavior, and uncertainty using the complete lookback context.",
  "Accept the initial prior without evidence search only when it is strongly coherent across those dimensions, contains no unexplained material regime change or segment-level deviation, and no available evidence type could materially confirm or reject an important forecast decision.",
  "Plausibility alone is insufficient for immediate acceptance. If the prior is merely plausible but an important level, direction, magnitude, timing, boundary, or regime decision remains weakly supported, set evidence_gap=1 when additional evidence could clarify it.",
  "Absence of committed evidence or existence of unused candidates is not by itself a gap. However, open a gap when the guided assessment finds a credible, decision-relevant uncertainty, weakly supported aspect, plausible anomaly, or conflict that could materially affect the forecast and that additional evidence could clarify. You do not need to prove that the prior is defective before seeking evidence; explain the specific concern and why resolving it could change whether the prior should be kept.",
  "Evidence search does not require changing the prediction. After examining relevant evidence, keeping the original prior unchanged is a valid outcome when it becomes the best-supported decision.",
  "For an accepted initial prior, evidence_coverage=1 means no material correction-relevant support gap was identified; it does not claim perfect pointwise knowledge of the hidden forecast horizon. Set missing_evidence and evidence_conflicts empty, set evidence_consistency=1, and calibrate evidence_confidence to the strength of the no-change conclusion.",
  "Do not change evidences or predictions.",
  "Follow this guided assessment before deciding the gap:",
  "1. Inventory only committed_evidence_candidates and identify the prediction claims and horizon segments each committed item actually supports.",
  "2. Assess coverage across target level, temporal shape, direction, magnitude, timing or boundaries, and uncertainty. Mark a dimension relevant only when the prediction or committed evidence makes it relevant. Do not count lookback context or unused summaries as committed support.",
  "3. Set evidence_coverage from 0 to 1 as the proportion of relevant support dimensions adequately covered. Do not reward duplicate or dependent evidence twice. List every material unsupported claim in missing_evidence.",
  "4. Assess consistency among committed evidence items and between each item, its declared scope, and the prediction it is claimed to support. Identify agreement, duplication, conditional compatibility, and contradiction. List every material contradiction or scope mismatch in evidence_conflicts, then set evidence_consistency from 0 to 1; 1 means no material conflict and lower values mean stronger unresolved conflict.",
  "5. Set evidence_confidence from 0 to 1 for how strongly the committed evidence supports the prediction as a whole. Base confidence on coverage, consistency, directness, evidence quality, scope match, and uncertainty. Lower it for indirect, conflicting, duplicated, weak, or extrapolated support. Unused summaries must not raise it.",
  "6. Set evidence_gap=0 when no material missing item or conflict remains and the current coverage, consistency, and confidence are jointly sufficient for keeping this prediction, including an already-good initial prior; otherwise set evidence_gap=1.",
  "Make needed_evidence a concise synthesis of missing_evidence and the support needed to resolve evidence_conflicts when the gap is open, or an empty string when closed.",
  "In rationale, separately explain coverage, consistency, and confidence, then state the remaining gaps and why they imply the final gap value.",
  "Submit the assessment exactly once through the required StructuredOutput tool. Do not return the assessment as plain text and do not update Critic state.",
  'Closed initial-prior example: {"evidence_gap":0,"needed_evidence":"","evidence_coverage":1,"missing_evidence":[],"evidence_consistency":1,"evidence_conflicts":[],"evidence_confidence":0.85,"rationale":"The unchanged prior is coherent with the visible history and has no material correction need."}.',
  'Open tool-argument example: {"evidence_gap":1,"needed_evidence":"concise synthesis","evidence_coverage":0.5,"missing_evidence":["material missing support"],"evidence_consistency":1,"evidence_conflicts":[],"evidence_confidence":0.4,"rationale":"complete guided assessment"}.',
].join("\n")

export const EVIDENCE_SEEKING_SYSTEM = [
  "Choose exactly one evidence candidate type when deterministic matching found no match for needed_evidence.",
  "Choose only from available_candidate_types and do not modify the state or prediction.",
  "The controller maps the selected type to the prepared candidate and computes whether candidates are exhausted.",
  'Return only {"candidate_type":"experience|context|knowledge","rationale":"concise reason"}.',
].join("\n")

export const UNIFIED_REASONING_SYSTEM = [
  "Act as the UnifiedEvidenceReasoner: generate one complete prediction proposal from the current Critic prediction and the provisional fused evidence set.",
  "The provisional fusion contains all evidence already committed to the current prediction plus exactly one newly selected evidence candidate. Treat it as a temporary reasoning view, not a committed state transition.",
  "Evidence adapters preserve type-specific semantics: experience may include an initial-based rule reference, context contains current-window facts, and knowledge contains stable scoped semantics. Jointly reason over their support, overlap, scope, dependencies, and conflicts instead of applying them as independent votes.",
  "When context evidence is fused, use its complete row-level lookback_window_exogenous and predict_window_exogenous tables as the primary context. Feature summaries are auxiliary only and must never substitute for, truncate, or override those tables.",
  "Use the complete lookback_window_target to ground historical target level, shape, ramps, and recurring boundaries.",
  "Treat critic_prediction_state.predictions as the only transition baseline. Estimate only effects not already represented in that prediction, and preserve the current values wherever the fused evidence does not justify a residual numerical change.",
  "An experience rule reference shows how validated rules act on the immutable initial prediction. It is reference evidence, not a target forecast or state transition. Never mechanically add its initial-to-reference delta to the current prediction.",
  "Context is time-local: never generalize it into a reusable rule. Knowledge is scoped and stable: never turn it into an unobserved current-window fact.",
  "Use no candidate outside provisional_evidence_fusion. The newly selected candidate id must be cited; cite committed evidence ids only when they materially support the proposal.",
  "Generate prediction points only. Never mutate, return, or claim to commit Critic state, evidences, evidence_gap, or needed_evidence.",
  "Never use observed targets in the forecast gap or horizon, residuals, metrics, later windows, or parent conversation.",
  "Keep every horizon index and timestamp exactly once and return finite values in original dataset units.",
  "Make rationale a concise auditable account of each cited evidence item, material conflicts or overlap, and segment-level effects; keep it under 1200 characters.",
  PREDICTION_REASONING_LENGTH_INSTRUCTION,
  ROW_ORIENTED_PREDICTION_OUTPUT_INSTRUCTION,
  "Do not use Markdown fences, XML, DSML, tool-call syntax, commentary, a predictions alias, or column-oriented arrays.",
].join("\n")

const CONTENT_RISK_LABEL_ALIASES: ReadonlyArray<readonly [string, string]> = [
  ["\u5317\u4eac\u5e02_t2m", "beijing_air_temperature_2m"],
  ["\u5317\u4eac\u5e02_q", "beijing_specific_humidity"],
  ["\u5317\u4eac\u5e02_ssrd", "beijing_surface_solar_radiation"],
  ["\u524d\u4e00\u65e5\u6700\u9ad8\u6e29", "previous_daily_high_temperature"],
  ["\u524d\u4e00\u65e5\u6700\u4f4e\u6e29", "previous_daily_low_temperature"],
  ["\u6700\u9ad8\u6e29\u53d8\u5316", "daily_high_temperature_change"],
  ["\u6700\u4f4e\u6e29\u53d8\u5316", "daily_low_temperature_change"],
  ["\u6700\u5927\u6e29\u5dee\u7edd\u5bf9\u503c", "maximum_temperature_change_absolute"],
  ["\u6700\u9ad8\u6e29_\u6570\u503c", "daily_high_temperature_numeric"],
  ["\u6700\u4f4e\u6e29_\u6570\u503c", "daily_low_temperature_numeric"],
  ["\u6700\u9ad8\u6e29", "daily_high_temperature"],
  ["\u6700\u4f4e\u6e29", "daily_low_temperature"],
  ["\u524d\u4e00\u65e5\u5929\u6c14", "previous_day_weather"],
  ["\u662f\u5426\u6781\u7aef\u5929\u6c14", "extreme_weather_flag"],
  ["\u662f\u5426\u91cd\u5408", "event_overlap_flag"],
  ["\u98ce\u5411\u98ce\u529b", "wind_category"],
  ["\u5929\u6c14", "weather_category"],
  ["\u65e5\u7167", "sunshine_category"],
  ["\u653f\u7b56", "event_signal"],
  ["\u73af\u4fdd", "control_event_signal"],
  ["\u8282\u5047\u65e5", "calendar_event_signal"],
]

function contentRiskSafeLabel(value: string): string {
  return CONTENT_RISK_LABEL_ALIASES.reduce(
    (current, [source, replacement]) => current.replaceAll(source, replacement),
    value,
  )
}

function contentRiskSafeClone(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(contentRiskSafeClone)
  if (!value || typeof value !== "object") {
    return typeof value === "string" ? contentRiskSafeLabel(value) : value
  }
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>).map(([key, nested]) => [
      contentRiskSafeLabel(key),
      contentRiskSafeClone(nested),
    ]),
  )
}

/** @internal Exported for provider-content-risk regression tests. */
export function buildProviderContentRiskSafeContext(blindContext: string): string {
  const parsed = JSON.parse(blindContext) as Record<string, unknown>
  const projected = contentRiskSafeClone(parsed) as Record<string, unknown>
  const task = projected.task_and_scenario
  if (task && typeof task === "object" && !Array.isArray(task)) {
    const source = task as Record<string, unknown>
    projected.task_and_scenario = {
      task_type: "regional_load_forecast",
      target_column: source.target_column,
      time_column: source.time_column,
      history_window_steps: source.history_window_steps,
      forecast_gap_steps: source.forecast_gap_steps,
      prediction_window_steps: source.prediction_window_steps,
      window_stride_steps: source.window_stride_steps,
      evidence_scope:
        "Use visible target history, forecast-gap model predictions, target-window model predictions, and future-known exogenous values.",
    }
  }

  const evidenceCandidates = projected.evidence_candidates
  if (evidenceCandidates && typeof evidenceCandidates === "object" && !Array.isArray(evidenceCandidates)) {
    const candidates = evidenceCandidates as Record<string, unknown>
    const knowledge = candidates.knowledge
    if (knowledge && typeof knowledge === "object" && !Array.isArray(knowledge)) {
      const knowledgeRecord = knowledge as Record<string, unknown>
      const payload = knowledgeRecord.payload
      if (payload && typeof payload === "object" && !Array.isArray(payload)) {
        const payloadRecord = payload as Record<string, unknown>
        const semantics = payloadRecord.task_semantics_and_structure
        if (semantics && typeof semantics === "object" && !Array.isArray(semantics)) {
          const source = semantics as Record<string, unknown>
          const dataAnalysis = source.data_analysis
          let safeDataAnalysis: unknown = dataAnalysis
          if (dataAnalysis && typeof dataAnalysis === "object" && !Array.isArray(dataAnalysis)) {
            const analysisRecord = structuredClone(dataAnalysis as Record<string, unknown>)
            const qualitative = analysisRecord.qualitative_analysis
            if (qualitative && typeof qualitative === "object" && !Array.isArray(qualitative)) {
              const qualitativeRecord = qualitative as Record<string, unknown>
              analysisRecord.qualitative_analysis = {
                source_path: qualitativeRecord.source_path,
                source_sha256: qualitativeRecord.source_sha256,
                truncated: qualitativeRecord.truncated,
                usage_constraint: qualitativeRecord.usage_constraint,
                content_withheld_for_provider_retry: true,
              }
            }
            safeDataAnalysis = analysisRecord
          }
          payloadRecord.task_semantics_and_structure = {
            task_type: "regional_load_forecast",
            target_column: source.target_column,
            time_column: source.time_column,
            history_window_steps: source.history_window_steps,
            forecast_gap_steps: source.forecast_gap_steps,
            prediction_window_steps: source.prediction_window_steps,
            window_stride_steps: source.window_stride_steps,
            data_analysis: safeDataAnalysis,
          }
        }
      }
    }
  }

  const features = projected.deterministic_sample_features
  if (features && typeof features === "object" && !Array.isArray(features)) {
    const featureRecord = features as Record<string, unknown>
    const evidence = featureRecord.rule_match_evidence_rows
    if (Array.isArray(evidence)) {
      const grouped = new Map<string, { rule_id: string; matched_horizon_indices: number[]; conditions: unknown[] }>()
      const seenConditions = new Map<string, Set<string>>()
      for (const item of evidence) {
        if (!item || typeof item !== "object" || Array.isArray(item)) continue
        const row = item as Record<string, unknown>
        const ruleId = typeof row.rule_id === "string" ? row.rule_id : "unknown"
        const summary = grouped.get(ruleId) ?? { rule_id: ruleId, matched_horizon_indices: [], conditions: [] }
        if (typeof row.horizon_index === "number") summary.matched_horizon_indices.push(row.horizon_index)
        if (row.condition_exogenous_values !== undefined) {
          const serialized = JSON.stringify(row.condition_exogenous_values)
          const seen = seenConditions.get(ruleId) ?? new Set<string>()
          if (!seen.has(serialized)) {
            seen.add(serialized)
            summary.conditions.push(row.condition_exogenous_values)
          }
          seenConditions.set(ruleId, seen)
        }
        grouped.set(ruleId, summary)
      }
      featureRecord.rule_match_evidence_summary = [...grouped.values()]
      delete featureRecord.rule_match_evidence_rows
    }
  }

  projected.provider_content_risk_projection = {
    applied: true,
    scope: "provider_retry_only",
    numeric_forecasting_evidence_preserved: true,
    free_form_scenario_prose_removed: true,
    repeated_rule_match_rows_collapsed: true,
    semantic_aliases_applied: CONTENT_RISK_LABEL_ALIASES.map(([, alias]) => alias),
  }
  return JSON.stringify(projected)
}

function packObjectRows(value: unknown, fields: string[]): unknown {
  if (!Array.isArray(value)) return value
  return {
    columns: fields,
    rows: value.map((item) => {
      const record = item && typeof item === "object" && !Array.isArray(item) ? (item as Record<string, unknown>) : {}
      return fields.map((field) => record[field])
    }),
  }
}

/** @internal Exported for provider-content-risk regression tests. */
export function buildProviderContentRiskCompactContext(blindContext: string): string {
  const projected = JSON.parse(buildProviderContentRiskSafeContext(blindContext)) as Record<string, unknown>
  projected.gap_prediction_table = packObjectRows(projected.gap_prediction_points, [
    "gap_index",
    "timestamp",
    "intermediate_prediction",
  ])
  projected.prediction_table = packObjectRows(projected.prediction_points, [
    "horizon_index",
    "timestamp",
    "raw_anchor",
    "pre_reflection_baseline",
  ])
  delete projected.gap_prediction_points
  delete projected.prediction_points
  const representation = projected.context_representation
  if (representation && typeof representation === "object" && !Array.isArray(representation)) {
    const source = representation as Record<string, unknown>
    projected.context_representation = {
      table_contract: source.table_contract,
      recent_exogenous_date_roles: source.recent_exogenous_date_roles,
    }
  }
  projected.instructions = {
    objective: "Choose the most accurate evidence-supported sample profile without using hidden targets.",
    calibration:
      "Keep experience, context, and knowledge evidence candidates distinct. Use the current sample's joint exogenous context and forecast-gap dynamics as primary segment-level evidence. Review every secondary exogenous column. Past observations are weak cross-checks, not model residuals or magnitude templates; semantic tags and bounds do not determine magnitude.",
    output: "Return every required horizon index and exact timestamp once in original units.",
  }
  projected.provider_content_risk_projection = {
    applied: true,
    level: "compact_semantic",
    scope: "provider_retry_only",
    numeric_forecasting_evidence_preserved: true,
    tabular_rows_preserved: true,
    object_rows_packed: true,
    free_form_scenario_prose_removed: true,
    repeated_rule_match_rows_collapsed: true,
    semantic_aliases_applied: CONTENT_RISK_LABEL_ALIASES.map(([, alias]) => alias),
  }
  return JSON.stringify(projected)
}

export function isWindowReflectionContentRisk(message: string): boolean {
  return message.toLowerCase().includes("content exists risk")
}

class WindowReflectionRequestError extends Error {
  constructor(
    message: string,
    readonly category: "provider" | "format",
    readonly retryable: boolean,
    readonly failureKind: WindowReflectionFailureKind = category === "provider" ? "provider_failed" : "schema_failed",
  ) {
    super(message)
    this.name = "WindowReflectionRequestError"
  }
}

type WindowReflectionFailureKind = "provider_failed" | "schema_failed" | "response_empty" | "output_truncated"

function responseWasTruncated(responseParts: unknown[] | null): boolean {
  return (responseParts ?? []).some((part) => isRecord(part) && part.type === "step-finish" && part.reason === "length")
}

function structuredResponseError(
  error: unknown,
  responseText: string | null,
  responseParts: unknown[] | null,
): WindowReflectionRequestError {
  if (error instanceof WindowReflectionRequestError) return error
  const message = error instanceof Error ? error.message : String(error)
  if (responseWasTruncated(responseParts)) {
    return new WindowReflectionRequestError(message, "format", true, "output_truncated")
  }
  if (!responseText?.trim() && providerStructuredResponseTexts(responseText, responseParts).length === 0) {
    return new WindowReflectionRequestError(
      message || "Provider returned an empty response",
      "format",
      true,
      "response_empty",
    )
  }
  return new WindowReflectionRequestError(message, "format", true, "schema_failed")
}

function evidenceLoopFailureReason(error: unknown): WindowReflectionFailureKind {
  return error instanceof WindowReflectionRequestError ? error.failureKind : "provider_failed"
}

/** @internal Exported for timeout and cancellation tests. */
export async function withWindowReflectionTimeout<T>(
  operation: Promise<T>,
  options: { timeoutMs: number; label: string; onTimeout?: () => void | Promise<void> },
): Promise<T> {
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs <= 0) {
    throw new Error("Window reflection timeout must be a positive finite number")
  }
  let timeout: ReturnType<typeof setTimeout> | undefined
  return Promise.race([
    operation,
    new Promise<never>((_, reject) => {
      timeout = setTimeout(() => {
        void Promise.resolve(options.onTimeout?.()).catch(() => undefined)
        reject(
          new WindowReflectionRequestError(`${options.label} timed out after ${options.timeoutMs}ms`, "provider", true),
        )
      }, options.timeoutMs)
    }),
  ]).finally(() => {
    if (timeout) clearTimeout(timeout)
  })
}

export function classifyWindowReflectionLlmError(error: unknown): {
  category: "provider" | "format"
  retryable: boolean
  message: string
  failureKind: WindowReflectionFailureKind
} {
  if (!error || typeof error !== "object") {
    return {
      category: "format",
      retryable: true,
      message: "Unknown isolated LLM response error",
      failureKind: "schema_failed",
    }
  }
  const record = error as { name?: unknown; data?: unknown; message?: unknown }
  const data =
    record.data && typeof record.data === "object"
      ? (record.data as { message?: unknown; isRetryable?: unknown })
      : null
  const name = typeof record.name === "string" ? record.name : "ProviderError"
  const message =
    typeof data?.message === "string"
      ? data.message
      : typeof record.message === "string"
        ? record.message
        : "Isolated LLM provider request failed"
  if (name === "StructuredOutputError" || name === "OutputLengthError" || name === "ZodError") {
    return {
      category: "format",
      retryable: true,
      message: `${name}: ${message}`,
      failureKind: name === "OutputLengthError" ? "output_truncated" : "schema_failed",
    }
  }
  if (name === "APIError" || name === "ContextOverflowError" || name === "ProviderAuthError") {
    const transientContentRisk = name === "APIError" && isWindowReflectionContentRisk(message)
    return {
      category: "provider",
      retryable: name === "APIError" && (data?.isRetryable === true || transientContentRisk),
      message: `${name}: ${message}`,
      failureKind: "provider_failed",
    }
  }
  return {
    category: "provider",
    retryable: false,
    message: `${name}: ${message}`,
    failureKind: "provider_failed",
  }
}

type FrozenLlmIdentity = {
  providerID: ProviderID
  modelID: ModelID
}

type WindowLlmAuditEvent = {
  call_type: "prediction" | "evidence_gap" | "evidence_seeking" | "unified_reasoning"
  system_prompt: string
  request: unknown
  status: "success" | "error"
  child_session_id: string | null
  raw_response: string | null
  parsed_response: unknown | null
  error: string | null
  failure_kind?: WindowReflectionFailureKind | null
  provider_response_parts?: unknown[] | null
  structured_response_source?: StructuredResponseSource | null
  rationale?: string | null
}

type WindowLlmAuditWriter = (event: WindowLlmAuditEvent) => Promise<void>

/** @internal Exported for durable per-window LLM audit tests. */
export async function createWindowLlmAuditWriter(
  correctionsDir: string,
  windowIndex: number,
): Promise<WindowLlmAuditWriter> {
  const prefix = `window-${windowIndex.toString().padStart(6, "0")}`
  const auditDir = path.join(correctionsDir, "llm-calls", prefix)
  await mkdir(auditDir, { recursive: true })
  const existing = await readdir(auditDir).catch(() => [])
  let sequence = existing.reduce((maximum, name) => {
    const match = /^call-(\d+)\.json$/.exec(name)
    return match ? Math.max(maximum, Number(match[1])) : maximum
  }, 0)

  return async (event) => {
    sequence += 1
    const recordedAt = new Date().toISOString()
    const responseParts = event.provider_response_parts ?? null
    const providerReasoningParts = (responseParts ?? []).filter((part) => isRecord(part) && part.type === "reasoning")
    const parsedRationale =
      isRecord(event.parsed_response) && typeof event.parsed_response.rationale === "string"
        ? event.parsed_response.rationale
        : null
    const callPrefix = `call-${sequence.toString().padStart(6, "0")}`
    await writeJsonFile(path.join(auditDir, `${callPrefix}.json`), {
      version: 2,
      window_index: windowIndex,
      sequence,
      recorded_at: recordedAt,
      ...event,
      provider_response_parts: responseParts,
      provider_reasoning_parts: providerReasoningParts,
      rationale: event.rationale ?? parsedRationale,
      reasoning_visibility:
        "All response parts returned by the provider are recorded verbatim, including provider-visible reasoning. Provider-hidden chain-of-thought is not available to the application.",
    })
    await Promise.all(
      providerReasoningParts.map((part, index) =>
        writeJsonFile(path.join(auditDir, `${callPrefix}-reasoning-${String(index + 1).padStart(6, "0")}.json`), {
          version: 1,
          window_index: windowIndex,
          call_sequence: sequence,
          reasoning_part_index: index + 1,
          recorded_at: recordedAt,
          provider_response_part: part,
        }),
      ),
    )
  }
}

async function windowLlmAuditArtifacts(correctionsDir: string, windowIndex: number): Promise<string[]> {
  const prefix = `window-${windowIndex.toString().padStart(6, "0")}`
  const auditDir = path.join(correctionsDir, "llm-calls", prefix)
  return (await readdir(auditDir).catch(() => []))
    .filter((name) => /^call-\d+(?:-reasoning-\d+)?\.json$/.test(name))
    .toSorted()
    .map((name) => path.join(auditDir, name))
}
function reflectionRoot(): string {
  return path.join(forecastRoot(), "reflection")
}

function projectOrAbsolute(file: string): string {
  const relocationRoot = process.env.ADACAST_LITE_REFLECTION_CHECKPOINT_RELOCATE_ROOT
  const normalized = file.replaceAll("\\", "/")
  if (relocationRoot) {
    const resolvedRoot = path.resolve(relocationRoot)
    const sourceRoot = process.env.ADACAST_LITE_REFLECTION_CHECKPOINT_SOURCE_ROOT?.replaceAll("\\", "/").replace(/\/$/, "")
    if (sourceRoot && (normalized === sourceRoot || normalized.startsWith(`${sourceRoot}/`))) {
      return path.join(resolvedRoot, normalized.slice(sourceRoot.length + 1))
    }
    const datasetName = path.basename(resolvedRoot)
    const marker = `/${datasetName}/`
    const markerIndex = normalized.lastIndexOf(marker)
    if (markerIndex >= 0) {
      return path.join(resolvedRoot, normalized.slice(markerIndex + marker.length))
    }
  }
  if (path.isAbsolute(file)) return file
  return path.join(Instance.directory, file)
}

async function taskConfig(): Promise<TaskConfig> {
  await assertFrozenTaskIntegrity(forecastRoot())
  return TaskConfig.parse(await readJsonFile(path.join(forecastRoot(), "task.json")))
}

async function assertReflectionConfigureAllowed(task: TaskConfig): Promise<void> {
  const state = await readPhaseState()
  if (!task.reflection_enabled) {
    throw new Error("Independent window reflection is disabled in the frozen task")
  }
  if (state.phase !== "pre-forecast" && state.phase !== "skills-review") {
    throw new Error("Window reflection policy can only be frozen before entering the forecast phase")
  }
  if (state.reflection.configured) {
    throw new Error("Window reflection policy is already frozen and cannot be reconfigured")
  }
}

async function readBoundRun(runId: string): Promise<{
  spec: Record<string, unknown>
  evalResult: z.infer<typeof EvalResult>
  identity: Awaited<ReturnType<typeof readBoundModelArtifactIdentity>>
}> {
  const root = path.join(forecastRoot(), "runs", runId)
  const spec = z.record(z.string(), z.unknown()).parse(await readJsonFile(path.join(root, "spec.json")))
  const evalResult = EvalResult.parse(await readJsonFile(path.join(root, "eval.json")))
  if (evalResult.run_id !== runId) {
    throw new Error(`Reflection source eval.json does not match run ${runId}`)
  }
  const identity = await readBoundModelArtifactIdentity(root, String(spec.model ?? ""), evalResult)
  return { spec, evalResult, identity }
}

async function assertValidationRunReady(runId: string): Promise<void> {
  const run = await readBoundRun(runId)
  normalizeValidationExecutionContract(run.spec)
  if (run.evalResult.eval_split !== "val") {
    throw new Error("Window reflection validation requires a bound forecast-phase validation run")
  }
}

async function assertTestRunMatchesValidation(testRunId: string, validationRunId: string): Promise<void> {
  const [validation, test] = await Promise.all([readBoundRun(validationRunId), readBoundRun(testRunId)])
  normalizeValidationExecutionContract(validation.spec)
  if (
    validation.evalResult.eval_split !== "val" ||
    test.evalResult.eval_split !== "test" ||
    !validationSpecMatchesPostForecastTestSpec(validation.spec, test.spec) ||
    test.evalResult.artifact_binding?.model_artifact.source_run_id !== validationRunId ||
    test.identity.kind !== validation.identity.kind ||
    test.identity.sha256 !== validation.identity.sha256 ||
    test.identity.revision !== validation.identity.revision
  ) {
    throw new Error("Window reflection test run does not reuse the matching validation model artifact")
  }
}

async function baselinePaths(
  split: "val" | "test",
  runId: string,
  candidateId?: string,
): Promise<{
  predictions: string
  deterministicPredictions: string | null
  candidate: string | null
  baselineKind: "raw_anchor" | "skill_adjusted"
}> {
  const raw = path.join(forecastRoot(), "runs", runId, "predictions.csv")
  if (!(await Filesystem.exists(raw))) throw new Error(`Missing raw predictions for run ${runId}`)
  if (candidateId) {
    const adjusted = path.join(
      forecastRoot(),
      "adjustments",
      split === "val" ? "validation" : "test",
      runId,
      candidateId,
      "adjusted_predictions.csv",
    )
    const candidate = path.join(forecastRoot(), "adjustments", "candidates", candidateId, "candidate.json")
    if ((await Filesystem.exists(adjusted)) && (await Filesystem.exists(candidate))) {
      return { predictions: raw, deterministicPredictions: adjusted, candidate, baselineKind: "skill_adjusted" }
    }
    throw new Error(`Locked skill baseline artifacts are incomplete for ${runId}/${candidateId}`)
  }
  return { predictions: raw, deterministicPredictions: null, candidate: null, baselineKind: "raw_anchor" }
}

/** @internal Exported to verify compatible runtime-limit migrations. */
export async function reflectionExecutionSignature(input: {
  split: "val" | "test"
  runId: string
  candidateId: string | null
  task: TaskConfig
  policyPath: string
  baseline: Awaited<ReturnType<typeof baselinePaths>>
  gapPredictions: string | null
  llmEnabled: boolean
  apiConcurrency: number
  maxRounds: number
  maxFormatRetries: number
  llmIdentity?: FrozenLlmIdentity
  predictionOutputTokenLimit?: number
  llmTimeoutMs?: number
}): Promise<string> {
  const taskPath = path.join(forecastRoot(), "task.json")
  const datasetPath = projectOrAbsolute(input.task.dataset_path)
  const [taskSha256, datasetSha256, policySha256, predictionsSha256, deterministicSha256, candidateSha256, gapSha256] =
    await Promise.all([
      computeFileSha256(taskPath),
      computeFileSha256(datasetPath),
      computeFileSha256(input.policyPath),
      computeFileSha256(input.baseline.predictions),
      input.baseline.deterministicPredictions
        ? computeFileSha256(input.baseline.deterministicPredictions)
        : Promise.resolve(null),
      input.baseline.candidate ? computeFileSha256(input.baseline.candidate) : Promise.resolve(null),
      input.gapPredictions ? computeFileSha256(input.gapPredictions) : Promise.resolve(null),
    ])
  const systemPrompt =
    input.split === "test" && input.baseline.baselineKind === "skill_adjusted"
      ? SKILL_GUIDED_TEST_SYSTEM
      : ISOLATED_SYSTEM
  const payload = {
    version: 1,
    implementation_revision: 19,
    split: input.split,
    run_id: input.runId,
    candidate_id: input.candidateId,
    baseline_kind: input.baseline.baselineKind,
    task_sha256: taskSha256,
    dataset_sha256: datasetSha256,
    policy_sha256: policySha256,
    predictions_sha256: predictionsSha256,
    deterministic_predictions_sha256: deterministicSha256,
    candidate_sha256: candidateSha256,
    gap_predictions_sha256: gapSha256,
    llm_enabled: input.llmEnabled,
    llm_provider_id: input.llmIdentity?.providerID ?? null,
    llm_model_id: input.llmIdentity?.modelID ?? null,
    api_concurrency: input.apiConcurrency,
    max_semantic_rounds: input.maxRounds,
    max_format_retries: input.maxFormatRetries,
    evidence_selection_mode: evidenceSelectionMode(),
    provider_retry_limit: WINDOW_REFLECTION_PROVIDER_RETRY_LIMIT,
    prediction_output_layout: "row_oriented_points",
    prediction_reasoning_token_guidance: WINDOW_REFLECTION_REASONING_TOKEN_GUIDANCE,
    prediction_output_token_limit: input.predictionOutputTokenLimit ?? windowReflectionPredictionOutputTokenLimit(),
    safety_reflection_llm_timeout_ms: input.llmTimeoutMs ?? windowReflectionLlmTimeoutMs(),
    system_prompt_sha256: createHash("sha256").update(systemPrompt).digest("hex"),
    response_schema_sha256: createHash("sha256")
      .update(JSON.stringify(z.toJSONSchema(LlmResponse)))
      .digest("hex"),
    evidence_gap_system_prompt_sha256: createHash("sha256").update(EVIDENCE_GAP_SYSTEM).digest("hex"),
    evidence_gap_response_schema_sha256: createHash("sha256")
      .update(JSON.stringify(z.toJSONSchema(EvidenceGapResponse)))
      .digest("hex"),
    evidence_seeking_system_prompt_sha256: createHash("sha256").update(EVIDENCE_SEEKING_SYSTEM).digest("hex"),
    evidence_seeking_response_schema_sha256: createHash("sha256")
      .update(JSON.stringify(z.toJSONSchema(EvidenceTypeSelectionResponse)))
      .digest("hex"),
    unified_reasoning_system_prompt_sha256: createHash("sha256").update(UNIFIED_REASONING_SYSTEM).digest("hex"),
    post_loop_safety_system_prompt_sha256: createHash("sha256").update(LEGACY_POST_LOOP_SAFETY_SYSTEM).digest("hex"),
    prediction_generator: "unified_reasoner_with_provisional_evidence_fusion_v1",
  }
  return createHash("sha256").update(JSON.stringify(payload)).digest("hex")
}

async function resolveLlmIdentity(ctx: Tool.Context): Promise<FrozenLlmIdentity> {
  const parent = await MessageV2.get({ sessionID: ctx.sessionID, messageID: ctx.messageID })
  if (parent.info.role !== "assistant") throw new Error("Window reflection requires an assistant parent message")
  return { providerID: parent.info.providerID, modelID: parent.info.modelID }
}

export const WINDOW_REFLECTION_TEST_API_CONCURRENCY = 4
export const WINDOW_REFLECTION_MAX_API_CONCURRENCY = 10
const LEGACY_WINDOW_REFLECTION_TEST_API_CONCURRENCY = 50
export const WINDOW_REFLECTION_PROVIDER_RETRY_LIMIT = 5
const WINDOW_REFLECTION_CHECK_CONCURRENCY = 8
const WINDOW_REFLECTION_PROVIDER_RETRY_BASE_MS = 750
const WINDOW_REFLECTION_PROVIDER_RETRY_MAX_MS = 15_000
export const WINDOW_REFLECTION_LLM_TIMEOUT_MS = 5 * 60 * 1000
export function windowReflectionLlmTimeoutMs(): number {
  return configuredWindowLlmTimeoutMs() ?? WINDOW_REFLECTION_LLM_TIMEOUT_MS
}

/** @internal Exported for deterministic configuration tests. */
export function resolveWindowReflectionTestApiConcurrency(
  value = process.env.ADACAST_LITE_WINDOW_REFLECTION_TEST_API_CONCURRENCY ?? (configuredWindowConcurrency()?.toString()),
): number {
  if (value === undefined || value.trim() === "") return WINDOW_REFLECTION_TEST_API_CONCURRENCY
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > WINDOW_REFLECTION_MAX_API_CONCURRENCY) {
    throw new Error(
      `ADACAST_LITE_WINDOW_REFLECTION_TEST_API_CONCURRENCY must be an integer between 1 and ${WINDOW_REFLECTION_MAX_API_CONCURRENCY}`,
    )
  }
  return parsed
}

type ConcurrencyLimiter = <T>(task: () => Promise<T>) => Promise<T>

/** @internal Exported for deterministic concurrency tests. */
export function createConcurrencyLimiter(maxConcurrency: number): ConcurrencyLimiter {
  if (!Number.isInteger(maxConcurrency) || maxConcurrency < 1) {
    throw new Error("Concurrency limit must be a positive integer")
  }
  let active = 0
  const waiters: Array<() => void> = []

  return async <T>(task: () => Promise<T>): Promise<T> => {
    let reservedByWaiter = false
    if (active >= maxConcurrency) {
      await new Promise<void>((resolve) =>
        waiters.push(() => {
          reservedByWaiter = true
          resolve()
        }),
      )
    }
    if (!reservedByWaiter) active++
    try {
      return await task()
    } finally {
      const next = waiters.shift()
      if (next) next()
      else active--
    }
  }
}

/** @internal Exported for deterministic retry tests. */
export function windowReflectionProviderRetryDelayMs(attempt: number, jitter = Math.random()): number {
  const boundedAttempt = Math.max(0, Math.floor(attempt))
  const boundedJitter = Math.min(1, Math.max(0, jitter))
  const exponential = Math.min(
    WINDOW_REFLECTION_PROVIDER_RETRY_MAX_MS,
    WINDOW_REFLECTION_PROVIDER_RETRY_BASE_MS * 2 ** boundedAttempt,
  )
  return Math.round(exponential * (0.5 + boundedJitter * 0.5))
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function throwIfAborted(signal: AbortSignal): void {
  if (!signal.aborted) return
  const error = new Error("Window reflection was aborted")
  error.name = "AbortError"
  throw error
}

class ChildSessionCancellationGroup {
  private readonly children = new Set<SessionID>()
  private readonly abortAll = () => void this.cancelAll()

  constructor(private readonly signal: AbortSignal) {
    signal.addEventListener("abort", this.abortAll)
  }

  track(sessionID: SessionID): void {
    this.children.add(sessionID)
    if (this.signal.aborted) void SessionPrompt.cancel(sessionID)
  }

  release(sessionID: SessionID): void {
    this.children.delete(sessionID)
  }

  async cancelAll(): Promise<void> {
    const children = [...this.children]
    this.children.clear()
    await Promise.allSettled(children.map((sessionID) => SessionPrompt.cancel(sessionID)))
  }

  dispose(): void {
    this.signal.removeEventListener("abort", this.abortAll)
  }
}

type WindowOutcomeStatus =
  | "accepted"
  | "resumed_accepted"
  | "provider_failed"
  | "schema_failed"
  | "response_empty"
  | "output_truncated"
  | "format_failed"
  | "semantic_failed"

/** @internal Exported for request-failure policy tests. */
export function windowOutcomeRequiresRerun(status: WindowOutcomeStatus): boolean {
  return ["provider_failed", "schema_failed", "response_empty", "output_truncated", "format_failed"].includes(status)
}

type WindowOutcome = {
  window_index: number
  status: WindowOutcomeStatus
  accepted: boolean
  rounds_attempted: number
  llm_calls: number
  provider_retries: number
  format_failures: number
  message: string | null
}

class WindowReflectionProgress {
  private readonly startedAt = new Date().toISOString()
  private readonly running = new Set<number>()
  private readonly outcomes = new Map<number, WindowOutcome>()
  private writeChain = Promise.resolve()

  constructor(
    private readonly file: string,
    private readonly windowIndices: number[],
    private readonly apiConcurrency: number,
    private readonly ctx: Tool.Context,
  ) {}

  initialize(): Promise<void> {
    const completed = this.outcomes.size
    this.ctx.metadata({
      title: `LLM window reasoning ${completed}/${this.windowIndices.length}`,
      metadata: {
        status: "running",
        completed,
        total: this.windowIndices.length,
        apiConcurrency: this.apiConcurrency,
      },
    })
    return this.write("running")
  }

  restore(outcome: WindowOutcome): void {
    this.outcomes.set(outcome.window_index, outcome)
  }

  markRunning(windowIndex: number): Promise<void> {
    this.running.add(windowIndex)
    return this.write("running")
  }

  async markCompleted(outcome: WindowOutcome): Promise<void> {
    this.running.delete(outcome.window_index)
    this.outcomes.set(outcome.window_index, outcome)
    this.ctx.metadata({
      title: `LLM window reasoning ${this.outcomes.size}/${this.windowIndices.length}`,
      metadata: {
        status: "running",
        completed: this.outcomes.size,
        total: this.windowIndices.length,
        apiConcurrency: this.apiConcurrency,
      },
    })
    await this.write("running")
  }

  finish(status: "completed" | "failed", error?: unknown): Promise<void> {
    this.ctx.metadata({
      title: `LLM window reasoning ${this.outcomes.size}/${this.windowIndices.length}`,
      metadata: {
        status,
        completed: this.outcomes.size,
        total: this.windowIndices.length,
        apiConcurrency: this.apiConcurrency,
      },
    })
    return this.write(status, error)
  }

  private write(status: "running" | "completed" | "failed", error?: unknown): Promise<void> {
    const completed = [...this.outcomes.values()].sort((a, b) => a.window_index - b.window_index)
    const completedIndices = new Set(completed.map((item) => item.window_index))
    const pendingIndices = this.windowIndices.filter(
      (windowIndex) => !completedIndices.has(windowIndex) && !this.running.has(windowIndex),
    )
    const snapshot = {
      version: 1,
      status,
      started_at: this.startedAt,
      updated_at: new Date().toISOString(),
      api_concurrency: this.apiConcurrency,
      checker_concurrency: WINDOW_REFLECTION_CHECK_CONCURRENCY,
      provider_retry_limit: WINDOW_REFLECTION_PROVIDER_RETRY_LIMIT,
      total_windows: this.windowIndices.length,
      scheduled_window_indices: [...this.windowIndices].sort((a, b) => a - b),
      completed_windows: completed.length,
      completed_window_indices: [...completedIndices].sort((a, b) => a - b),
      pending_windows: pendingIndices.length,
      pending_window_indices: pendingIndices,
      in_flight_windows: [...this.running].sort((a, b) => a - b),
      accepted_windows: completed.filter((item) => item.accepted).length,
      request_failed_windows: completed.filter((item) => windowOutcomeRequiresRerun(item.status)).length,
      semantic_fallback_windows: completed.filter((item) => item.status === "semantic_failed").length,
      fallback_windows: completed.filter((item) => item.status === "semantic_failed").length,
      windows: completed,
      error: error instanceof Error ? `${error.name}: ${error.message}` : error ? String(error) : null,
    }
    this.writeChain = this.writeChain.then(() => writeJsonFile(this.file, snapshot))
    return this.writeChain
  }
}

function requestFailure(error: unknown): WindowReflectionRequestError {
  if (error instanceof WindowReflectionRequestError) return error
  const classified = classifyWindowReflectionLlmError(error)
  return new WindowReflectionRequestError(
    classified.message,
    classified.category,
    classified.retryable,
    classified.failureKind,
  )
}

async function persistWindowFailure(input: {
  correctionsDir: string
  windowIndex: number
  round: number
  formatAttempt: number
  providerAttempt: number
  failure: WindowReflectionRequestError
  contextProjection: "original" | "semantic" | "compact_semantic"
}): Promise<void> {
  const segment = input.failure.failureKind.replaceAll("_", "-")
  await writeJsonFile(
    path.join(
      input.correctionsDir,
      `window-${input.windowIndex.toString().padStart(6, "0")}-round-${input.round}-format-${input.formatAttempt}-provider-${input.providerAttempt}-${segment}.json`,
    ),
    {
      window_index: input.windowIndex,
      round: input.round,
      format_attempt: input.formatAttempt,
      provider_attempt: input.providerAttempt,
      category: input.failure.category,
      failure_kind: input.failure.failureKind,
      retryable: input.failure.retryable,
      message: input.failure.message,
      context_projection: input.contextProjection,
      recorded_at: new Date().toISOString(),
    },
  )
}

const ReusableWindowCorrection = CheckResult.extend({
  window_index: z.number().int().nonnegative(),
  points: LlmResponse.shape.points,
  rationale: z.string(),
  evidence_ids: z.array(z.string()),
})

/** @internal Exported for checkpoint validation tests. */
export function isReusableWindowCorrection(
  value: unknown,
  expected: { windowIndex: number; horizonIndices: number[]; timestamps: string[] },
): boolean {
  const parsed = ReusableWindowCorrection.safeParse(value)
  if (
    !parsed.success ||
    !parsed.data.accepted ||
    parsed.data.format_retry ||
    parsed.data.window_index !== expected.windowIndex ||
    expected.horizonIndices.length !== expected.timestamps.length ||
    parsed.data.points.length !== expected.horizonIndices.length
  ) {
    return false
  }
  return parsed.data.points.every(
    (point, index) =>
      point.horizon_index === expected.horizonIndices[index] && point.timestamp === expected.timestamps[index],
  )
}

/** @internal Exported for durable checkpoint tests. */
export async function isReusableEvidenceLoopCorrection(
  value: unknown,
  expected: { windowIndex: number; horizonIndices: number[]; timestamps: string[] },
): Promise<boolean> {
  if (!isReusableWindowCorrection(value, expected) || !isRecord(value)) return false
  if (value.assessment_stage !== "evidence_loop_terminal") return false
  if (!isRecord(value.evidence_loop)) return false
  const terminationReason = value.evidence_loop.termination_reason
  if (typeof terminationReason === "string" && REOPENABLE_EVIDENCE_LOOP_FAILURES.has(terminationReason)) return false
  const forcedDepth = forcedEvidenceLoopDepth()
  if (
    forcedDepth &&
    (!isRecord(value.evidence_loop) ||
      typeof value.evidence_loop.accepted_transitions !== "number" ||
      value.evidence_loop.accepted_transitions < forcedDepth)
  ) {
    return false
  }
  const stateResult = CriticPredictionState.safeParse(value.critic_prediction_state)
  const artifactResult = CriticStateArtifact.safeParse(value.critic_prediction_state_artifact)
  if (!stateResult.success || !artifactResult.success) return false
  const state = stateResult.data
  const artifact = artifactResult.data
  if (state.window_index !== expected.windowIndex) return false
  if (state.evidence_gap === 1 && ["evidence_exhausted", "no_valid_transition"].includes(String(terminationReason)) &&
      Array.isArray(value.evidence_loop.rejected_transitions) &&
      value.evidence_loop.rejected_transitions.some((record: unknown) => isRecord(record) &&
        typeof record.reason === "string" && REOPENABLE_EVIDENCE_LOOP_FAILURES.has(record.reason) &&
        !state.evidences.some((evidence) => evidence.evidence_candidate_id === record.candidate_id))) return false
  try {
    if ((await computeFileSha256(projectOrAbsolute(artifact.path))) !== artifact.sha256) return false
    const persisted = CriticPredictionState.parse(await readJsonFile<unknown>(projectOrAbsolute(artifact.path)))
    if (JSON.stringify(persisted) !== JSON.stringify(state)) return false
  } catch {
    return false
  }
  const points = ReusableWindowCorrection.parse(value).points
  return (
    state.predictions.length === points.length &&
    state.predictions.every(
      (point, index) =>
        point.horizon_index === points[index]?.horizon_index &&
        point.timestamp === points[index]?.timestamp &&
        point.value === points[index]?.value,
    )
  )
}

/** @internal Exported for legacy Safety Reflection checkpoint migration tests. */
export async function restoreLegacyPreSafetyCheckpoint(
  value: unknown,
  correctionsDir: string,
  expected: { windowIndex: number; horizonIndices: number[]; timestamps: string[] },
): Promise<boolean> {
  if (!isRecord(value) || value.assessment_stage !== "state_terminal" || !isRecord(value.safety_reflection)) {
    return false
  }
  const preSafetySha256 = value.safety_reflection.pre_safety_state_sha256
  const evidenceLoop = value.evidence_loop
  if (typeof preSafetySha256 !== "string" || preSafetySha256.length !== 64 || !isRecord(evidenceLoop)) return false
  const terminationReason = EvidenceLoopTerminationReason.safeParse(evidenceLoop.termination_reason)
  const predictability = evidenceLoop.predictability
  if (!terminationReason.success || (predictability !== "predictable" && predictability !== "unpredictable")) {
    return false
  }

  const prefix = `window-${expected.windowIndex.toString().padStart(6, "0")}`
  const statesDir = path.join(correctionsDir, "states", prefix)
  const preSafetyPath = path.join(statesDir, `state-${preSafetySha256}.json`)
  try {
    if ((await computeFileSha256(preSafetyPath)) !== preSafetySha256) return false
    const state = CriticPredictionState.parse(await readJsonFile<unknown>(preSafetyPath))
    if (
      state.window_index !== expected.windowIndex ||
      state.predictions.length !== expected.horizonIndices.length ||
      !state.predictions.every(
        (point, index) =>
          point.horizon_index === expected.horizonIndices[index] && point.timestamp === expected.timestamps[index],
      )
    ) {
      return false
    }
    const priorCheckpoint = EvidenceLoopCheckpoint.safeParse(
      await readJsonFile<unknown>(path.join(statesDir, "loop-checkpoint.json")).catch(() => null),
    )
    const seekingTrace = Array.isArray(evidenceLoop.seeking_trace) ? evidenceLoop.seeking_trace : []
    const rejectedTransitions = Array.isArray(evidenceLoop.rejected_transitions)
      ? evidenceLoop.rejected_transitions
      : []
    await writeJsonFile(path.join(statesDir, "current.json"), {
      path: preSafetyPath,
      sha256: preSafetySha256,
    })
    await writeJsonFile(path.join(statesDir, "loop-checkpoint.json"), {
      version: 1,
      window_index: expected.windowIndex,
      state_sha256: preSafetySha256,
      phase: "terminated",
      selected_candidate_id: null,
      attempted_candidate_ids: priorCheckpoint.success ? priorCheckpoint.data.attempted_candidate_ids : [],
      seeking_trace: seekingTrace,
      rejected_transitions: rejectedTransitions,
      accepted_transitions:
        typeof evidenceLoop.accepted_transitions === "number"
          ? evidenceLoop.accepted_transitions
          : state.evidences.length,
      termination_reason: terminationReason.data,
      predictability,
      updated_at: new Date().toISOString(),
    })
    return true
  } catch {
    return false
  }
}

async function isolatedResponse(
  ctx: Tool.Context,
  blindContext: string,
  windowIndex: number,
  expectedPointCount: number,
  round: number,
  maxRounds: number,
  priorDetectorReport: unknown,
  priorProposal: unknown,
  reasoningMode: "safety_reflection" | "skill_guided_test" | "evidence_loop_test",
  cancellationGroup: ChildSessionCancellationGroup,
  frozenIdentity?: FrozenLlmIdentity,
  systemOverride?: string,
  audit?: WindowLlmAuditWriter,
): Promise<z.infer<typeof LlmResponse>> {
  throwIfAborted(ctx.abort)
  const model = frozenIdentity ?? (await resolveLlmIdentity(ctx))
  const child = await Session.create({
    parentID: ctx.sessionID,
    title:
      reasoningMode === "skill_guided_test"
        ? `Window ${windowIndex.toString().padStart(6, "0")} skill-guided test round ${round}`
        : `Window ${windowIndex.toString().padStart(6, "0")} blind reflection round ${round}`,
    permission: Permission.fromConfig({ "*": "deny" }),
  })
  cancellationGroup.track(child.id)
  const system = systemOverride ?? (reasoningMode === "skill_guided_test" ? SKILL_GUIDED_TEST_SYSTEM : ISOLATED_SYSTEM)
  const requestText = [
    reasoningMode === "skill_guided_test"
      ? `Sample-specific forecast reasoning round: ${round}/${maxRounds}.`
      : `Semantic repair round: ${round}/${maxRounds}.`,
    priorDetectorReport
      ? `${reasoningMode === "skill_guided_test" ? "Previous proposal safety diagnosis; use it as a boundary check, not as an accuracy objective" : "Previous deterministic rejection"}:\n${JSON.stringify(priorDetectorReport)}`
      : "",
    priorProposal ? `Previous rejected proposal:\n${JSON.stringify(priorProposal)}` : "",
    "Blind window context:",
    blindContext,
  ]
    .filter(Boolean)
    .join("\n\n")
  let responseText: string | null = null
  let responseParts: unknown[] | null = null
  try {
    const result = await withWindowReflectionTimeout(
      SessionPrompt.prompt({
        sessionID: child.id,
        messageID: MessageID.ascending(),
        model,
        agent: "window-reflection",
        system,
        tools: {},
        maxOutputTokens: windowReflectionPredictionOutputTokenLimit(),
        parts: [
          {
            type: "text",
            text: requestText,
          },
        ],
      }),
      {
        timeoutMs: windowReflectionLlmTimeoutMs(),
        label: `${reasoningMode} LLM request`,
        onTimeout: () => SessionPrompt.cancel(child.id),
      },
    )
    responseParts = result.parts
    if (result.info.role !== "assistant") {
      throw new WindowReflectionRequestError("Isolated LLM returned a non-assistant response", "format", true)
    }
    if (result.info.error) {
      const failure = classifyWindowReflectionLlmError(result.info.error)
      throw new WindowReflectionRequestError(failure.message, failure.category, failure.retryable, failure.failureKind)
    }
    responseText = result.parts
      .map((part) => (part.type === "text" ? part.text : ""))
      .filter(Boolean)
      .join("\n")
      .trim()
    try {
      const recovered = parseStructuredResponse(responseText, responseParts, (text) =>
        parseWindowReflectionText(text, windowIndex, expectedPointCount),
      )
      const parsed = recovered.value
      await audit?.({
        call_type: "prediction",
        system_prompt: system,
        request: requestText,
        status: "success",
        child_session_id: child.id,
        raw_response: recovered.text,
        parsed_response: parsed,
        error: null,
        provider_response_parts: responseParts,
        structured_response_source: recovered.source,
        rationale: parsed.rationale,
      })
      return parsed
    } catch (error) {
      throw structuredResponseError(error, responseText, responseParts)
    }
  } catch (error) {
    const failure = requestFailure(error)
    await audit?.({
      call_type: "prediction",
      system_prompt: system,
      request: requestText,
      status: "error",
      child_session_id: child.id,
      raw_response: responseText,
      parsed_response: null,
      error: failure.message,
      failure_kind: failure.failureKind,
      provider_response_parts: responseParts,
    })
    throw failure
  } finally {
    cancellationGroup.release(child.id)
  }
}

async function isolatedEvidenceGapAssessmentOnce(
  ctx: Tool.Context,
  state: z.infer<typeof CriticPredictionState>,
  evidenceCandidates: unknown,
  lookbackTargetWindow: unknown,
  windowIndex: number,
  cancellationGroup: ChildSessionCancellationGroup,
  frozenIdentity?: FrozenLlmIdentity,
  audit?: WindowLlmAuditWriter,
): Promise<z.infer<typeof EvidenceGapResponse>> {
  throwIfAborted(ctx.abort)
  const model = frozenIdentity ?? (await resolveLlmIdentity(ctx))
  const child = await Session.create({
    parentID: ctx.sessionID,
    title: `Window ${windowIndex.toString().padStart(6, "0")} evidence-gap assessment`,
    permission: Permission.fromConfig({ "*": "deny" }),
  })
  cancellationGroup.track(child.id)
  const request = buildEvidenceGapAssessmentInput(state, evidenceCandidates, lookbackTargetWindow)
  let responseText: string | null = null
  let responseParts: unknown[] | null = null
  try {
    const result = await withWindowReflectionTimeout(
      SessionPrompt.prompt({
        sessionID: child.id,
        messageID: MessageID.ascending(),
        model,
        agent: "window-reflection",
        system: EVIDENCE_GAP_SYSTEM,
        tools: {},
        format: evidenceGapStructuredOutputFormat(),
        parts: [
          {
            type: "text",
            text: JSON.stringify(request),
          },
        ],
      }),
      {
        timeoutMs: windowReflectionLlmTimeoutMs(),
        label: "Evidence-gap LLM request",
        onTimeout: () => SessionPrompt.cancel(child.id),
      },
    )
    responseParts = result.parts
    if (result.info.role !== "assistant") {
      throw new WindowReflectionRequestError("Evidence-gap assessor returned a non-assistant response", "format", true)
    }
    responseText = result.parts
      .map((part) => (part.type === "text" ? part.text : ""))
      .filter(Boolean)
      .join("\n")
      .trim()
    const missingStructuredOutput = isRecord(result.info.error) && result.info.error.name === "StructuredOutputError"
    if (result.info.error && !missingStructuredOutput) {
      const failure = classifyWindowReflectionLlmError(result.info.error)
      throw new WindowReflectionRequestError(failure.message, failure.category, failure.retryable, failure.failureKind)
    }
    const recovered = (() => {
      try {
        return recoverEvidenceGapSubmission(result.info.structured, responseText)
      } catch (error) {
        throw structuredResponseError(error, responseText, responseParts)
      }
    })()
    const parsed = recovered.value
    await audit?.({
      call_type: "evidence_gap",
      system_prompt: EVIDENCE_GAP_SYSTEM,
      request,
      status: "success",
      child_session_id: child.id,
      raw_response: recovered.text,
      parsed_response: parsed,
      error: null,
      provider_response_parts: responseParts,
      structured_response_source: recovered.source,
      rationale: parsed.rationale,
    })
    return parsed
  } catch (error) {
    const failure = requestFailure(error)
    await audit?.({
      call_type: "evidence_gap",
      system_prompt: EVIDENCE_GAP_SYSTEM,
      request,
      status: "error",
      child_session_id: child.id,
      raw_response: responseText,
      parsed_response: null,
      error: failure.message,
      failure_kind: failure.failureKind,
      provider_response_parts: responseParts,
    })
    throw failure
  } finally {
    cancellationGroup.release(child.id)
  }
}

export const EVIDENCE_GAP_MAX_RETRIES = 2

/** @internal Exported for bounded evidence-gap retry tests. */
export async function retryEvidenceGapAssessment<T>(operation: (attempt: number) => Promise<T>): Promise<T> {
  let lastError: unknown = new Error("Evidence-gap assessment did not run")
  for (let attempt = 0; attempt <= EVIDENCE_GAP_MAX_RETRIES; attempt++) {
    try {
      return await operation(attempt)
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") throw error
      lastError = error
    }
  }
  throw lastError
}

async function isolatedEvidenceGapAssessment(
  ctx: Tool.Context,
  state: z.infer<typeof CriticPredictionState>,
  evidenceCandidates: unknown,
  lookbackTargetWindow: unknown,
  windowIndex: number,
  cancellationGroup: ChildSessionCancellationGroup,
  frozenIdentity?: FrozenLlmIdentity,
  audit?: WindowLlmAuditWriter,
  onAttempt?: () => void,
): Promise<z.infer<typeof EvidenceGapResponse>> {
  return retryEvidenceGapAssessment(async () => {
    throwIfAborted(ctx.abort)
    onAttempt?.()
    return isolatedEvidenceGapAssessmentOnce(
      ctx,
      state,
      evidenceCandidates,
      lookbackTargetWindow,
      windowIndex,
      cancellationGroup,
      frozenIdentity,
      audit,
    )
  })
}

async function isolatedEvidenceSeeking(
  ctx: Tool.Context,
  state: z.infer<typeof CriticPredictionState>,
  evidenceCandidates: unknown,
  attemptedCandidateIds: ReadonlySet<string>,
  windowIndex: number,
  cancellationGroup: ChildSessionCancellationGroup,
  frozenIdentity?: FrozenLlmIdentity,
  audit?: WindowLlmAuditWriter,
  onLlmCall?: () => void,
): Promise<z.infer<typeof EvidenceSeekingResponse>> {
  throwIfAborted(ctx.abort)
  const unused = unusedEvidenceCandidates(state, evidenceCandidates, attemptedCandidateIds)
  const availableTypes = EVIDENCE_CANDIDATE_PRIORITY.filter((candidateType) => isRecord(unused[candidateType]))
  if (availableTypes.length === 0) {
    return { selected_evidence_candidate_ids: [], rationale: "No unused candidates remain.", exhausted: true }
  }
  const selectionMode = evidenceSelectionMode()
  if (selectionMode !== "adaptive") {
    const selectedType = selectEvidenceTypeByMode(unused, selectionMode)
    if (!selectedType) {
      return { selected_evidence_candidate_ids: [], rationale: "No unused candidates remain.", exhausted: true }
    }
    return finalizeEvidenceSeekingResponse(
      unused,
      selectedType,
      selectionMode === "random"
        ? `Controller randomly selected the ${selectedType} evidence family without using needed_evidence.`
        : `Controller selected the ${selectedType} evidence family using fixed order experience > context > knowledge without using needed_evidence.`,
    )
  }
  const deterministicType = deterministicEvidenceTypeFromNeed(state.needed_evidence, unused)
  if (deterministicType) {
    return finalizeEvidenceSeekingResponse(
      unused,
      deterministicType,
      `Controller matched needed_evidence to ${deterministicType}; ties use context > experience > knowledge.`,
    )
  }
  if (availableTypes.length === 1) {
    const selectedType = availableTypes[0]!
    return finalizeEvidenceSeekingResponse(
      unused,
      selectedType,
      `Controller selected the only remaining candidate type: ${selectedType}.`,
    )
  }
  const seekingInput = {
    needed_evidence: state.needed_evidence,
    available_candidate_types: availableTypes.map((candidateType) => ({
      candidate_type: candidateType,
      evidence_candidate_id: unused[candidateType]?.evidence_candidate_id,
    })),
  }
  const model = frozenIdentity ?? (await resolveLlmIdentity(ctx))
  const child = await Session.create({
    parentID: ctx.sessionID,
    title: `Window ${windowIndex.toString().padStart(6, "0")} evidence seeking`,
    permission: Permission.fromConfig({ "*": "deny" }),
  })
  cancellationGroup.track(child.id)
  let responseText: string | null = null
  let responseParts: unknown[] | null = null
  try {
    onLlmCall?.()
    const result = await withWindowReflectionTimeout(
      SessionPrompt.prompt({
        sessionID: child.id,
        messageID: MessageID.ascending(),
        model,
        agent: "window-reflection",
        system: EVIDENCE_SEEKING_SYSTEM,
        tools: {},
        parts: [
          {
            type: "text",
            text: JSON.stringify(seekingInput),
          },
        ],
      }),
      {
        timeoutMs: windowReflectionLlmTimeoutMs(),
        label: "Evidence-seeking LLM request",
        onTimeout: () => SessionPrompt.cancel(child.id),
      },
    )
    responseParts = result.parts
    if (result.info.role !== "assistant") {
      throw new WindowReflectionRequestError("Evidence seeker returned a non-assistant response", "format", true)
    }
    if (result.info.error) {
      const failure = classifyWindowReflectionLlmError(result.info.error)
      throw new WindowReflectionRequestError(failure.message, failure.category, failure.retryable, failure.failureKind)
    }
    responseText = result.parts
      .map((part) => (part.type === "text" ? part.text : ""))
      .filter(Boolean)
      .join("\n")
      .trim()
    const recovered = (() => {
      try {
        return parseStructuredResponse(responseText, responseParts, parseEvidenceTypeSelectionText)
      } catch (error) {
        throw structuredResponseError(error, responseText, responseParts)
      }
    })()
    const selection = recovered.value
    if (!availableTypes.includes(selection.candidate_type)) {
      throw new WindowReflectionRequestError(
        `Evidence seeker selected unavailable candidate type ${selection.candidate_type}`,
        "format",
        true,
        "schema_failed",
      )
    }
    const seeking = finalizeEvidenceSeekingResponse(unused, selection.candidate_type, selection.rationale)
    await audit?.({
      call_type: "evidence_seeking",
      system_prompt: EVIDENCE_SEEKING_SYSTEM,
      request: seekingInput,
      status: "success",
      child_session_id: child.id,
      raw_response: recovered.text,
      parsed_response: { ...selection, controller_response: seeking },
      error: null,
      provider_response_parts: responseParts,
      structured_response_source: recovered.source,
      rationale: seeking.rationale,
    })
    return seeking
  } catch (error) {
    const failure = requestFailure(error)
    await audit?.({
      call_type: "evidence_seeking",
      system_prompt: EVIDENCE_SEEKING_SYSTEM,
      request: seekingInput,
      status: "error",
      child_session_id: child.id,
      raw_response: responseText,
      parsed_response: null,
      error: failure.message,
      failure_kind: failure.failureKind,
      provider_response_parts: responseParts,
    })
    throw failure
  } finally {
    cancellationGroup.release(child.id)
  }
}

async function isolatedEvidenceStageNeedAssessment(
  ctx: Tool.Context,
  state: z.infer<typeof CriticPredictionState>,
  evidenceCandidate: unknown,
  candidateType: EvidenceCandidateType,
  windowIndex: number,
  cancellationGroup: ChildSessionCancellationGroup,
  frozenIdentity?: FrozenLlmIdentity,
  audit?: WindowLlmAuditWriter,
): Promise<z.infer<typeof EvidenceStageNeedResponse>> {
  if (state.evidence_gap === 0) return { use_current_evidence: false, rationale: "The evidence gap is closed." }
  const seeking = await isolatedEvidenceSeeking(
    ctx,
    state,
    { [candidateType]: evidenceCandidate },
    new Set(),
    windowIndex,
    cancellationGroup,
    frozenIdentity,
    audit,
  )
  return {
    use_current_evidence: seeking.selected_evidence_candidate_ids.length === 1,
    rationale: seeking.rationale,
  }
}

const ExperienceRuleReference = z.object({
  version: z.literal(1),
  window_index: z.number().int().nonnegative(),
  semantics: z.literal("locked_rules_applied_to_initial_prediction_only"),
  experience_candidate_id: z.string().min(1),
  experience_candidate_artifact: CriticStateArtifact,
  initial_predictions: z.array(CriticPredictionPoint),
  predictions: z.array(CriticPredictionPoint),
  reasoning_contract: z.object({
    reference_only: z.literal(true),
    not_a_state_transition: z.literal(true),
    do_not_add_reference_delta_to_current: z.literal(true),
  }),
})

type EvidenceAdapterAttachment = {
  evidence_candidate_id: string
  candidate_type: EvidenceCandidateType
  adapter: "experience_adapter" | "context_adapter" | "knowledge_adapter"
  content: Record<string, unknown>
  auxiliary: Record<string, unknown>
}

function positionalContextTable(input: {
  value: unknown
  label: string
  expectedTimestamps: string[]
}): Record<string, unknown> {
  if (!isRecord(input.value) || !Array.isArray(input.value.columns) || !Array.isArray(input.value.rows)) {
    throw new Error(`Context evidence is missing the complete ${input.label} table`)
  }
  const columns = input.value.columns
  const rows = input.value.rows
  if (columns[0] !== "timestamp" || columns.some((column) => typeof column !== "string")) {
    throw new Error(`Context ${input.label} table has an invalid columns contract`)
  }
  if (input.value.row_count !== rows.length || rows.length !== input.expectedTimestamps.length) {
    throw new Error(`Context ${input.label} table is incomplete`)
  }
  rows.forEach((row, index) => {
    if (!Array.isArray(row) || row.length !== columns.length || String(row[0]) !== input.expectedTimestamps[index]) {
      throw new Error(`Context ${input.label} table is not aligned to its forecast window`)
    }
  })
  return input.value
}

function adaptEvidenceCandidate(input: {
  candidate: Record<string, unknown>
  state: z.infer<typeof CriticPredictionState>
  initialState: z.infer<typeof CriticPredictionState>
  lookbackTargetWindow: unknown
  exogenousHistorySteps?: number
  experienceRuleReference: z.infer<typeof ExperienceRuleReference> | null
}): EvidenceAdapterAttachment {
  const candidateType = input.candidate.candidate_type
  const candidateId = input.candidate.evidence_candidate_id
  if (!(["experience", "context", "knowledge"] as unknown[]).includes(candidateType)) {
    throw new Error("Evidence adapter received an unknown candidate type")
  }
  if (typeof candidateId !== "string" || !candidateId) {
    throw new Error("Evidence adapter received a candidate without an id")
  }
  const normalizedCandidateType = candidateType as EvidenceCandidateType
  if (candidateType === "experience") {
    const reference = input.experienceRuleReference
    if (!reference) throw new Error("Prepared experience rule reference is missing")
    if (reference.experience_candidate_id !== candidateId) {
      throw new Error("Experience rule reference belongs to another evidence candidate")
    }
    if (
      reference.window_index !== input.state.window_index ||
      input.initialState.window_index !== input.state.window_index
    ) {
      throw new Error("Unified reasoner inputs belong to different windows")
    }
    if (JSON.stringify(reference.initial_predictions) !== JSON.stringify(input.initialState.predictions)) {
      throw new Error("Experience rule reference is not bound to the initial prediction")
    }
    return {
      evidence_candidate_id: candidateId,
      candidate_type: "experience",
      adapter: "experience_adapter",
      content: input.candidate,
      auxiliary: { rule_reference_prediction: reference },
    }
  }
  if (normalizedCandidateType === "context") {
    const payload = input.candidate.payload
    if (!isRecord(payload)) throw new Error("Context evidence candidate is missing its payload")
    const lookback = input.lookbackTargetWindow
    if (!isRecord(lookback) || !Array.isArray(lookback.rows)) {
      throw new Error("Unified reasoner is missing its complete lookback target window")
    }
    const lookbackTimestamps = lookback.rows.map((row) => (Array.isArray(row) ? String(row[0]) : ""))
    const historySteps = input.exogenousHistorySteps ?? lookbackTimestamps.length
    if (!Number.isInteger(historySteps) || historySteps <= 0 || historySteps > lookbackTimestamps.length) {
      throw new Error("Context exogenous history scope is invalid")
    }
    const predictionTimestamps = input.state.predictions.map((point) => point.timestamp)
    const lookbackWindowExogenous = positionalContextTable({
      value: payload.lookback_window_exogenous,
      label: "lookback-window exogenous",
      expectedTimestamps: lookbackTimestamps.slice(-historySteps),
    })
    const predictWindowExogenous = positionalContextTable({
      value: payload.predict_window_exogenous,
      label: "predict-window exogenous",
      expectedTimestamps: predictionTimestamps,
    })
    return {
      evidence_candidate_id: candidateId,
      candidate_type: "context",
      adapter: "context_adapter",
      content: input.candidate,
      auxiliary: {
        lookback_window_exogenous: lookbackWindowExogenous,
        predict_window_exogenous: predictWindowExogenous,
      },
    }
  }
  return {
    evidence_candidate_id: candidateId,
    candidate_type: "knowledge",
    adapter: "knowledge_adapter",
    content: input.candidate,
    auxiliary: {},
  }
}

/** @internal Exported for provisional evidence-fusion and UnifiedReasoner boundary tests. */
export function buildUnifiedEvidenceReasoningInput(input: {
  state: z.infer<typeof CriticPredictionState>
  initialState: z.infer<typeof CriticPredictionState>
  selectedCandidate: Record<string, unknown>
  allCandidates: unknown
  lookbackTargetWindow: unknown
  exogenousHistorySteps?: number
  experienceRuleReference: z.infer<typeof ExperienceRuleReference> | null
}) {
  const selectedId = input.selectedCandidate.evidence_candidate_id
  if (typeof selectedId !== "string" || !selectedId) {
    throw new Error("Selected evidence candidate is missing its id")
  }
  if (input.state.evidences.some((item) => item.evidence_candidate_id === selectedId)) {
    throw new Error("Selected evidence candidate is already committed")
  }
  const candidates = Object.values(isRecord(input.allCandidates) ? input.allCandidates : {}).filter(isRecord)
  const byId = new Map(
    candidates
      .filter((candidate) => typeof candidate.evidence_candidate_id === "string")
      .map((candidate) => [String(candidate.evidence_candidate_id), candidate]),
  )
  const committedCandidates = input.state.evidences.map((evidence) => {
    const candidate = byId.get(evidence.evidence_candidate_id)
    if (!candidate) throw new Error(`Committed evidence candidate ${evidence.evidence_candidate_id} is unavailable`)
    if (candidate.candidate_type !== evidence.candidate_type) {
      throw new Error(`Committed evidence candidate ${evidence.evidence_candidate_id} changed type`)
    }
    return candidate
  })
  const selectedType = input.selectedCandidate.candidate_type
  if (!(["experience", "context", "knowledge"] as unknown[]).includes(selectedType)) {
    throw new Error("Selected evidence candidate has an unknown type")
  }
  const attachments = [...committedCandidates, input.selectedCandidate].map((candidate) =>
    adaptEvidenceCandidate({
      candidate,
      state: input.state,
      initialState: input.initialState,
      lookbackTargetWindow: input.lookbackTargetWindow,
      exogenousHistorySteps: input.exogenousHistorySteps,
      experienceRuleReference: input.experienceRuleReference,
    }),
  )
  return {
    critic_prediction_state: input.state,
    initial_prediction_state: input.initialState,
    provisional_evidence_fusion: {
      version: 1,
      status: "provisional",
      base_committed_evidence_ids: input.state.evidences.map((item) => item.evidence_candidate_id),
      selected_evidence_candidate_id: selectedId,
      fused_evidence_ids: attachments.map((item) => item.evidence_candidate_id),
      evidence_attachments: attachments,
      fusion_contract: {
        preserve_provenance: true,
        deduplicate_dependent_support: true,
        surface_scope_mismatches_and_conflicts: true,
        no_state_mutation_before_proposal_acceptance: true,
      },
    },
    lookback_window_target: input.lookbackTargetWindow,
    reasoning_contract: {
      reasoner_type: "unified",
      output_role: "Generate a prediction proposal only; state mutation is performed by a separate reducer.",
      current_state_is_transition_baseline: true,
      estimate_only_residual_effect_not_already_committed: true,
      unchanged_current_prediction_is_valid: true,
      selected_candidate_is_only_new_evidence: true,
      jointly_reason_over_fused_evidence: true,
    },
  }
}

async function isolatedUnifiedReasoner(
  ctx: Tool.Context,
  reasoningContext: string,
  windowIndex: number,
  expectedPointCount: number,
  cancellationGroup: ChildSessionCancellationGroup,
  frozenIdentity?: FrozenLlmIdentity,
  audit?: WindowLlmAuditWriter,
): Promise<z.infer<typeof LlmResponse>> {
  throwIfAborted(ctx.abort)
  const model = frozenIdentity ?? (await resolveLlmIdentity(ctx))
  const child = await Session.create({
    parentID: ctx.sessionID,
    title: `Window ${windowIndex.toString().padStart(6, "0")} unified evidence reasoning`,
    permission: Permission.fromConfig({ "*": "deny" }),
  })
  cancellationGroup.track(child.id)
  const system = UNIFIED_REASONING_SYSTEM
  let responseText: string | null = null
  let responseParts: unknown[] | null = null
  try {
    const result = await withWindowReflectionTimeout(
      SessionPrompt.prompt({
        sessionID: child.id,
        messageID: MessageID.ascending(),
        model,
        agent: "window-reflection",
        system,
        tools: {},
        maxOutputTokens: windowReflectionPredictionOutputTokenLimit(),
        parts: [{ type: "text", text: reasoningContext }],
      }),
      {
        timeoutMs: windowReflectionLlmTimeoutMs(),
        label: "Unified-reasoning LLM request",
        onTimeout: () => SessionPrompt.cancel(child.id),
      },
    )
    responseParts = result.parts
    if (result.info.role !== "assistant") {
      throw new WindowReflectionRequestError("Unified reasoner returned a non-assistant response", "format", true)
    }
    if (result.info.error) {
      const failure = classifyWindowReflectionLlmError(result.info.error)
      throw new WindowReflectionRequestError(failure.message, failure.category, failure.retryable, failure.failureKind)
    }
    responseText = result.parts
      .map((part) => (part.type === "text" ? part.text : ""))
      .filter(Boolean)
      .join("\n")
      .trim()
    const recovered = (() => {
      try {
        return parseStructuredResponse(responseText, responseParts, (text) =>
          parseWindowReflectionText(text, windowIndex, expectedPointCount),
        )
      } catch (error) {
        throw structuredResponseError(error, responseText, responseParts)
      }
    })()
    const parsed = recovered.value
    await audit?.({
      call_type: "unified_reasoning",
      system_prompt: system,
      request: JSON.parse(reasoningContext),
      status: "success",
      child_session_id: child.id,
      raw_response: recovered.text,
      parsed_response: parsed,
      error: null,
      provider_response_parts: responseParts,
      structured_response_source: recovered.source,
      rationale: parsed.rationale,
    })
    return parsed
  } catch (error) {
    const failure = requestFailure(error)
    await audit?.({
      call_type: "unified_reasoning",
      system_prompt: system,
      request: JSON.parse(reasoningContext),
      status: "error",
      child_session_id: child.id,
      raw_response: responseText,
      parsed_response: null,
      error: failure.message,
      failure_kind: failure.failureKind,
      provider_response_parts: responseParts,
    })
    throw failure
  } finally {
    cancellationGroup.release(child.id)
  }
}

type PredictionGenerator = "unified_reasoner"

async function generatePredictionProposal(input: {
  ctx: Tool.Context
  selectedCandidate: Record<string, unknown>
  state: z.infer<typeof CriticPredictionState>
  reasoningInput: ReturnType<typeof buildUnifiedEvidenceReasoningInput>
  windowIndex: number
  cancellationGroup: ChildSessionCancellationGroup
  frozenIdentity?: FrozenLlmIdentity
  audit?: WindowLlmAuditWriter
}): Promise<{
  predictions: z.infer<typeof CriticPredictionPoint>[]
  generator: PredictionGenerator
  rationale: string
  citedEvidenceIds: string[]
}> {
  const reasoningContext = JSON.stringify(input.reasoningInput)
  const response = await isolatedUnifiedReasoner(
    input.ctx,
    reasoningContext,
    input.windowIndex,
    input.state.predictions.length,
    input.cancellationGroup,
    input.frozenIdentity,
    input.audit,
  )
  const selectedId = String(input.selectedCandidate.evidence_candidate_id)
  const allowedEvidenceIds = new Set([...input.state.evidences.map((item) => item.evidence_candidate_id), selectedId])
  if (!response.evidence_ids.includes(selectedId)) {
    throw new Error("Unified reasoner did not cite its selected evidence candidate")
  }
  if (response.evidence_ids.some((id) => !allowedEvidenceIds.has(id))) {
    throw new Error("Unified reasoner cited evidence outside its fused reasoning boundary")
  }
  let predictions: z.infer<typeof CriticPredictionPoint>[]
  try {
    predictions = normalizeProposalLayout(response.points, input.state.predictions)
  } catch (error) {
    throw new WindowReflectionRequestError(
      error instanceof Error ? error.message : String(error),
      "format",
      true,
      "schema_failed",
    )
  }
  return {
    predictions,
    generator: "unified_reasoner",
    rationale: response.rationale,
    citedEvidenceIds: response.evidence_ids,
  }
}

async function processEvidenceLoopWindow(input: {
  ctx: Tool.Context
  window: PreparedResult["windows"][number]
  blindContext: string
  prepared: PreparedResult
  preparedSha256: string
  correctionsDir: string
  maxEvidenceIterations: number
  maxFormatRetries: number
  apiLimit: ConcurrencyLimiter
  checkLimit: ConcurrencyLimiter
  cancellationGroup: ChildSessionCancellationGroup
  frozenIdentity?: FrozenLlmIdentity
}): Promise<WindowOutcome> {
  const windowIndex = input.window.window_index
  const forcedDepth = forcedEvidenceLoopDepth()
  if (forcedDepth && forcedDepth > input.maxEvidenceIterations) {
    throw new Error(
      `Forced Evidence Loop depth ${forcedDepth} exceeds the configured maximum ${input.maxEvidenceIterations}`,
    )
  }
  const prefix = `window-${windowIndex.toString().padStart(6, "0")}`
  const statesDir = path.join(input.correctionsDir, "states", prefix)
  const transitionsDir = path.join(input.correctionsDir, "transitions", prefix)
  const seekingDir = path.join(input.correctionsDir, "seeking", prefix)
  const acceptedPath = path.join(input.correctionsDir, `${prefix}.json`)
  const llmAudit = await createWindowLlmAuditWriter(input.correctionsDir, windowIndex)
  const blind = JSON.parse(input.blindContext) as Record<string, unknown>
  const allCandidates: Record<string, unknown> = isRecord(blind.evidence_candidates)
    ? { ...blind.evidence_candidates }
    : {}
  const lookbackTargetWindow = requireCompleteLookbackTargetWindow(blind)
  const exogenousHistorySteps = (await taskConfig()).exogenous_history_steps
  const candidateStorage: Record<string, unknown> = isRecord(blind.evidence_candidate_storage)
    ? { ...blind.evidence_candidate_storage }
    : {}
  const initialArtifact = CriticStateArtifact.parse(input.window.critic_prediction_state_history.initial)
  const initialState = CriticPredictionState.parse(await readJsonFile<unknown>(projectOrAbsolute(initialArtifact.path)))
  if ((await computeFileSha256(projectOrAbsolute(initialArtifact.path))) !== initialArtifact.sha256) {
    throw new Error("Initial Critic prediction state changed after preparation")
  }
  const ruleReferenceArtifact = input.window.experience_rule_reference_artifact
  let experienceRuleReference = ruleReferenceArtifact
    ? ExperienceRuleReference.parse(await readJsonFile<unknown>(projectOrAbsolute(ruleReferenceArtifact.path)))
    : null
  if (
    ruleReferenceArtifact &&
    (await computeFileSha256(projectOrAbsolute(ruleReferenceArtifact.path))) !== ruleReferenceArtifact.sha256
  ) {
    throw new Error("Experience rule reference changed after preparation")
  }
  if (experienceRuleReference) {
    const experienceStorage = candidateStorage.experience
    if (
      !isRecord(experienceStorage) ||
      experienceRuleReference.experience_candidate_artifact.path !== experienceStorage.path ||
      experienceRuleReference.experience_candidate_artifact.sha256 !== experienceStorage.sha256
    ) {
      throw new Error("Experience rule reference is not bound to the prepared experience candidate")
    }
  }
  if (forcedDepth && !isRecord(allCandidates.experience)) {
    const forcedCandidateId = "experience:forced-no-matching-rule"
    const forcedCandidatePath = path.join(input.correctionsDir, "forced-evidence", `${prefix}-experience.json`)
    await writeJsonFile(forcedCandidatePath, {
      version: 1,
      evidence_candidate_id: forcedCandidateId,
      candidate_type: "experience",
      scope: "forced-depth-control",
      payload: {
        matched_rule: false,
        instruction:
          "No locked validation-back-accepted rule matches this sample. This is negative experience evidence: preserve the current prediction unless independently committed context or knowledge supports a change.",
      },
    })
    const forcedCandidateArtifact = {
      path: forcedCandidatePath,
      sha256: await computeFileSha256(forcedCandidatePath),
    }
    allCandidates.experience = await readJsonFile<unknown>(forcedCandidatePath)
    candidateStorage.experience = forcedCandidateArtifact
    experienceRuleReference = {
      version: 1,
      window_index: windowIndex,
      semantics: "locked_rules_applied_to_initial_prediction_only",
      experience_candidate_id: forcedCandidateId,
      experience_candidate_artifact: forcedCandidateArtifact,
      initial_predictions: initialState.predictions,
      predictions: initialState.predictions,
      reasoning_contract: {
        reference_only: true,
        not_a_state_transition: true,
        do_not_add_reference_delta_to_current: true,
      },
    }
  }
  let state = initialState
  let stateArtifact = initialArtifact
  let resumedFromCurrent = false
  const currentPointerPath = path.join(statesDir, "current.json")
  const loopCheckpointPath = path.join(statesDir, "loop-checkpoint.json")
  if (await Filesystem.exists(currentPointerPath)) {
    const currentArtifact = CriticStateArtifact.parse(await readJsonFile<unknown>(currentPointerPath))
    if ((await computeFileSha256(projectOrAbsolute(currentArtifact.path))) !== currentArtifact.sha256) {
      throw new Error("Current Critic state changed after it was committed")
    }
    const resumedState = CriticPredictionState.parse(
      await readJsonFile<unknown>(projectOrAbsolute(currentArtifact.path)),
    )
    if (resumedState.window_index !== windowIndex) throw new Error("Current Critic state belongs to another window")
    for (const evidence of resumedState.evidences) {
      const storage = candidateStorage[evidence.candidate_type]
      if (
        !isRecord(storage) ||
        storage.path !== evidence.artifact_path ||
        storage.sha256 !== evidence.artifact_sha256 ||
        (await computeFileSha256(projectOrAbsolute(evidence.artifact_path))) !== evidence.artifact_sha256
      ) {
        throw new Error("Current Critic state contains invalid evidence provenance")
      }
    }
    state = resumedState
    stateArtifact = currentArtifact
    resumedFromCurrent = true
    if (needsInitialEvidenceAssessmentRetry(state)) {
      state = initialState
      stateArtifact = initialArtifact
      resumedFromCurrent = false
    }
  }
  let llmCalls = 0
  let acceptedTransitions = state.evidences.length
  let formatFailures = 0
  const restoredCheckpoint = EvidenceLoopCheckpoint.safeParse(
    await readJsonFile<unknown>(loopCheckpointPath).catch(() => null),
  )
  const checkpointMatchesState =
    restoredCheckpoint.success &&
    restoredCheckpoint.data.window_index === windowIndex &&
    restoredCheckpoint.data.state_sha256 === stateArtifact.sha256
  const reopenAbnormalCheckpoint =
    checkpointMatchesState && shouldReopenAbnormalEvidenceLoopCheckpoint(restoredCheckpoint.data)
  const reopenForcedCheckpoint =
    checkpointMatchesState && shouldReopenForcedEvidenceLoopCheckpoint(restoredCheckpoint.data)
  const seekingTrace: Array<Record<string, unknown>> = checkpointMatchesState
    ? [...restoredCheckpoint.data.seeking_trace]
    : []
  const rejectedTransitions: Array<Record<string, unknown>> = checkpointMatchesState
    ? [...restoredCheckpoint.data.rejected_transitions]
    : []
  let attemptedForState = new Set<string>(
    checkpointMatchesState && !reopenAbnormalCheckpoint && !reopenForcedCheckpoint
      ? restoredCheckpoint.data.attempted_candidate_ids
      : [],
  )

  const persistLoopCheckpoint = async (
    phase: z.infer<typeof EvidenceLoopCheckpoint>["phase"],
    selectedCandidateId: string | null = null,
    terminationReason: string | null = null,
  ): Promise<void> => {
    await writeJsonFile(loopCheckpointPath, {
      version: 1,
      window_index: windowIndex,
      state_sha256: stateArtifact.sha256,
      phase,
      selected_candidate_id: selectedCandidateId,
      attempted_candidate_ids: [...attemptedForState].sort(),
      seeking_trace: seekingTrace,
      rejected_transitions: rejectedTransitions,
      accepted_transitions: acceptedTransitions,
      termination_reason: terminationReason,
      predictability:
        terminationReason === null
          ? null
          : ["evidence_exhausted", "no_valid_transition"].includes(terminationReason) && state.evidence_gap === 1
            ? "unpredictable"
            : "predictable",
      updated_at: new Date().toISOString(),
    } satisfies z.infer<typeof EvidenceLoopCheckpoint>)
  }

  const checkProposal = async (
    predictions: z.infer<typeof CriticPredictionPoint>[],
    evidenceIds: string[],
    label: string,
  ): Promise<z.infer<typeof CheckResult>> => {
    const points = normalizeProposalLayout(predictions, state.predictions)
    const responsePath = path.join(transitionsDir, `${label}-proposal.json`)
    await writeJsonFile(responsePath, {
      window_index: windowIndex,
      points,
      rationale: `Adaptive evidence-loop proposal from ${evidenceIds.join(", ") || "the current state"}.`,
      evidence_ids: evidenceIds,
    })
    return {
      accepted: true,
      format_retry: false,
      format_errors: [],
      detector_report: {
        check: "format_only",
        expected_points: state.predictions.length,
        parsed_points: points.length,
        semantic_safety_policy_applied: false,
      },
      points,
    }
  }

  const finalizeEvidenceLoopState = async (
    reason: EvidenceLoopTerminationReason,
    predictability: EvidenceLoopPredictability = "predictable",
  ): Promise<WindowOutcome> => {
    await persistLoopCheckpoint("terminated", null, reason)
    const evidenceIds = state.evidences.map((item) => item.evidence_candidate_id)
    const checked = await checkProposal(state.predictions, evidenceIds, "state-terminal")
    await writeJsonFile(acceptedPath, {
      ...checked,
      window_index: windowIndex,
      assessment_stage: "evidence_loop_terminal",
      rationale: "Persist the committed Evidence Loop terminal prediction without post-loop Safety Reflection.",
      evidence_ids: evidenceIds,
      critic_prediction_state: state,
      critic_prediction_state_artifact: stateArtifact,
      critic_prediction_state_history: {
        ...input.window.critic_prediction_state_history,
        current: stateArtifact,
      },
      evidence_loop: {
        termination_reason: reason,
        predictability,
        accepted_transitions: acceptedTransitions,
        max_evidence_iterations: input.maxEvidenceIterations,
        seeking_trace: seekingTrace,
        rejected_transitions: rejectedTransitions,
      },
      audit_trail: {
        state_artifacts: [
          initialArtifact.path,
          ...(await readdir(statesDir).catch(() => []))
            .filter((name) => /^state-[a-f0-9]{64}\.json$/.test(name))
            .toSorted()
            .map((name) => path.join(statesDir, name)),
        ],
        llm_call_artifacts: await windowLlmAuditArtifacts(input.correctionsDir, windowIndex),
        hidden_chain_of_thought_recorded: false,
        recorded_reasoning:
          "All provider-returned response parts, including visible reasoning, plus structured rationale, errors, and retries. Provider-hidden chain-of-thought is unavailable.",
      },
    })
    return {
      window_index: windowIndex,
      status: "accepted",
      accepted: true,
      rounds_attempted: acceptedTransitions,
      llm_calls: llmCalls,
      provider_retries: 0,
      format_failures: formatFailures,
      message: null,
    }
  }

  const terminateEvidenceLoopFailure = async (error: unknown): Promise<WindowOutcome> => {
    const failure = requestFailure(error)
    await persistLoopCheckpoint("terminated", null, evidenceLoopFailureReason(failure))
    return {
      window_index: windowIndex,
      status: failure.failureKind,
      accepted: false,
      rounds_attempted: acceptedTransitions,
      llm_calls: llmCalls,
      provider_retries: 0,
      format_failures: formatFailures,
      message: failure.message,
    }
  }

  const restoredTerminalReason = EvidenceLoopTerminationReason.safeParse(
    checkpointMatchesState &&
      !reopenAbnormalCheckpoint &&
      !reopenForcedCheckpoint &&
      restoredCheckpoint.success &&
      ["safety_reflection", "terminated"].includes(restoredCheckpoint.data.phase)
      ? restoredCheckpoint.data.termination_reason
      : null,
  )
  if (resumedFromCurrent && restoredTerminalReason.success) {
    const predictability =
      checkpointMatchesState && restoredCheckpoint.success && restoredCheckpoint.data.predictability
        ? restoredCheckpoint.data.predictability
        : ["evidence_exhausted", "no_valid_transition"].includes(restoredTerminalReason.data) &&
            state.evidence_gap === 1
          ? "unpredictable"
          : "predictable"
    return finalizeEvidenceLoopState(restoredTerminalReason.data, predictability)
  }

  if (forcedDepth && acceptedTransitions < forcedDepth && state.evidence_gap === 0) {
    ;({ state, artifact: stateArtifact } = await commitCriticGap({
      statesDir,
      previousState: state,
      previousArtifact: stateArtifact,
      evidenceGap: 1,
      neededEvidence: `Forced depth control requires ${forcedDepth - acceptedTransitions} additional evidence transition(s).`,
    }))
    await persistLoopCheckpoint("ready")
  }

  if (!resumedFromCurrent) {
    try {
      const gap = await input.apiLimit(() =>
        isolatedEvidenceGapAssessment(
          input.ctx,
          state,
          allCandidates,
          lookbackTargetWindow,
          windowIndex,
          input.cancellationGroup,
          input.frozenIdentity,
          llmAudit,
          () => {
            llmCalls++
          },
        ),
      )
      ;({ state, artifact: stateArtifact } = await commitCriticGap({
        statesDir,
        previousState: state,
        previousArtifact: stateArtifact,
        evidenceGap: forcedDepth ? 1 : gap.evidence_gap,
        neededEvidence: forcedDepth
          ? `Forced depth control requires ${forcedDepth} additional evidence transition(s).`
          : gap.needed_evidence,
      }))
      await persistLoopCheckpoint("ready")
    } catch (error) {
      ;({ state, artifact: stateArtifact } = await commitCriticGap({
        statesDir,
        previousState: state,
        previousArtifact: stateArtifact,
        evidenceGap: 1,
        neededEvidence: INITIAL_EVIDENCE_ASSESSMENT_FAILURE,
      }))
      await persistLoopCheckpoint("ready")
      return terminateEvidenceLoopFailure(error)
    }
  }

  if (state.evidence_gap === 0) {
    return finalizeEvidenceLoopState("evidence_sufficient")
  }

  const finishExhaustedState = async (
    reason: Extract<EvidenceLoopTerminationReason, "evidence_exhausted" | "no_valid_transition">,
  ): Promise<WindowOutcome> => {
    if (forcedDepth && acceptedTransitions < forcedDepth) {
      return terminateEvidenceLoopFailure(
        new WindowReflectionRequestError(
          `Forced Evidence Loop depth ${forcedDepth} could not be reached from the current checkpoint`,
          "format",
          true,
          "schema_failed",
        ),
      )
    }
    let gap: z.infer<typeof EvidenceGapResponse>
    try {
      await persistLoopCheckpoint("reassessing")
      gap = await input.apiLimit(() =>
        isolatedEvidenceGapAssessment(
          input.ctx,
          state,
          allCandidates,
          lookbackTargetWindow,
          windowIndex,
          input.cancellationGroup,
          input.frozenIdentity,
          llmAudit,
          () => {
            llmCalls++
          },
        ),
      )
    } catch (error) {
      return terminateEvidenceLoopFailure(error)
    }
    ;({ state, artifact: stateArtifact } = await commitCriticGap({
      statesDir,
      previousState: state,
      previousArtifact: stateArtifact,
      evidenceGap: gap.evidence_gap,
      neededEvidence: gap.needed_evidence,
    }))
    await persistLoopCheckpoint("ready")
    return gap.evidence_gap === 0
      ? finalizeEvidenceLoopState("evidence_sufficient")
      : finalizeEvidenceLoopState(reason, "unpredictable")
  }

  while (state.evidence_gap === 1) {
    const unused = unusedEvidenceCandidates(state, allCandidates, attemptedForState)
    if (Object.keys(unused).length === 0) {
      return finishExhaustedState(rejectedTransitions.length > 0 ? "no_valid_transition" : "evidence_exhausted")
    }
    if (acceptedTransitions >= input.maxEvidenceIterations) return finalizeEvidenceLoopState("max_iterations")

    let seeking: z.infer<typeof EvidenceSeekingResponse>
    try {
      await persistLoopCheckpoint("seeking")
      seeking = await input.apiLimit(() =>
        isolatedEvidenceSeeking(
          input.ctx,
          state,
          allCandidates,
          attemptedForState,
          windowIndex,
          input.cancellationGroup,
          input.frozenIdentity,
          llmAudit,
          () => {
            llmCalls++
          },
        ),
      )
    } catch (error) {
      return terminateEvidenceLoopFailure(error)
    }
    const seekingRecord = {
      iteration: acceptedTransitions + 1,
      state_sha256: stateArtifact.sha256,
      ...seeking,
    }
    seekingTrace.push(seekingRecord)
    await writeJsonFile(
      path.join(seekingDir, `seeking-${seekingTrace.length.toString().padStart(4, "0")}.json`),
      seekingRecord,
    )
    const selectedId = seeking.selected_evidence_candidate_ids[0]
    if (!selectedId) return finishExhaustedState("evidence_exhausted")
    const selectedEntry = Object.entries(unused).find(
      ([, candidate]) => candidate?.evidence_candidate_id === selectedId,
    )
    if (!selectedEntry) {
      attemptedForState.add(selectedId)
      rejectedTransitions.push({ candidate_id: selectedId, reason: "seeker_selected_unavailable_candidate" })
      await persistLoopCheckpoint("ready")
      continue
    }
    const [candidateType, selectedCandidate] = selectedEntry as [EvidenceCandidateType, Record<string, unknown>]
    const storage = candidateStorage[candidateType]
    if (!isRecord(storage) || typeof storage.path !== "string" || typeof storage.sha256 !== "string") {
      throw new Error(`Prepared ${candidateType} candidate artifact is missing`)
    }
    const selectedEvidence = {
      evidence_candidate_id: selectedId,
      candidate_type: candidateType,
      artifact_path: storage.path,
      artifact_sha256: storage.sha256,
    }

    let proposal: z.infer<typeof CriticPredictionPoint>[]
    let predictionGenerator: PredictionGenerator
    let predictionRationale: string
    let citedEvidenceIds: string[]
    await persistLoopCheckpoint("fusing_evidence", selectedId)
    const unifiedReasoningInput = buildUnifiedEvidenceReasoningInput({
      state,
      initialState,
      selectedCandidate,
      allCandidates,
      lookbackTargetWindow,
      exogenousHistorySteps,
      experienceRuleReference,
    })
    await writeJsonFile(path.join(transitionsDir, `fusion-${seekingTrace.length.toString().padStart(4, "0")}.json`), {
      window_index: windowIndex,
      base_state_sha256: stateArtifact.sha256,
      ...unifiedReasoningInput.provisional_evidence_fusion,
    })
    let generated: Awaited<ReturnType<typeof generatePredictionProposal>> | null = null
    let generationFailure: WindowReflectionRequestError | null = null
    for (let formatAttempt = 0; formatAttempt <= input.maxFormatRetries; formatAttempt++) {
      try {
        await persistLoopCheckpoint("generating_prediction", selectedId)
        llmCalls++
        generated = await input.apiLimit(() =>
          generatePredictionProposal({
            ctx: input.ctx,
            selectedCandidate,
            state,
            reasoningInput: unifiedReasoningInput,
            windowIndex,
            cancellationGroup: input.cancellationGroup,
            frozenIdentity: input.frozenIdentity,
            audit: llmAudit,
          }),
        )
        break
      } catch (error) {
        generationFailure = requestFailure(error)
        if (generationFailure.category === "format" && formatAttempt < input.maxFormatRetries) {
          formatFailures++
          continue
        }
        break
      }
    }
    if (!generated) {
      rejectedTransitions.push({
        candidate_id: selectedId,
        candidate_type: candidateType,
        reason: generationFailure?.failureKind ?? "provider_failed",
        message: generationFailure?.message ?? "Prediction proposal generation failed",
      })
      // A transport/schema failure provides no evidence about the candidate.
      // Preserve the committed state and retry this work on the next invocation.
      return terminateEvidenceLoopFailure(generationFailure ?? new Error("Prediction proposal generation failed"))
    }
    {
      proposal = generated.predictions
      predictionGenerator = generated.generator
      predictionRationale = generated.rationale
      citedEvidenceIds = generated.citedEvidenceIds
      await writeJsonFile(
        path.join(transitionsDir, `generation-${seekingTrace.length.toString().padStart(4, "0")}-prediction.json`),
        {
          window_index: windowIndex,
          state_sha256: stateArtifact.sha256,
          selected_evidence_candidate_id: selectedId,
          prediction_generator: predictionGenerator,
          points: proposal,
          rationale: predictionRationale,
          evidence_ids: citedEvidenceIds,
        },
      )
    }

    await persistLoopCheckpoint("checking", selectedId)
    const checked = await checkProposal(
      proposal,
      citedEvidenceIds,
      `generation-${seekingTrace.length.toString().padStart(4, "0")}`,
    )
    if (!checked.accepted || !checked.points) {
      attemptedForState.add(selectedId)
      rejectedTransitions.push({
        candidate_id: selectedId,
        candidate_type: candidateType,
        reason: "schema_failed",
        detector_report: checked.detector_report,
      })
      await persistLoopCheckpoint("ready")
      continue
    }

    const proposedState = CriticPredictionState.parse({
      ...state,
      evidences: [...state.evidences, selectedEvidence],
      predictions: checked.points,
      evidence_gap: 1,
      needed_evidence: "Pending assessment of the accepted evidence transition.",
    })
    let gap: z.infer<typeof EvidenceGapResponse>
    try {
      await persistLoopCheckpoint("reassessing", selectedId)
      gap = await input.apiLimit(() =>
        isolatedEvidenceGapAssessment(
          input.ctx,
          proposedState,
          allCandidates,
          lookbackTargetWindow,
          windowIndex,
          input.cancellationGroup,
          input.frozenIdentity,
          llmAudit,
          () => {
            llmCalls++
          },
        ),
      )
    } catch (error) {
      return terminateEvidenceLoopFailure(error)
    }
    const previousStateSha256 = stateArtifact.sha256
    const nextAcceptedTransitions = acceptedTransitions + 1
    const forceGapOpen = Boolean(forcedDepth && nextAcceptedTransitions < forcedDepth)
    ;({ state, artifact: stateArtifact } = await commitCriticEvidenceTransition({
      statesDir,
      previousState: state,
      previousArtifact: stateArtifact,
      selectedEvidence,
      expectedCandidateArtifact: { path: storage.path, sha256: storage.sha256 },
      predictions: checked.points,
      evidenceGap: forceGapOpen ? 1 : gap.evidence_gap,
      neededEvidence: forceGapOpen
        ? `Forced depth control requires ${forcedDepth! - nextAcceptedTransitions} additional evidence transition(s).`
        : gap.needed_evidence,
    }))
    acceptedTransitions++
    attemptedForState = new Set()
    await persistLoopCheckpoint("ready")
    await writeJsonFile(
      path.join(transitionsDir, `transition-${acceptedTransitions.toString().padStart(4, "0")}.json`),
      {
        previous_state_sha256: previousStateSha256,
        selected_evidence: selectedEvidence,
        prediction_generator: predictionGenerator,
        resulting_state: stateArtifact,
        evidence_gap: state.evidence_gap,
        needed_evidence: state.needed_evidence,
      },
    )
  }
  return finalizeEvidenceLoopState("evidence_sufficient")
}

async function processWindow(input: {
  ctx: Tool.Context
  window: PreparedResult["windows"][number]
  blindContext: string
  prepared: PreparedResult
  preparedSha256: string
  correctionsDir: string
  maxRounds: number
  maxFormatRetries: number
  apiLimit: ConcurrencyLimiter
  checkLimit: ConcurrencyLimiter
  cancellationGroup: ChildSessionCancellationGroup
  frozenIdentity?: FrozenLlmIdentity
}): Promise<WindowOutcome> {
  if (["skill_guided_test", "evidence_loop_test"].includes(input.window.reasoning_mode)) {
    return processEvidenceLoopWindow({
      ...input,
      maxEvidenceIterations: input.maxRounds,
    })
  }
  const llmAudit = await createWindowLlmAuditWriter(input.correctionsDir, input.window.window_index)
  let priorDetectorReport: unknown = null
  let priorProposal: unknown = null
  let llmCalls = 0
  let providerRetries = 0
  let formatFailures = 0
  let lastMessage: string | null = null
  let activeBlindContext = input.blindContext
  let stageBoundBlindContext = input.blindContext
  let providerContentRiskProjection: "original" | "semantic" | "compact_semantic" = "original"
  let initialGapAssessedArtifact: { path: string; sha256: string } | null = null
  let initialGapAssessment: {
    llm_called: true
    output: 0 | 1
    needed_evidence: string
    evidence_coverage: number | null
    missing_evidence: string[]
    evidence_consistency: number | null
    evidence_conflicts: string[]
    evidence_confidence: number | null
    rationale: string | null
    error: string | null
  } | null = null
  let currentState: z.infer<typeof CriticPredictionState> | null = null
  let currentStateArtifact: { path: string; sha256: string } | null = null
  const evidenceStageAssessments: Array<{
    candidate_type: "experience" | "context" | "knowledge"
    use_current_evidence: boolean
    rationale: string
    error: string | null
  }> = []

  const persistTerminalState = async (
    state: z.infer<typeof CriticPredictionState>,
    stateArtifact: { path: string; sha256: string },
    rationale: string,
  ): Promise<boolean> => {
    const prefix = `window-${input.window.window_index.toString().padStart(6, "0")}`
    const responsePath = path.join(input.correctionsDir, `${prefix}-state-terminal.json`)
    const acceptedPath = path.join(input.correctionsDir, `${prefix}.json`)
    await writeJsonFile(responsePath, {
      window_index: input.window.window_index,
      points: state.predictions,
      rationale,
      evidence_ids: state.evidences.map((evidence) => evidence.evidence_candidate_id),
      assessment_stage: "state_terminal",
      critic_prediction_state: state,
      critic_prediction_state_artifact: stateArtifact,
    })
    const checked = CheckResult.parse(
      await input.checkLimit(() =>
        runAdjustmentPython("adacast_lite_ml.engine.reflection.controller", [
          "check-response",
          "--prepared-json",
          projectOrAbsolute(input.prepared.prepared_path),
          "--prepared-sha256",
          input.preparedSha256,
          "--response-json",
          responsePath,
          "--window-index",
          String(input.window.window_index),
          "--round",
          "0",
          "--out",
          acceptedPath,
        ]),
      ),
    )
    if (!checked.accepted) {
      priorDetectorReport = checked.detector_report
      return false
    }
    const acceptedCorrection = await readJsonFile<Record<string, unknown>>(acceptedPath)
    await writeJsonFile(acceptedPath, {
      ...acceptedCorrection,
      critic_prediction_state: state,
      critic_prediction_state_artifact: stateArtifact,
      critic_prediction_state_history: {
        ...input.window.critic_prediction_state_history,
        ...(initialGapAssessedArtifact ? { initial_gap_assessed: initialGapAssessedArtifact } : {}),
        current: stateArtifact,
      },
      initial_evidence_gap_assessment: initialGapAssessment,
      evidence_stage_assessments: evidenceStageAssessments,
    })
    return true
  }

  if (input.window.reasoning_mode === "skill_guided_test") {
    const initialArtifact = input.window.critic_prediction_state_history.initial
    if (!initialArtifact) throw new Error("Skill-guided window is missing its initial Critic prediction state")
    const initialState = CriticPredictionState.parse(
      await readJsonFile<unknown>(projectOrAbsolute(initialArtifact.path)),
    )
    const blind = JSON.parse(input.blindContext) as Record<string, unknown>
    const lookbackTargetWindow = requireCompleteLookbackTargetWindow(blind)
    let initialGap: 0 | 1 = 1
    let initialNeededEvidence = "The assessment failed, so additional relevant evidence may still be needed."
    let initialAssessmentDetails: z.infer<typeof EvidenceGapResponse> | null = null
    let initialGapError: string | null = null
    try {
      const assessment = await input.apiLimit(() =>
        isolatedEvidenceGapAssessment(
          input.ctx,
          initialState,
          blind.evidence_candidates,
          lookbackTargetWindow,
          input.window.window_index,
          input.cancellationGroup,
          input.frozenIdentity,
          llmAudit,
          () => {
            llmCalls++
          },
        ),
      )
      initialGap = assessment.evidence_gap
      initialNeededEvidence = assessment.needed_evidence.trim()
      initialAssessmentDetails = assessment
    } catch (error) {
      initialGapError = error instanceof Error ? error.message : "Initial evidence-gap assessment failed"
    }

    const assessedInitialState = CriticPredictionState.parse({
      ...initialState,
      evidence_gap: initialGap,
      needed_evidence: initialNeededEvidence,
    })
    const statePrefix = `window-${input.window.window_index.toString().padStart(6, "0")}.critic-prediction-state`
    const assessedInitialPath = path.join(input.correctionsDir, `${statePrefix}.initial-gap-assessed.json`)
    await writeJsonFile(assessedInitialPath, assessedInitialState)
    initialGapAssessedArtifact = {
      path: assessedInitialPath,
      sha256: await computeFileSha256(assessedInitialPath),
    }
    initialGapAssessment = {
      llm_called: true,
      output: initialGap,
      needed_evidence: initialNeededEvidence,
      evidence_coverage: initialAssessmentDetails?.evidence_coverage ?? null,
      missing_evidence: initialAssessmentDetails?.missing_evidence ?? [],
      evidence_consistency: initialAssessmentDetails?.evidence_consistency ?? null,
      evidence_conflicts: initialAssessmentDetails?.evidence_conflicts ?? [],
      evidence_confidence: initialAssessmentDetails?.evidence_confidence ?? null,
      rationale: initialAssessmentDetails?.rationale ?? null,
      error: initialGapError,
    }
    currentState = assessedInitialState
    currentStateArtifact = initialGapAssessedArtifact

    if (initialGap === 0) {
      const responsePath = path.join(
        input.correctionsDir,
        `window-${input.window.window_index.toString().padStart(6, "0")}-initial-prior.json`,
      )
      const acceptedPath = path.join(
        input.correctionsDir,
        `window-${input.window.window_index.toString().padStart(6, "0")}.json`,
      )
      await writeJsonFile(responsePath, {
        window_index: input.window.window_index,
        points: assessedInitialState.predictions,
        rationale: "The initial evidence-gap assessment found the prediction prior sufficiently supported.",
        evidence_ids: [],
        assessment_stage: "initial_prior",
        critic_prediction_state: assessedInitialState,
      })
      const checked = CheckResult.parse(
        await input.checkLimit(() =>
          runAdjustmentPython("adacast_lite_ml.engine.reflection.controller", [
            "check-response",
            "--prepared-json",
            projectOrAbsolute(input.prepared.prepared_path),
            "--prepared-sha256",
            input.preparedSha256,
            "--response-json",
            responsePath,
            "--window-index",
            String(input.window.window_index),
            "--round",
            "0",
            "--out",
            acceptedPath,
          ]),
        ),
      )
      if (checked.accepted) {
        const acceptedCorrection = await readJsonFile<Record<string, unknown>>(acceptedPath)
        await writeJsonFile(acceptedPath, {
          ...acceptedCorrection,
          critic_prediction_state: assessedInitialState,
          critic_prediction_state_history: {
            ...input.window.critic_prediction_state_history,
            initial_gap_assessed: initialGapAssessedArtifact,
          },
          initial_evidence_gap_assessment: initialGapAssessment,
        })
        return {
          window_index: input.window.window_index,
          status: "accepted",
          accepted: true,
          rounds_attempted: 0,
          llm_calls: llmCalls,
          provider_retries: providerRetries,
          format_failures: formatFailures,
          message: null,
        }
      }
      priorDetectorReport = checked.detector_report
      lastMessage = "Prediction prior was judged sufficient but failed the frozen deterministic safety policy"
      currentState = CriticPredictionState.parse({
        ...assessedInitialState,
        evidence_gap: 1,
        needed_evidence:
          "Additional evidence is required to produce a prediction that satisfies the frozen deterministic safety policy.",
      })
      const safetyReopenedPath = path.join(input.correctionsDir, `${statePrefix}.initial-safety-reopened.json`)
      await writeJsonFile(safetyReopenedPath, currentState)
      currentStateArtifact = {
        path: safetyReopenedPath,
        sha256: await computeFileSha256(safetyReopenedPath),
      }
    }

    if (currentState.evidence_gap === 1) {
      const candidates = isRecord(blind.evidence_candidates) ? blind.evidence_candidates : {}
      let experienceGate: z.infer<typeof EvidenceStageNeedResponse> = {
        use_current_evidence: true,
        rationale: "The stage assessment failed; conservatively retain the experience stage.",
      }
      let experienceGateError: string | null = null
      try {
        llmCalls++
        experienceGate = await input.apiLimit(() =>
          isolatedEvidenceStageNeedAssessment(
            input.ctx,
            currentState!,
            candidates.experience,
            "experience",
            input.window.window_index,
            input.cancellationGroup,
            input.frozenIdentity,
            llmAudit,
          ),
        )
      } catch (error) {
        experienceGateError = error instanceof Error ? error.message : "Experience-stage assessment failed"
      }
      evidenceStageAssessments.push({
        candidate_type: "experience",
        ...experienceGate,
        error: experienceGateError,
      })

      if (experienceGate.use_current_evidence) {
        const deterministicArtifact = input.window.critic_prediction_state_history.deterministic
        if (!deterministicArtifact) throw new Error("Skill-guided window is missing its deterministic state proposal")
        const deterministicState = CriticPredictionState.parse(
          await readJsonFile<unknown>(projectOrAbsolute(deterministicArtifact.path)),
        )
        let deterministicGap: 0 | 1 = 1
        let deterministicNeededEvidence = "The assessment failed, so additional relevant evidence may still be needed."
        try {
          const assessment = await input.apiLimit(() =>
            isolatedEvidenceGapAssessment(
              input.ctx,
              deterministicState,
              blind.evidence_candidates,
              lookbackTargetWindow,
              input.window.window_index,
              input.cancellationGroup,
              input.frozenIdentity,
              llmAudit,
              () => {
                llmCalls++
              },
            ),
          )
          deterministicGap = assessment.evidence_gap
          deterministicNeededEvidence = assessment.needed_evidence.trim()
        } catch {
          // Retain the conservative open gap and fallback explanation.
        }
        currentState = CriticPredictionState.parse({
          ...deterministicState,
          evidence_gap: deterministicGap,
          needed_evidence: deterministicNeededEvidence,
        })
        const deterministicAssessedPath = path.join(
          input.correctionsDir,
          `${statePrefix}.deterministic-gap-assessed.json`,
        )
        await writeJsonFile(deterministicAssessedPath, currentState)
        currentStateArtifact = {
          path: deterministicAssessedPath,
          sha256: await computeFileSha256(deterministicAssessedPath),
        }
        if (
          deterministicGap === 0 &&
          (await persistTerminalState(
            currentState!,
            currentStateArtifact,
            "Experience evidence closed the evidence gap; later evidence stages were skipped.",
          ))
        ) {
          return {
            window_index: input.window.window_index,
            status: "accepted",
            accepted: true,
            rounds_attempted: 0,
            llm_calls: llmCalls,
            provider_retries: providerRetries,
            format_failures: formatFailures,
            message: null,
          }
        }
      }

      if (!currentState || !currentStateArtifact) throw new Error("Critic state was not initialized")
      let contextGate: z.infer<typeof EvidenceStageNeedResponse> = {
        use_current_evidence: true,
        rationale: "The stage assessment failed; conservatively retain the context stage.",
      }
      let contextGateError: string | null = null
      try {
        llmCalls++
        contextGate = await input.apiLimit(() =>
          isolatedEvidenceStageNeedAssessment(
            input.ctx,
            currentState!,
            candidates.context,
            "context",
            input.window.window_index,
            input.cancellationGroup,
            input.frozenIdentity,
            llmAudit,
          ),
        )
      } catch (error) {
        contextGateError = error instanceof Error ? error.message : "Context-stage assessment failed"
      }
      evidenceStageAssessments.push({ candidate_type: "context", ...contextGate, error: contextGateError })
      if (
        !contextGate.use_current_evidence &&
        (await persistTerminalState(
          currentState,
          currentStateArtifact,
          "The current evidence need does not require context evidence; the context stage was skipped.",
        ))
      ) {
        return {
          window_index: input.window.window_index,
          status: "accepted",
          accepted: true,
          rounds_attempted: 0,
          llm_calls: llmCalls,
          provider_retries: providerRetries,
          format_failures: formatFailures,
          message: null,
        }
      }
      stageBoundBlindContext = bindCriticStateToBlindContext(input.blindContext, currentState, currentStateArtifact)
      activeBlindContext = stageBoundBlindContext
    }
  }

  for (let round = 1; round <= input.maxRounds; round++) {
    formatLoop: for (let formatAttempt = 0; formatAttempt <= input.maxFormatRetries; formatAttempt++) {
      let response: z.infer<typeof LlmResponse>
      let providerAttempt = 0
      while (true) {
        throwIfAborted(input.ctx.abort)
        try {
          llmCalls++
          response = await input.apiLimit(() =>
            isolatedResponse(
              input.ctx,
              activeBlindContext,
              input.window.window_index,
              input.window.horizon_indices.length,
              round,
              input.maxRounds,
              priorDetectorReport,
              priorProposal,
              input.window.reasoning_mode,
              input.cancellationGroup,
              input.frozenIdentity,
              undefined,
              llmAudit,
            ),
          )
          break
        } catch (error) {
          throwIfAborted(input.ctx.abort)
          const failure = requestFailure(error)
          lastMessage = failure.message
          await persistWindowFailure({
            correctionsDir: input.correctionsDir,
            windowIndex: input.window.window_index,
            round,
            formatAttempt,
            providerAttempt,
            failure,
            contextProjection: providerContentRiskProjection,
          })
          if (failure.category === "provider") {
            if (input.window.reasoning_mode === "skill_guided_test" && isWindowReflectionContentRisk(failure.message)) {
              if (providerContentRiskProjection === "original") {
                activeBlindContext = buildProviderContentRiskSafeContext(stageBoundBlindContext)
                providerContentRiskProjection = "semantic"
              } else if (providerContentRiskProjection === "semantic") {
                activeBlindContext = buildProviderContentRiskCompactContext(stageBoundBlindContext)
                providerContentRiskProjection = "compact_semantic"
              }
            }
            if (failure.retryable && providerAttempt < WINDOW_REFLECTION_PROVIDER_RETRY_LIMIT) {
              await delay(windowReflectionProviderRetryDelayMs(providerAttempt))
              providerAttempt++
              providerRetries++
              continue
            }
            if (
              input.window.reasoning_mode === "skill_guided_test" &&
              currentState &&
              currentStateArtifact &&
              (await persistTerminalState(
                currentState,
                currentStateArtifact,
                "The context stage failed; preserve the last assessed safe state without applying unselected evidence.",
              ))
            ) {
              return {
                window_index: input.window.window_index,
                status: "accepted",
                accepted: true,
                rounds_attempted: round,
                llm_calls: llmCalls,
                provider_retries: providerRetries,
                format_failures: formatFailures,
                message: null,
              }
            }
            return {
              window_index: input.window.window_index,
              status: failure.failureKind,
              accepted: false,
              rounds_attempted: round,
              llm_calls: llmCalls,
              provider_retries: providerRetries,
              format_failures: formatFailures,
              message: lastMessage,
            }
          }
          formatFailures++
          if (formatAttempt < input.maxFormatRetries) continue formatLoop
          if (
            input.window.reasoning_mode === "skill_guided_test" &&
            currentState &&
            currentStateArtifact &&
            (await persistTerminalState(
              currentState,
              currentStateArtifact,
              "The context response was invalid; preserve the last assessed safe state.",
            ))
          ) {
            return {
              window_index: input.window.window_index,
              status: "accepted",
              accepted: true,
              rounds_attempted: round,
              llm_calls: llmCalls,
              provider_retries: providerRetries,
              format_failures: formatFailures,
              message: null,
            }
          }
          return {
            window_index: input.window.window_index,
            status: failure.failureKind,
            accepted: false,
            rounds_attempted: round,
            llm_calls: llmCalls,
            provider_retries: providerRetries,
            format_failures: formatFailures,
            message: lastMessage,
          }
        }
      }

      const responsePath = path.join(
        input.correctionsDir,
        `window-${input.window.window_index.toString().padStart(6, "0")}-round-${round}-attempt-${formatAttempt}.json`,
      )
      if (input.window.reasoning_mode === "skill_guided_test" && input.window.evidence_candidate_ids.context) {
        response = {
          ...response,
          evidence_ids: [...new Set([...response.evidence_ids, input.window.evidence_candidate_ids.context])],
        }
      }
      await writeJsonFile(responsePath, response)
      const acceptedPath = path.join(
        input.correctionsDir,
        `window-${input.window.window_index.toString().padStart(6, "0")}.json`,
      )
      const checked = CheckResult.parse(
        await input.checkLimit(() =>
          runAdjustmentPython("adacast_lite_ml.engine.reflection.controller", [
            "check-response",
            "--prepared-json",
            projectOrAbsolute(input.prepared.prepared_path),
            "--prepared-sha256",
            input.preparedSha256,
            "--response-json",
            responsePath,
            "--window-index",
            String(input.window.window_index),
            "--round",
            String(round),
            "--out",
            acceptedPath,
          ]),
        ),
      )
      if (checked.format_retry) {
        formatFailures++
        lastMessage = checked.format_errors.join("; ") || "Deterministic response-format validation failed"
        await writeJsonFile(
          path.join(
            input.correctionsDir,
            `window-${input.window.window_index.toString().padStart(6, "0")}-round-${round}-attempt-${formatAttempt}-check-error.json`,
          ),
          {
            window_index: input.window.window_index,
            round,
            format_attempt: formatAttempt,
            category: "format",
            retryable: formatAttempt < input.maxFormatRetries,
            errors: checked.format_errors,
            recorded_at: new Date().toISOString(),
          },
        )
        if (formatAttempt < input.maxFormatRetries) continue formatLoop
        if (
          input.window.reasoning_mode === "skill_guided_test" &&
          currentState &&
          currentStateArtifact &&
          (await persistTerminalState(
            currentState,
            currentStateArtifact,
            "The context proposal failed deterministic format checks; preserve the last assessed safe state.",
          ))
        ) {
          return {
            window_index: input.window.window_index,
            status: "accepted",
            accepted: true,
            rounds_attempted: round,
            llm_calls: llmCalls,
            provider_retries: providerRetries,
            format_failures: formatFailures,
            message: null,
          }
        }
        return {
          window_index: input.window.window_index,
          status: "format_failed",
          accepted: false,
          rounds_attempted: round,
          llm_calls: llmCalls,
          provider_retries: providerRetries,
          format_failures: formatFailures,
          message: lastMessage,
        }
      }
      if (checked.accepted) {
        if (input.window.reasoning_mode === "skill_guided_test") {
          const contextualState = buildContextualCriticPredictionState(
            input.blindContext,
            checked.points ?? [],
            currentState ?? undefined,
          )
          const statePrefix = `window-${input.window.window_index.toString().padStart(6, "0")}.critic-prediction-state`
          let evidenceGap: 0 | 1 = 1
          let neededEvidence = "The assessment failed, so additional relevant evidence may still be needed."
          let gapAssessmentDetails: z.infer<typeof EvidenceGapResponse> | null = null
          let gapAssessmentError: string | null = null
          try {
            const blind = JSON.parse(input.blindContext) as Record<string, unknown>
            const assessment = await input.apiLimit(() =>
              isolatedEvidenceGapAssessment(
                input.ctx,
                contextualState,
                blind.evidence_candidates,
                requireCompleteLookbackTargetWindow(blind),
                input.window.window_index,
                input.cancellationGroup,
                input.frozenIdentity,
                llmAudit,
                () => {
                  llmCalls++
                },
              ),
            )
            evidenceGap = assessment.evidence_gap
            neededEvidence = assessment.needed_evidence.trim()
            gapAssessmentDetails = assessment
          } catch (error) {
            gapAssessmentError = error instanceof Error ? error.message : "Evidence-gap assessment failed"
          }

          const assessedState = CriticPredictionState.parse({
            ...contextualState,
            evidence_gap: evidenceGap,
            needed_evidence: neededEvidence,
          })
          const assessedStatePath = path.join(input.correctionsDir, `${statePrefix}.context-gap-assessed.json`)
          await writeJsonFile(assessedStatePath, assessedState)
          const assessedStateArtifact = {
            path: assessedStatePath,
            sha256: await computeFileSha256(assessedStatePath),
          }
          const acceptedCorrection = await readJsonFile<Record<string, unknown>>(acceptedPath)
          await writeJsonFile(acceptedPath, {
            ...acceptedCorrection,
            critic_prediction_state: assessedState,
            critic_prediction_state_history: {
              ...input.window.critic_prediction_state_history,
              ...(initialGapAssessedArtifact ? { initial_gap_assessed: initialGapAssessedArtifact } : {}),
              context: assessedStateArtifact,
              gap_assessed: assessedStateArtifact,
            },
            initial_evidence_gap_assessment: initialGapAssessment,
            evidence_stage_assessments: evidenceStageAssessments,
            evidence_gap_assessment: {
              llm_called: true,
              output: evidenceGap,
              needed_evidence: neededEvidence,
              evidence_coverage: gapAssessmentDetails?.evidence_coverage ?? null,
              missing_evidence: gapAssessmentDetails?.missing_evidence ?? [],
              evidence_consistency: gapAssessmentDetails?.evidence_consistency ?? null,
              evidence_conflicts: gapAssessmentDetails?.evidence_conflicts ?? [],
              evidence_confidence: gapAssessmentDetails?.evidence_confidence ?? null,
              rationale: gapAssessmentDetails?.rationale ?? null,
              error: gapAssessmentError,
            },
          })
        }
        return {
          window_index: input.window.window_index,
          status: "accepted",
          accepted: true,
          rounds_attempted: round,
          llm_calls: llmCalls,
          provider_retries: providerRetries,
          format_failures: formatFailures,
          message: null,
        }
      }
      priorDetectorReport = checked.detector_report
      priorProposal = checked.points ?? null
      lastMessage = "Proposal remained outside the frozen deterministic safety policy"
      break
    }
  }

  if (
    input.window.reasoning_mode === "skill_guided_test" &&
    currentState &&
    currentStateArtifact &&
    (await persistTerminalState(
      currentState,
      currentStateArtifact,
      "No context proposal passed safety checks; preserve the last assessed safe state.",
    ))
  ) {
    return {
      window_index: input.window.window_index,
      status: "accepted",
      accepted: true,
      rounds_attempted: input.maxRounds,
      llm_calls: llmCalls,
      provider_retries: providerRetries,
      format_failures: formatFailures,
      message: null,
    }
  }
  return {
    window_index: input.window.window_index,
    status: "semantic_failed",
    accepted: false,
    rounds_attempted: input.maxRounds,
    llm_calls: llmCalls,
    provider_retries: providerRetries,
    format_failures: formatFailures,
    message: lastMessage,
  }
}

async function processTriggeredWindows(
  ctx: Tool.Context,
  prepared: PreparedResult,
  preparedSha256: string,
  correctionsDir: string,
  progressPath: string,
  llmEnabled: boolean,
  maxRounds: number,
  maxFormatRetries: number,
  apiConcurrency: number,
  frozenIdentity?: FrozenLlmIdentity,
): Promise<void> {
  if (!llmEnabled) return
  const requiredWindows = prepared.windows.filter((window) => window.llm_required && window.blind_context_path)
  if (requiredWindows.length === 0) return

  const progress = new WindowReflectionProgress(
    progressPath,
    requiredWindows.map((window) => window.window_index),
    apiConcurrency,
    ctx,
  )
  const pendingWindows: typeof requiredWindows = []
  for (const window of requiredWindows) {
    const correctionPath = path.join(correctionsDir, `window-${window.window_index.toString().padStart(6, "0")}.json`)
    const correction = await readJsonFile<unknown>(correctionPath).catch(() => null)
    const expectedWindow = {
      windowIndex: window.window_index,
      horizonIndices: window.horizon_indices,
      timestamps: window.timestamps,
    }
    if (["skill_guided_test", "evidence_loop_test"].includes(window.reasoning_mode)) {
      await restoreLegacyPreSafetyCheckpoint(correction, correctionsDir, expectedWindow)
    }
    const reusable = ["skill_guided_test", "evidence_loop_test"].includes(window.reasoning_mode)
      ? await isReusableEvidenceLoopCorrection(correction, expectedWindow)
      : isReusableWindowCorrection(correction, expectedWindow)
    if (!reusable) {
      pendingWindows.push(window)
      continue
    }
    progress.restore({
      window_index: window.window_index,
      status: "resumed_accepted",
      accepted: true,
      rounds_attempted: 0,
      llm_calls: 0,
      provider_retries: 0,
      format_failures: 0,
      message: "Reused an accepted correction from the matching execution checkpoint",
    })
  }
  await progress.initialize()
  if (pendingWindows.length === 0) {
    await progress.finish("completed")
    return
  }

  const contexts = await Promise.all(
    pendingWindows.map(async (window) => {
      const blindContextBuffer = await readFile(projectOrAbsolute(window.blind_context_path!))
      if (["skill_guided_test", "evidence_loop_test"].includes(window.reasoning_mode)) {
        if (!window.blind_context_sha256) {
          throw new Error(`Evidence-loop test window ${window.window_index} is missing its blind-context hash`)
        }
        const actualContextSha256 = createHash("sha256").update(blindContextBuffer).digest("hex")
        if (actualContextSha256 !== window.blind_context_sha256) {
          throw new Error(`Evidence-loop test window ${window.window_index} blind context changed after preparation`)
        }
      }
      return { window, blindContext: blindContextBuffer.toString("utf-8") }
    }),
  )

  const apiLimit = createConcurrencyLimiter(apiConcurrency)
  const windowLimit = createConcurrencyLimiter(apiConcurrency)
  const checkLimit = createConcurrencyLimiter(WINDOW_REFLECTION_CHECK_CONCURRENCY)
  const cancellationGroup = new ChildSessionCancellationGroup(ctx.abort)

  const tasks = contexts.map(({ window, blindContext }) =>
    windowLimit(async () => {
      await progress.markRunning(window.window_index)
      const outcome = await processWindow({
        ctx,
        window,
        blindContext,
        prepared,
        preparedSha256,
        correctionsDir,
        maxRounds,
        maxFormatRetries,
        apiLimit,
        checkLimit,
        cancellationGroup,
        frozenIdentity,
      })
      await progress.markCompleted(outcome)
      return outcome
    }),
  )

  try {
    const outcomes = await Promise.all(tasks)
    const requestFailures = outcomes.filter((outcome) => windowOutcomeRequiresRerun(outcome.status))
    if (requestFailures.length > 0) {
      const examples = requestFailures
        .slice(0, 5)
        .map((outcome) => `window ${outcome.window_index}: ${outcome.status} (${outcome.message ?? "unknown error"})`)
        .join("; ")
      throw new Error(
        `LLM window reasoning did not complete for ${requestFailures.length} window(s); rerun to resume only those windows. ${examples}`,
      )
    }
    await progress.finish("completed")
  } catch (error) {
    await cancellationGroup.cancelAll()
    await Promise.allSettled(tasks)
    await progress.finish("failed", error)
    throw error
  } finally {
    cancellationGroup.dispose()
  }
}
export async function resetReflectionRunDirectory(runRoot: string, outDir: string): Promise<string> {
  const resolvedRunRoot = path.resolve(runRoot)
  const resolvedOutDir = path.resolve(outDir)
  const relative = path.relative(resolvedRunRoot, resolvedOutDir)
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error("Unsafe reflection run directory")
  }
  await rm(resolvedOutDir, { recursive: true, force: true })
  const correctionsDir = path.join(resolvedOutDir, "corrections")
  await mkdir(correctionsDir, { recursive: true })
  return correctionsDir
}

async function configure(params: z.infer<typeof parameters>) {
  const task = await taskConfig()
  await assertReflectionConfigureAllowed(task)
  const root = reflectionRoot()
  await mkdir(root, { recursive: true })
  const plannerPath = path.join(root, "planner-constraints.json")
  if (params.planner_constraints) await writeJsonFile(plannerPath, params.planner_constraints)
  const policyPath = path.join(root, "policy.json")
  const result = await runAdjustmentPython("adacast_lite_ml.engine.reflection.controller", [
    "configure",
    "--task-json",
    path.join(forecastRoot(), "task.json"),
    ...(params.planner_constraints ? ["--planner-json", plannerPath] : []),
    "--out",
    policyPath,
  ])
  const policy = ForecastWindowReflectionPolicy.parse(result)
  const policySha256 = await computeFileSha256(policyPath)
  const state = await readPhaseState()
  const updated = {
    ...state,
    reflection: {
      ...state.reflection,
      enabled: task.reflection_enabled,
      configured: true,
      policy_path: forecastPath("reflection", "policy.json"),
      policy_sha256: policySha256,
      controller_version: policy.controller_version,
      max_reflection_rounds: policy.max_reflection_rounds,
      max_format_retries: policy.max_format_retries,
      validation_records: [],
      test_records: [],
    },
  }
  await writePhaseState(
    updated,
    "## Window Reflection Policy Frozen\n\nTraining-derived checks were calibrated on validation-front and frozen before validation-back/test use.",
  )
  return result
}

async function runReflection(
  params: z.infer<typeof parameters>,
  ctx: Tool.Context,
  split: "val" | "test",
): Promise<ReflectionResult> {
  if (!params.run_id || !runIdIsSafe(params.run_id)) throw new Error(`${split} reflection requires a safe run_id`)
  const sourceRunId = params.run_id
  if (params.adjustment_candidate_id && !runIdIsSafe(params.adjustment_candidate_id)) {
    throw new Error(`Invalid adjustment_candidate_id "${params.adjustment_candidate_id}"`)
  }
  let state = await readPhaseState()
  const task = await taskConfig()
  if (!task.reflection_enabled) {
    throw new Error("Independent window reflection is disabled in the frozen task")
  }
  const requiredPhase = split === "val" ? "forecast" : "post-forecast"
  if (state.phase !== requiredPhase) {
    throw new Error(
      split === "val"
        ? "Window reflection validation is only allowed during the forecast phase"
        : "Window reflection test execution is only allowed during the post-forecast phase",
    )
  }
  if (!state.reflection.configured || !state.reflection.policy_path) {
    throw new Error("Window reflection policy must be frozen before entering the forecast phase")
  }
  const policy = await validateWindowReflectionPolicyArtifact(forecastRoot(), state.reflection.policy_path, {
    enabled: true,
    controllerVersion: state.reflection.controller_version,
    maxReflectionRounds: state.reflection.max_reflection_rounds,
    maxFormatRetries: state.reflection.max_format_retries,
    policySha256: state.reflection.policy_sha256,
  })
  if (!policy) throw new Error("Window reflection policy is missing, inconsistent, or outside the frozen directory")
  const resolvedPolicyPath = path.join(reflectionRoot(), "policy.json")
  const candidateId = params.adjustment_candidate_id ?? null
  const existingRecord =
    split === "val"
      ? state.reflection.validation_records.find(
          (record) => record.validation_run_id === params.run_id && record.adjustment_candidate_id === candidateId,
        )
      : state.reflection.test_records.find((record) => record.test_run_id === params.run_id)
  const existingArtifact = existingRecord
    ? await validateWindowReflectionRecordArtifact(forecastRoot(), existingRecord, split)
    : null
  if (existingArtifact?.auditPassed) {
    const existingManifestPath = path.join(
      reflectionRoot(),
      split === "val" ? "validation" : "test",
      params.run_id,
      candidateId ?? "raw",
      "execution-manifest.json",
    )
    const existingManifest = ReflectionExecutionManifest.safeParse(
      await readJsonFile<unknown>(existingManifestPath).catch(() => null),
    )
    if (!existingManifest.success) {
      throw new Error(`Window reflection ${split} result is already complete and cannot be rerun`)
    }
  }

  if (split === "val") {
    await assertValidationRunReady(params.run_id)
  }
  let validationStatus = "pending"
  let llmEnabled: boolean = true
  if (split === "test") {
    if (!params.validation_run_id || !runIdIsSafe(params.validation_run_id)) {
      throw new Error("run_test requires the matching validation_run_id")
    }
    await assertTestRunMatchesValidation(params.run_id, params.validation_run_id)
    const validation = state.reflection.validation_records.find(
      (record) =>
        record.validation_run_id === params.validation_run_id &&
        record.adjustment_candidate_id === (params.adjustment_candidate_id ?? null),
    )
    if (!validation || validation.status === "pending" || validation.status === "failed") {
      throw new Error("No completed matching validation reflection record exists for the selected test model")
    }
    const validatedArtifact = await validateWindowReflectionRecordArtifact(forecastRoot(), validation, "val")
    if (!validatedArtifact?.auditPassed || validatedArtifact.artifact.status === "completed") {
      throw new Error("Matching validation reflection artifact is missing, inconsistent, or failed its leakage audit")
    }
    validationStatus = validatedArtifact.artifact.status
    llmEnabled = validationStatus !== "disabled_negative_mse"
  }
  const baseline = await baselinePaths(split, params.run_id, params.adjustment_candidate_id)
  const evidenceLoopTest = split === "test"
  const skillGuidedTest = split === "test" && baseline.baselineKind === "skill_adjusted"
  if (evidenceLoopTest) {
    // The matching validation result gates the selected test baseline, while every
    // test window still runs the test-only adaptive evidence protocol.
    llmEnabled = true
  }
  const sourceGapPredictions = path.join(forecastRoot(), "runs", params.run_id, "gap_predictions.csv")
  const gapPredictions =
    split === "test" && task.forecast_gap > 0 && (await Filesystem.exists(sourceGapPredictions))
      ? sourceGapPredictions
      : null
  if (evidenceLoopTest && task.forecast_gap > 0 && !gapPredictions) {
    throw new Error("Test evidence-loop reasoning requires gap_predictions.csv from the source test run")
  }
  const apiConcurrency = evidenceLoopTest ? resolveWindowReflectionTestApiConcurrency() : 1
  let frozenIdentity: FrozenLlmIdentity | undefined
  if (llmEnabled) frozenIdentity = await resolveLlmIdentity(ctx)
  const executionSignatureSha256 = await reflectionExecutionSignature({
    split,
    runId: params.run_id,
    candidateId: params.adjustment_candidate_id ?? null,
    task,
    policyPath: resolvedPolicyPath,
    baseline,
    gapPredictions,
    llmEnabled,
    apiConcurrency,
    maxRounds: state.reflection.max_reflection_rounds,
    maxFormatRetries: state.reflection.max_format_retries,
    llmIdentity: frozenIdentity,
  })
  const legacyExecutionSignatureSha256 = evidenceLoopTest
    ? await reflectionExecutionSignature({
        split,
        runId: params.run_id,
        candidateId: params.adjustment_candidate_id ?? null,
        task,
        policyPath: resolvedPolicyPath,
        baseline,
        gapPredictions,
        llmEnabled,
        apiConcurrency: LEGACY_WINDOW_REFLECTION_TEST_API_CONCURRENCY,
        maxRounds: state.reflection.max_reflection_rounds,
        maxFormatRetries: state.reflection.max_format_retries,
        llmIdentity: frozenIdentity,
      })
    : null
  const legacyPredictionOutputSignatureSha256 = evidenceLoopTest
    ? await reflectionExecutionSignature({
        split,
        runId: params.run_id,
        candidateId: params.adjustment_candidate_id ?? null,
        task,
        policyPath: resolvedPolicyPath,
        baseline,
        gapPredictions,
        llmEnabled,
        apiConcurrency,
        maxRounds: state.reflection.max_reflection_rounds,
        maxFormatRetries: state.reflection.max_format_retries,
        llmIdentity: frozenIdentity,
        predictionOutputTokenLimit: 32_768,
      })
    : null
  const legacyDefaultRuntimeSignatureSha256 = evidenceLoopTest
    ? await reflectionExecutionSignature({
        split,
        runId: params.run_id,
        candidateId: params.adjustment_candidate_id ?? null,
        task,
        policyPath: resolvedPolicyPath,
        baseline,
        gapPredictions,
        llmEnabled,
        apiConcurrency: WINDOW_REFLECTION_TEST_API_CONCURRENCY,
        maxRounds: state.reflection.max_reflection_rounds,
        maxFormatRetries: state.reflection.max_format_retries,
        llmIdentity: frozenIdentity,
        predictionOutputTokenLimit: 32_000,
      })
    : null
  // Increasing declared output/timeout limits can resume the same immutable
  // states produced with the former 32K/5-minute limits at this concurrency.
  const legacyConfiguredRuntimeSignatureSha256 = evidenceLoopTest &&
    (windowReflectionPredictionOutputTokenLimit() !== 32_000 || windowReflectionLlmTimeoutMs() !== WINDOW_REFLECTION_LLM_TIMEOUT_MS)
    ? await reflectionExecutionSignature({
        split, runId: params.run_id, candidateId: params.adjustment_candidate_id ?? null,
        task, policyPath: resolvedPolicyPath, baseline, gapPredictions, llmEnabled, apiConcurrency,
        maxRounds: state.reflection.max_reflection_rounds,
        maxFormatRetries: state.reflection.max_format_retries,
        llmIdentity: frozenIdentity, predictionOutputTokenLimit: 32_000,
        llmTimeoutMs: WINDOW_REFLECTION_LLM_TIMEOUT_MS,
      })
    : null

  const runRoot = path.join(reflectionRoot(), split === "val" ? "validation" : "test", params.run_id)
  const candidateSegment = params.adjustment_candidate_id ?? "raw"
  const outDir = path.join(runRoot, candidateSegment)
  const reopenableExistingRun =
    evidenceLoopTest && (await hasReopenableEvidenceLoopCheckpoint(outDir))
  const executionManifestPath = path.join(outDir, "execution-manifest.json")
  const existingManifestResult = ReflectionExecutionManifest.safeParse(
    await readJsonFile<unknown>(executionManifestPath).catch(() => null),
  )
  let resumeCheckpoint =
    existingManifestResult.success &&
    (existingManifestResult.data.signature_sha256 === executionSignatureSha256 ||
      existingManifestResult.data.signature_sha256 === legacyExecutionSignatureSha256 ||
      existingManifestResult.data.signature_sha256 === legacyPredictionOutputSignatureSha256 ||
      existingManifestResult.data.signature_sha256 === legacyDefaultRuntimeSignatureSha256 ||
      existingManifestResult.data.signature_sha256 === legacyConfiguredRuntimeSignatureSha256)
  if (
    params.subcommand === "resume_test" &&
    existingManifestResult.success &&
    reopenableExistingRun &&
    forcedEvidenceLoopDepth() &&
    process.env.ADACAST_LITE_REFLECTION_CHECKPOINT_RELOCATE_ROOT
  ) {
    resumeCheckpoint = true
  }
  if (
    existingArtifact?.auditPassed &&
    (!existingManifestResult.success || resumeCheckpoint) &&
    !reopenableExistingRun
  ) {
    // Replaying a completed tool response repairs interrupted host projections
    // without invoking the model or altering any committed predictions.
    return { ...existingArtifact.artifact, result_path: projectRelative(existingArtifact.evalPath) }
  }
  if (params.subcommand === "resume_test" && !resumeCheckpoint) {
    throw new Error("No matching interrupted test reflection checkpoint exists to resume")
  }
  const stateBeforeRun = await readPhaseState()
  await writePhaseState(
    split === "val"
      ? {
          ...stateBeforeRun,
          reflection: {
            ...stateBeforeRun.reflection,
            validation_records: stateBeforeRun.reflection.validation_records.filter(
              (item) =>
                item.validation_run_id !== params.run_id ||
                item.adjustment_candidate_id !== (params.adjustment_candidate_id ?? null),
            ),
          },
        }
      : {
          ...stateBeforeRun,
          reflection: {
            ...stateBeforeRun.reflection,
            test_records: stateBeforeRun.reflection.test_records.filter((item) => item.test_run_id !== params.run_id),
          },
        },
    `## Window Reflection Started\n\n${split}:${params.run_id}/${candidateSegment} ${resumeCheckpoint ? "is resuming a matching checkpoint" : "started with a fresh execution directory"}.`,
  )
  let correctionsDir: string
  if (resumeCheckpoint) {
    correctionsDir = path.join(outDir, "corrections")
    await mkdir(correctionsDir, { recursive: true })
  } else {
    correctionsDir = await resetReflectionRunDirectory(runRoot, outDir)
  }
  const prepareReflection = async (): Promise<PreparedResult> =>
    PreparedResult.parse(
      await runAdjustmentPython("adacast_lite_ml.engine.reflection.controller", [
        "prepare",
        "--task-json",
        path.join(forecastRoot(), "task.json"),
        "--policy-json",
        resolvedPolicyPath,
        "--predictions-csv",
        baseline.predictions,
        ...(baseline.deterministicPredictions
          ? ["--deterministic-predictions-csv", baseline.deterministicPredictions]
          : []),
        ...(gapPredictions ? ["--gap-predictions-csv", gapPredictions] : []),
        ...(baseline.candidate ? ["--candidate-json", baseline.candidate] : []),
        ...(params.adjustment_candidate_id ? ["--adjustment-candidate-id", params.adjustment_candidate_id] : []),
        "--baseline-kind",
        baseline.baselineKind,
        "--source-run-id",
        sourceRunId,
        "--split",
        split,
        "--out-dir",
        outDir,
      ]),
    )
  const relocatedPreparedPath = path.join(outDir, "prepared.json")
  const useRelocatedCheckpoint = Boolean(resumeCheckpoint && process.env.ADACAST_LITE_REFLECTION_CHECKPOINT_RELOCATE_ROOT)
  let prepared = useRelocatedCheckpoint
    ? PreparedResult.parse({
        ...(await readJsonFile<Record<string, unknown>>(relocatedPreparedPath)),
        prepared_path: relocatedPreparedPath,
      })
    : await prepareReflection()
  let preparedSha256 = await computeFileSha256(projectOrAbsolute(prepared.prepared_path))
  if (
    resumeCheckpoint &&
    !useRelocatedCheckpoint &&
    existingManifestResult.success &&
    existingManifestResult.data.prepared_sha256 !== preparedSha256
  ) {
    resumeCheckpoint = false
    correctionsDir = await resetReflectionRunDirectory(runRoot, outDir)
    prepared = await prepareReflection()
    preparedSha256 = await computeFileSha256(projectOrAbsolute(prepared.prepared_path))
  }
  const startedAt =
    resumeCheckpoint && existingManifestResult.success
      ? existingManifestResult.data.started_at
      : new Date().toISOString()
  const executionManifest: ReflectionExecutionManifest = {
    version: 1,
    signature_sha256: executionSignatureSha256,
    prepared_sha256: preparedSha256,
    status: "running",
    started_at: startedAt,
    updated_at: new Date().toISOString(),
  }
  await writeJsonFile(executionManifestPath, executionManifest)
  let llmProtocolPath: string | null = null
  let llmProtocolSha256: string | null = null
  if (evidenceLoopTest && llmEnabled && prepared.llm_required_count > 0) {
    if (!frozenIdentity) throw new Error("Evidence-loop LLM identity was not frozen")
    llmProtocolPath = path.join(outDir, "llm-protocol.json")
    await writeJsonFile(llmProtocolPath, {
      protocol_version: 16,
      mode: skillGuidedTest ? "skill_guided_test" : "evidence_loop_test",
      response_transport: "strict_json_text",
      provider_id: frozenIdentity.providerID,
      model_id: frozenIdentity.modelID,
      system_prompt: UNIFIED_REASONING_SYSTEM,
      response_schema: z.toJSONSchema(LlmResponse),
      prediction_output_layout: "row_oriented_points",
      prediction_reasoning_token_guidance: WINDOW_REFLECTION_REASONING_TOKEN_GUIDANCE,
      prediction_output_token_limit: windowReflectionPredictionOutputTokenLimit(),
      evidence_seeking_system_prompt: EVIDENCE_SEEKING_SYSTEM,
      evidence_seeking_response_schema: z.toJSONSchema(EvidenceTypeSelectionResponse),
      evidence_seeking_controller_response_schema: z.toJSONSchema(EvidenceSeekingResponse),
      evidence_gap_system_prompt: EVIDENCE_GAP_SYSTEM,
      evidence_gap_response_schema: z.toJSONSchema(EvidenceGapResponse),
      evidence_gap_retry_limit: EVIDENCE_GAP_MAX_RETRIES,
      unified_reasoning_system_prompt: UNIFIED_REASONING_SYSTEM,
      // Legacy signature field retained so archived checkpoints remain resumable.
      post_loop_safety_system_prompt: LEGACY_POST_LOOP_SAFETY_SYSTEM,
      prediction_generator: "unified_reasoner_with_provisional_evidence_fusion_v1",
      prepared_sha256: preparedSha256,
      max_semantic_rounds: state.reflection.max_reflection_rounds,
      max_format_retries: state.reflection.max_format_retries,
      api_concurrency: apiConcurrency,
      checker_concurrency: WINDOW_REFLECTION_CHECK_CONCURRENCY,
      provider_retry_limit: WINDOW_REFLECTION_PROVIDER_RETRY_LIMIT,
      parent_conversation_visible: false,
      frozen_at: new Date().toISOString(),
    })
    llmProtocolSha256 = await computeFileSha256(llmProtocolPath)
  }
  try {
    await processTriggeredWindows(
      ctx,
      prepared,
      preparedSha256,
      correctionsDir,
      path.join(outDir, "window-progress.json"),
      llmEnabled,
      state.reflection.max_reflection_rounds,
      state.reflection.max_format_retries,
      apiConcurrency,
      frozenIdentity,
    )
  } catch (error) {
    await writeJsonFile(executionManifestPath, {
      ...executionManifest,
      status: "failed",
      updated_at: new Date().toISOString(),
    } satisfies ReflectionExecutionManifest).catch(() => undefined)
    throw error
  }
  if (llmProtocolPath && llmProtocolSha256 && (await computeFileSha256(llmProtocolPath)) !== llmProtocolSha256) {
    throw new Error("Frozen evidence-loop LLM protocol changed before reflection finalization")
  }
  const result = ReflectionResult.parse(
    await runAdjustmentPython("adacast_lite_ml.engine.reflection.controller", [
      "finalize",
      "--prepared-json",
      projectOrAbsolute(prepared.prepared_path),
      "--prepared-sha256",
      preparedSha256,
      "--corrections-dir",
      correctionsDir,
      "--out-dir",
      outDir,
      ...(llmProtocolPath && llmProtocolSha256
        ? ["--llm-protocol-json", llmProtocolPath, "--llm-protocol-sha256", llmProtocolSha256]
        : []),
    ]),
  )
  if (split === "val" && result.status === "completed") {
    throw new Error("Validation reflection returned a test-only completion status")
  }
  if (split === "test" && result.status !== "completed") {
    throw new Error("Test reflection returned a validation-only gate status")
  }

  const completedAt = new Date().toISOString()
  const current = await readPhaseState()
  if (split === "val") {
    const record = {
      validation_run_id: params.run_id,
      baseline_kind: baseline.baselineKind,
      adjustment_candidate_id: params.adjustment_candidate_id ?? null,
      status: result.status as
        | "enabled_no_trigger"
        | "enabled_improved"
        | "enabled_unchanged"
        | "disabled_negative_mse",
      trigger_count: result.trigger_count,
      corrected_count: result.corrected_count,
      fallback_count: result.fallback_count,
      baseline_metrics: result.baseline_metrics,
      final_metrics: result.final_metrics,
      delta_mse: result.delta_mse,
      result_path: projectRelative(projectOrAbsolute(result.result_path)),
      completed_at: completedAt,
    }
    const records = current.reflection.validation_records.filter(
      (item) =>
        item.validation_run_id !== params.run_id ||
        item.adjustment_candidate_id !== (params.adjustment_candidate_id ?? null),
    )
    await writePhaseState(
      { ...current, reflection: { ...current.reflection, validation_records: [...records, record] } },
      `## Window Reflection Validation\n\n${params.run_id}: ${result.status}, triggers=${result.trigger_count}, delta_mse=${result.delta_mse}.`,
    )
  } else {
    const record = {
      test_run_id: params.run_id,
      validation_run_id: params.validation_run_id!,
      baseline_kind: baseline.baselineKind,
      adjustment_candidate_id: params.adjustment_candidate_id ?? null,
      validation_status: validationStatus as
        | "enabled_no_trigger"
        | "enabled_improved"
        | "enabled_unchanged"
        | "disabled_negative_mse",
      llm_refinement_enabled: llmEnabled,
      trigger_count: result.trigger_count,
      skill_match_count: result.skill_match_count,
      llm_required_count: result.llm_required_count,
      corrected_count: result.corrected_count,
      fallback_count: result.fallback_count,
      deterministic_fallback_count: result.deterministic_fallback_count,
      raw_fallback_count: result.raw_fallback_count,
      baseline_metrics: result.baseline_metrics,
      final_metrics: result.final_metrics,
      result_path: projectRelative(projectOrAbsolute(result.result_path)),
      completed_at: completedAt,
    }
    const records = current.reflection.test_records.filter((item) => item.test_run_id !== params.run_id)
    await writePhaseState(
      { ...current, reflection: { ...current.reflection, test_records: [...records, record] } },
      `## Window Reflection Test\n\n${params.run_id}: validation=${validationStatus}, llm=${llmEnabled}, safety_triggers=${result.trigger_count}, skill_matches=${result.skill_match_count}, llm_windows=${result.llm_required_count}.`,
    )
  }
  await writeJsonFile(executionManifestPath, {
    ...executionManifest,
    status: "completed",
    updated_at: new Date().toISOString(),
  } satisfies ReflectionExecutionManifest)
  return result
}

async function reflectionExecutionStatus(params: z.infer<typeof parameters>): Promise<Record<string, unknown>> {
  if (!params.run_id || !runIdIsSafe(params.run_id)) throw new Error("status requires a safe run_id")
  if (params.adjustment_candidate_id && !runIdIsSafe(params.adjustment_candidate_id)) {
    throw new Error("status requires a safe adjustment_candidate_id")
  }
  const split = params.status_split ?? "test"
  const candidateSegment = params.adjustment_candidate_id ?? "raw"
  const outDir = path.join(reflectionRoot(), split === "val" ? "validation" : "test", params.run_id, candidateSegment)
  const [manifest, progress, preparedValue] = await Promise.all([
    readJsonFile<unknown>(path.join(outDir, "execution-manifest.json")).catch(() => null),
    readJsonFile<unknown>(path.join(outDir, "window-progress.json")).catch(() => null),
    readJsonFile<unknown>(path.join(outDir, "prepared.json")).catch(() => null),
  ])
  const preparedPath = path.join(outDir, "prepared.json")
  const preparedResult = PreparedResult.safeParse(
    isRecord(preparedValue) ? { ...preparedValue, prepared_path: preparedPath } : preparedValue,
  )
  if (!preparedResult.success) {
    return {
      run_id: params.run_id,
      split,
      adjustment_candidate_id: params.adjustment_candidate_id ?? null,
      execution_manifest: manifest,
      progress,
      status: manifest ? "prepared_missing_or_invalid" : "not_started",
      windows: [],
      pending_window_indices: [],
    }
  }

  const correctionsDir = path.join(outDir, "corrections")
  const windows: Array<Record<string, unknown>> = []
  for (const window of preparedResult.data.windows) {
    const required = window.llm_required && Boolean(window.blind_context_path)
    if (!required) {
      windows.push({ window_index: window.window_index, status: "not_required" })
      continue
    }
    const expectedWindow = {
      windowIndex: window.window_index,
      horizonIndices: window.horizon_indices,
      timestamps: window.timestamps,
    }
    const prefix = `window-${window.window_index.toString().padStart(6, "0")}`
    const correction = await readJsonFile<unknown>(path.join(correctionsDir, `${prefix}.json`)).catch(() => null)
    const completed = ["skill_guided_test", "evidence_loop_test"].includes(window.reasoning_mode)
      ? await isReusableEvidenceLoopCorrection(correction, expectedWindow)
      : isReusableWindowCorrection(correction, expectedWindow)
    if (completed) {
      const record = isRecord(correction) && isRecord(correction.evidence_loop) ? correction.evidence_loop : null
      windows.push({
        window_index: window.window_index,
        status: "completed",
        termination_reason: record?.termination_reason ?? null,
        predictability: record?.predictability ?? null,
        audit_trail: isRecord(correction) && isRecord(correction.audit_trail) ? correction.audit_trail : null,
      })
      continue
    }

    const statesDir = path.join(correctionsDir, "states", prefix)
    const pointer = CriticStateArtifact.safeParse(
      await readJsonFile<unknown>(path.join(statesDir, "current.json")).catch(() => null),
    )
    const checkpoint = EvidenceLoopCheckpoint.safeParse(
      await readJsonFile<unknown>(path.join(statesDir, "loop-checkpoint.json")).catch(() => null),
    )
    let currentState: z.infer<typeof CriticPredictionState> | null = null
    let stateError: string | null = null
    if (pointer.success) {
      try {
        if ((await computeFileSha256(projectOrAbsolute(pointer.data.path))) !== pointer.data.sha256) {
          throw new Error("current state hash mismatch")
        }
        currentState = CriticPredictionState.parse(await readJsonFile<unknown>(projectOrAbsolute(pointer.data.path)))
        if (currentState.window_index !== window.window_index) throw new Error("current state window mismatch")
      } catch (error) {
        stateError = error instanceof Error ? error.message : "invalid current state"
      }
    }
    const matchingCheckpoint =
      checkpoint.success && pointer.success && checkpoint.data.state_sha256 === pointer.data.sha256
        ? checkpoint.data
        : null
    const reopenAbnormalCheckpoint = matchingCheckpoint
      ? shouldReopenAbnormalEvidenceLoopCheckpoint(matchingCheckpoint)
      : false
    const terminalCheckpoint =
      matchingCheckpoint &&
      ["safety_reflection", "terminated"].includes(matchingCheckpoint.phase) &&
      EvidenceLoopTerminationReason.safeParse(matchingCheckpoint.termination_reason).success &&
      !reopenAbnormalCheckpoint
    const stateArtifacts = [
      window.critic_prediction_state_history.initial?.path,
      ...(await readdir(statesDir).catch(() => []))
        .filter((name) => /^state-[a-f0-9]{64}\.json$/.test(name))
        .toSorted()
        .map((name) => path.join(statesDir, name)),
    ].filter((value): value is string => Boolean(value))
    windows.push({
      window_index: window.window_index,
      status: stateError ? "state_invalid" : currentState ? "incomplete" : "not_started",
      current_state_sha256: pointer.success ? pointer.data.sha256 : null,
      evidence_gap: currentState?.evidence_gap ?? null,
      needed_evidence: currentState?.needed_evidence ?? null,
      used_evidence_ids: currentState?.evidences.map((item) => item.evidence_candidate_id) ?? [],
      loop_phase: matchingCheckpoint?.phase ?? null,
      termination_reason: matchingCheckpoint?.termination_reason ?? null,
      predictability: matchingCheckpoint?.predictability ?? null,
      selected_candidate_id: matchingCheckpoint?.selected_candidate_id ?? null,
      attempted_candidate_ids: matchingCheckpoint?.attempted_candidate_ids ?? [],
      accepted_transitions: matchingCheckpoint?.accepted_transitions ?? currentState?.evidences.length ?? 0,
      next_action: currentState
        ? reopenAbnormalCheckpoint
          ? "seek_evidence"
          : terminalCheckpoint
            ? "persist_evidence_loop_terminal"
            : currentState.evidence_gap === 0
              ? "persist_evidence_loop_terminal"
              : "seek_evidence"
        : "initialize",
      error: stateError,
      audit_trail: {
        state_artifacts: stateArtifacts,
        llm_call_artifacts: await windowLlmAuditArtifacts(correctionsDir, window.window_index),
        hidden_chain_of_thought_recorded: false,
        provider_visible_reasoning_recorded: true,
      },
    })
  }
  const pendingWindowIndices = windows
    .filter((window) => window.status !== "completed" && window.status !== "not_required")
    .map((window) => Number(window.window_index))
  return {
    run_id: params.run_id,
    split,
    adjustment_candidate_id: params.adjustment_candidate_id ?? null,
    status: pendingWindowIndices.length === 0 ? "completed" : "incomplete",
    execution_manifest: manifest,
    progress,
    total_windows: windows.length,
    completed_window_indices: windows
      .filter((window) => window.status === "completed")
      .map((window) => Number(window.window_index)),
    pending_window_indices: pendingWindowIndices,
    windows,
  }
}

export const ForecastWindowReflectionTool = Tool.define("forecast_window_reflection", {
  description: DESCRIPTION,
  parameters,
  async execute(params, ctx) {
    let result: unknown
    if (params.subcommand === "read") {
      result = (await readPhaseState()).reflection
    } else if (params.subcommand === "status") {
      result = await reflectionExecutionStatus(params)
    } else if (params.subcommand === "configure") {
      result = await configure(params)
    } else if (params.subcommand === "run_validation") {
      result = await runReflection(params, ctx, "val")
    } else {
      result = await runReflection(params, ctx, "test")
    }
    return {
      title: `Window reflection: ${params.subcommand}`,
      metadata: { subcommand: params.subcommand },
      output: JSON.stringify(result),
    }
  },
})
