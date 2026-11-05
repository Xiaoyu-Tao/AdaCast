import { mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises"
import { createHash } from "node:crypto"
import path from "node:path"
import z from "zod"
import { Filesystem } from "./runtime"
import { Tool } from "./runtime"
import DESCRIPTION from "./forecast-adjustment-candidate.txt"
import {
  AdjustmentCandidate,
  AdjustmentCandidateOrigin,
  AdjustmentFitRunManifest,
  AdjustmentPolicy,
  AdjustmentRule,
  AdjustmentRuleCondition,
  AdjustmentSkillManifest,
  AdjustmentSourceModelSet,
  AdjustmentUserPriorContract,
  AdjustmentUserPriorEvidence,
  parseStoredAdjustmentPolicy,
} from "./types"
import { computeSpecHash } from "./hash"
import {
  adjustmentRoot,
  ensureAdjustmentDirs,
  forecastRoot,
  projectPath,
  readJsonFile,
  readPhaseState,
  writeJsonFile,
} from "./forecast-adjustment-common"

const parameters = z.object({
  subcommand: z.enum(["write", "read", "list", "inspect_user_prior"]),
  candidate_id: z.string().optional(),
  source_model_set_id: z.string().optional(),
  parent_skill_version: z.string().nullable().optional(),
  origin: AdjustmentCandidateOrigin.optional(),
  user_prior_contract: AdjustmentUserPriorContract.optional(),
  rules: z.array(AdjustmentRule).optional(),
  skill_markdown: z.string().optional(),
  llm_provenance: z.record(z.string(), z.unknown()).optional(),
  reference_paths: z
    .array(z.string().min(1))
    .optional()
    .describe("Explicit text/code paths mentioned by adjustment_skill_priors; files are read as text and never executed"),
})

const MAX_USER_PRIOR_REFERENCE_BYTES = 512 * 1024
const MAX_USER_PRIOR_TOTAL_REFERENCE_BYTES = 1024 * 1024
const USER_PRIOR_REFERENCE_EXTENSIONS = new Set([".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml"])

function candidatesRoot(): string {
  return path.join(adjustmentRoot(), "candidates")
}

function candidateDir(candidateId: string): string {
  if (!/^[A-Za-z0-9_-]+$/.test(candidateId)) {
    throw new Error(`Invalid candidate_id "${candidateId}". Use a simple directory-safe id.`)
  }
  return path.join(candidatesRoot(), candidateId)
}

function sha256(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex")
}

function normalizePathMention(value: string): string {
  return value.trim().replace(/^['"]|['"]$/g, "").replaceAll("\\", "/").toLocaleLowerCase("en-US")
}

function userPriorEvidencePath(artifactSha256: string): string {
  return path.join(adjustmentRoot(), "user-prior", `evidence-${artifactSha256}.json`)
}

function userPriorEvidenceRelativePath(artifactSha256: string): string {
  return path.posix.join(".forecast", "adjustments", "user-prior", `evidence-${artifactSha256}.json`)
}

function userPriorEvidenceHash(input: {
  source_sha256: string
  references: Array<{ requested_path: string; resolved_path: string; sha256: string; size_bytes: number }>
}): string {
  return computeSpecHash({
    version: 1,
    source_sha256: input.source_sha256,
    references: input.references.map((reference) => ({
      requested_path: reference.requested_path,
      resolved_path: reference.resolved_path,
      sha256: reference.sha256,
      size_bytes: reference.size_bytes,
    })),
  })
}

async function inspectUserPrior(referencePaths: string[]): Promise<z.infer<typeof AdjustmentUserPriorEvidence>> {
  const task = await readTaskForCandidateValidation()
  const sourceText = String(task?.adjustment_skill_priors ?? "").trim()
  if (!sourceText) {
    throw new Error("inspect_user_prior requires task.json adjustment_skill_priors")
  }

  const sourceForMentions = normalizePathMention(sourceText)
  const uniqueReferencePaths = [...new Set(referencePaths.map((value) => value.trim()).filter(Boolean))]
  const references: z.infer<typeof AdjustmentUserPriorEvidence>["references"] = []
  let totalBytes = 0
  for (const requestedPath of uniqueReferencePaths) {
    if (!sourceForMentions.includes(normalizePathMention(requestedPath))) {
      throw new Error(
        `User-prior reference path must appear verbatim in adjustment_skill_priors before it can be read: ${requestedPath}`,
      )
    }
    const extension = path.extname(requestedPath).toLocaleLowerCase("en-US")
    if (!USER_PRIOR_REFERENCE_EXTENSIONS.has(extension)) {
      throw new Error(`Unsupported user-prior reference extension ${extension || "<none>"}; only text/code files are allowed`)
    }
    const resolvedPath = path.isAbsolute(requestedPath) ? path.normalize(requestedPath) : path.resolve(projectPath(), requestedPath)
    const metadata = await stat(resolvedPath).catch(() => null)
    if (!metadata?.isFile()) {
      throw new Error(`User-prior reference is not a readable file: ${requestedPath}`)
    }
    if (metadata.size > MAX_USER_PRIOR_REFERENCE_BYTES) {
      throw new Error(
        `User-prior reference exceeds ${MAX_USER_PRIOR_REFERENCE_BYTES} bytes and will not be loaded: ${requestedPath}`,
      )
    }
    totalBytes += metadata.size
    if (totalBytes > MAX_USER_PRIOR_TOTAL_REFERENCE_BYTES) {
      throw new Error(`User-prior references exceed the ${MAX_USER_PRIOR_TOTAL_REFERENCE_BYTES}-byte total limit`)
    }
    const bytes = await readFile(resolvedPath)
    let content: string
    try {
      content = new TextDecoder("utf-8", { fatal: true }).decode(bytes)
    } catch {
      throw new Error(`User-prior reference is not valid UTF-8 text: ${requestedPath}`)
    }
    if (content.includes("\u0000")) {
      throw new Error(`User-prior reference is not UTF-8 text: ${requestedPath}`)
    }
    references.push({
      requested_path: requestedPath,
      resolved_path: resolvedPath,
      sha256: sha256(bytes),
      size_bytes: bytes.byteLength,
      content,
    })
  }

  const sourceSha256 = sha256(sourceText)
  const artifactSha256 = userPriorEvidenceHash({ source_sha256: sourceSha256, references })
  const evidence = AdjustmentUserPriorEvidence.parse({
    version: 1,
    source_text: sourceText,
    source_sha256: sourceSha256,
    references,
    inspected_at: new Date().toISOString(),
    artifact_sha256: artifactSha256,
  })
  const evidencePath = userPriorEvidencePath(artifactSha256)
  if (!(await Filesystem.exists(evidencePath))) {
    await writeJsonFile(evidencePath, evidence)
  }
  return evidence
}

async function listCandidates() {
  const root = candidatesRoot()
  const entries = await readdir(root, { withFileTypes: true }).catch(() => [])
  return entries
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort()
}

async function readPolicy(): Promise<z.infer<typeof AdjustmentPolicy>> {
  const file = path.join(adjustmentRoot(), "policy.json")
  try {
    return parseStoredAdjustmentPolicy(await readJsonFile(file))
  } catch {
    return AdjustmentPolicy.parse({})
  }
}

async function readSourceSet(): Promise<z.infer<typeof AdjustmentSourceModelSet>> {
  const file = path.join(adjustmentRoot(), "source-model-set.json")
  return AdjustmentSourceModelSet.parse(await readJsonFile(file))
}

type FitDiagnosticsEvidence = {
  horizonIndices: Set<number>
  timestampHours: Set<number>
  timestampMonths: Set<number>
  timestampMinutesOfDay: Set<number>
  rawPredictedRange: { min: number; max: number } | null
  exogenousRanges: Map<string, { min: number; max: number }>
  exogenousTextValues: Map<string, Set<string>>
}

function normalizedProjectPath(value: string): string {
  return value.replaceAll("\\", "/").replace(/^\.?\//, "")
}

function finiteRange(value: unknown): { min: number; max: number } | null {
  if (!value || typeof value !== "object") return null
  const record = value as Record<string, unknown>
  const min = Number(record.min)
  const max = Number(record.max)
  if (!Number.isFinite(min) || !Number.isFinite(max) || min > max) return null
  return { min, max }
}

function numberSet(value: unknown): Set<number> {
  if (!Array.isArray(value)) return new Set()
  return new Set(value.map(Number).filter(Number.isFinite))
}

function stringSet(value: unknown): Set<string> {
  if (!Array.isArray(value)) return new Set()
  return new Set(value.map((item) => String(item).trim()).filter(Boolean))
}

function dayOffsetTag(dayOffset: number): string {
  if (dayOffset === 0) return "current"
  return dayOffset > 0 ? `p${dayOffset}` : `m${Math.abs(dayOffset)}`
}

function exogenousEvidenceKey(column: string, dayOffset?: number): string {
  const offset = dayOffset ?? 0
  return offset === 0 ? column : `day_${dayOffsetTag(offset)}__${column}`
}

function parseMinuteOfDay(value: string): number {
  const match = /^(\d{2}):(\d{2})$/.exec(value)
  if (!match) throw new Error(`Invalid timestamp_time_between value ${value}; expected HH:mm`)
  const hour = Number(match[1])
  const minute = Number(match[2])
  if (!Number.isInteger(hour) || !Number.isInteger(minute) || hour < 0 || hour > 23 || minute < 0 || minute > 59) {
    throw new Error(`Invalid timestamp_time_between value ${value}; expected HH:mm`)
  }
  return hour * 60 + minute
}

function minuteRangeOverlapsObserved(start: number, end: number, observed: Set<number>): boolean {
  for (const minute of observed) {
    if (start <= end) {
      if (minute >= start && minute <= end) return true
    } else if (minute >= start || minute <= end) {
      return true
    }
  }
  return false
}

function normalizedBooleanText(value: boolean): Set<string> {
  return value ? new Set(["true", "1", "yes", "y", "是"]) : new Set(["false", "0", "no", "n", "否", ""])
}

async function readFitDiagnosticsEvidence(
  evidenceRefs: string[],
  sourceModelSetId: string,
): Promise<FitDiagnosticsEvidence[]> {
  const fitEvidenceRefs = evidenceRefs.flatMap((rawRef) => {
    const ref = normalizedProjectPath(rawRef)
    const match = /^\.forecast\/adjustments\/diagnostics\/fit\/([A-Za-z0-9_-]+)\/diagnostics\.json$/.exec(ref)
    return match ? [{ rawRef, ref, fitRunId: match[1] }] : []
  })
  if (fitEvidenceRefs.length === 0) return []

  const fitManifest = AdjustmentFitRunManifest.parse(
    await readJsonFile(path.join(adjustmentRoot(), "fit-runs", "manifest.json")),
  )
  if (fitManifest.source_model_set_id !== sourceModelSetId) {
    throw new Error(
      `Adjustment fit diagnostics source set ${fitManifest.source_model_set_id} does not match frozen source set ${sourceModelSetId}`,
    )
  }
  const completedRunIds = new Set(
    fitManifest.members
      .filter((member) => member.status === "success" || member.status === "cached")
      .map((member) => member.fit_run_id),
  )
  const evidenceItems: FitDiagnosticsEvidence[] = []

  for (const { rawRef, ref, fitRunId } of fitEvidenceRefs) {
    if (!completedRunIds.has(fitRunId)) {
      throw new Error(`Adjustment rule evidence references incomplete fit diagnostics: ${rawRef}`)
    }
    const absolute = projectPath(ref)
    if (!(await Filesystem.exists(absolute))) {
      throw new Error(`Adjustment rule evidence diagnostics are missing: ${rawRef}`)
    }
    const diagnostics = z.record(z.string(), z.unknown()).parse(await readJsonFile(absolute))
    if (diagnostics.status !== "success" || diagnostics.split_scope !== "fit_only_validation_front") {
      throw new Error(`Adjustment rule evidence is not validation-front fit diagnostics: ${rawRef}`)
    }
    const domains = z.record(z.string(), z.unknown()).parse(diagnostics.observed_condition_domains)
    const exogenousRanges = new Map<string, { min: number; max: number }>()
    const exogenous = z.record(z.string(), z.unknown()).parse(domains.exogenous_numeric ?? {})
    for (const [column, value] of Object.entries(exogenous)) {
      const range = finiteRange(value)
      if (!range) continue
      exogenousRanges.set(column, range)
    }
    const exogenousTextValues = new Map<string, Set<string>>()
    const textExogenous = z.record(z.string(), z.unknown()).parse(domains.exogenous_text ?? {})
    for (const [column, value] of Object.entries(textExogenous)) {
      if (!value || typeof value !== "object") continue
      const values = stringSet((value as Record<string, unknown>).values)
      if (values.size > 0) exogenousTextValues.set(column, values)
    }
    evidenceItems.push({
      horizonIndices: numberSet(domains.horizon_indices),
      timestampHours: numberSet(domains.timestamp_hours),
      timestampMonths: numberSet(domains.timestamp_months),
      timestampMinutesOfDay: numberSet(domains.timestamp_minutes_of_day),
      rawPredictedRange: finiteRange(domains.raw_predicted_original),
      exogenousRanges,
      exogenousTextValues,
    })
  }
  return evidenceItems
}

function assertValuesObserved(values: number[], observed: Set<number>, label: string): void {
  const unsupported = values.filter((value) => !observed.has(value))
  if (unsupported.length > 0) {
    throw new Error(`${label} contains values not observed in the validation-front fit evidence: ${unsupported.join(", ")}`)
  }
}

function validateConditionEvidence(
  condition: AdjustmentRuleCondition,
  evidence: FitDiagnosticsEvidence,
  location = "condition",
): void {
  if (condition.kind === "horizon_index_in") {
    assertValuesObserved(condition.values, evidence.horizonIndices, `${location}.horizon_index_in`)
  } else if (condition.kind === "timestamp_hour_in") {
    assertValuesObserved(condition.values, evidence.timestampHours, `${location}.timestamp_hour_in`)
  } else if (condition.kind === "timestamp_month_in") {
    assertValuesObserved(condition.values, evidence.timestampMonths, `${location}.timestamp_month_in`)
  } else if (condition.kind === "timestamp_time_between") {
    const start = parseMinuteOfDay(condition.start)
    const end = parseMinuteOfDay(condition.end)
    if (!minuteRangeOverlapsObserved(start, end, evidence.timestampMinutesOfDay)) {
      throw new Error(`${location}.timestamp_time_between does not overlap validation-front observed timestamps`)
    }
  } else if (condition.kind === "raw_predicted_original_between") {
    const range = evidence.rawPredictedRange
    if (!range || condition.min < range.min || condition.max > range.max) {
      throw new Error(`${location}.raw_predicted_original_between exceeds the validation-front observed range`)
    }
  } else if (condition.kind === "exogenous_numeric_between") {
    const range = evidence.exogenousRanges.get(exogenousEvidenceKey(condition.column, condition.day_offset))
    if (!range || condition.min < range.min || condition.max > range.max) {
      throw new Error(
        `${location}.exogenous_numeric_between for ${condition.column} exceeds the validation-front observed range`,
      )
    }
  } else if (condition.kind === "exogenous_text_in") {
    const observed = evidence.exogenousTextValues.get(exogenousEvidenceKey(condition.column, condition.day_offset))
    const unsupported = condition.values.map((value) => String(value).trim()).filter((value) => !observed?.has(value))
    if (unsupported.length > 0) {
      throw new Error(
        `${location}.exogenous_text_in for ${condition.column} contains values not observed in validation-front evidence: ${unsupported.join(", ")}`,
      )
    }
  } else if (condition.kind === "exogenous_text_contains") {
    const observed = evidence.exogenousTextValues.get(exogenousEvidenceKey(condition.column, condition.day_offset))
    const observedValues = [...(observed ?? new Set<string>())]
    const unsupported = condition.values
      .map((value) => String(value).trim())
      .filter((value) => !observedValues.some((observedValue) => observedValue.includes(value)))
    if (unsupported.length > 0) {
      throw new Error(
        `${location}.exogenous_text_contains for ${condition.column} contains substrings not observed in validation-front evidence: ${unsupported.join(", ")}`,
      )
    }
  } else if (condition.kind === "exogenous_bool_is") {
    const key = exogenousEvidenceKey(condition.column, condition.day_offset)
    const textObserved = evidence.exogenousTextValues.get(key)
    const numericObserved = evidence.exogenousRanges.get(key)
    const boolTexts = normalizedBooleanText(condition.value)
    const textSupported = [...(textObserved ?? new Set<string>())].some((value) =>
      boolTexts.has(String(value).trim().toLocaleLowerCase("en-US")) || boolTexts.has(String(value).trim()),
    )
    const numericSupported =
      numericObserved &&
      (condition.value
        ? numericObserved.max > 0
        : numericObserved.min <= 0)
    if (!textSupported && !numericSupported) {
      throw new Error(`${location}.exogenous_bool_is for ${condition.column} is not supported by validation-front evidence`)
    }
  } else if (condition.kind === "all" || condition.kind === "any") {
    condition.conditions.forEach((child, index) => validateConditionEvidence(child, evidence, `${location}.${index}`))
  }
}

function validateUserPriorConditionCapability(
  condition: AdjustmentRuleCondition,
  evidence: FitDiagnosticsEvidence,
  location = "condition",
): void {
  if (condition.kind === "exogenous_numeric_between") {
    const key = exogenousEvidenceKey(condition.column, condition.day_offset)
    if (!evidence.exogenousRanges.has(key)) {
      throw new Error(`${location}.exogenous_numeric_between cannot use unavailable forecast-time column ${condition.column}`)
    }
  } else if (
    condition.kind === "exogenous_text_in" ||
    condition.kind === "exogenous_text_contains" ||
    condition.kind === "exogenous_bool_is"
  ) {
    const key = exogenousEvidenceKey(condition.column, condition.day_offset)
    if (!evidence.exogenousTextValues.has(key) && !evidence.exogenousRanges.has(key)) {
      throw new Error(`${location}.${condition.kind} cannot use unavailable forecast-time column ${condition.column}`)
    }
  } else if (condition.kind === "all" || condition.kind === "any") {
    condition.conditions.forEach((child, index) =>
      validateUserPriorConditionCapability(child, evidence, `${location}.${index}`),
    )
  }
}

async function readVerifiedUserPriorEvidence(
  contract: z.infer<typeof AdjustmentUserPriorContract>,
  task: Record<string, unknown>,
): Promise<z.infer<typeof AdjustmentUserPriorEvidence>> {
  const evidencePath = projectPath(normalizedProjectPath(contract.evidence_path))
  const evidence = AdjustmentUserPriorEvidence.parse(await readJsonFile(evidencePath))
  const sourceText = String(task.adjustment_skill_priors ?? "").trim()
  if (!sourceText || evidence.source_text !== sourceText) {
    throw new Error("User-prior evidence does not match the frozen task adjustment_skill_priors")
  }
  const sourceSha256 = sha256(sourceText)
  if (contract.source_sha256 !== sourceSha256 || evidence.source_sha256 !== sourceSha256) {
    throw new Error("User-prior source hash does not match the frozen task prior")
  }
  const artifactSha256 = userPriorEvidenceHash({
    source_sha256: evidence.source_sha256,
    references: evidence.references,
  })
  if (
    artifactSha256 !== evidence.artifact_sha256 ||
    contract.evidence_path !== userPriorEvidenceRelativePath(artifactSha256)
  ) {
    throw new Error("User-prior evidence artifact identity is invalid")
  }
  for (const reference of evidence.references) {
    const capturedBytes = Buffer.from(reference.content, "utf-8")
    if (capturedBytes.byteLength !== reference.size_bytes || sha256(capturedBytes) !== reference.sha256) {
      throw new Error(`User-prior evidence captured content is invalid: ${reference.requested_path}`)
    }
    const bytes = await readFile(reference.resolved_path).catch(() => null)
    if (!bytes || bytes.byteLength !== reference.size_bytes || sha256(bytes) !== reference.sha256) {
      throw new Error(`User-prior read-only reference changed after inspection: ${reference.requested_path}`)
    }
  }
  return evidence
}

function parameterKey(ruleId: string, field: string): string {
  return `${ruleId}\u0000${field}`
}

function readParameterField(rule: z.infer<typeof AdjustmentRule>, field: string): unknown {
  let current: unknown = rule
  for (const segment of field.split(".")) {
    if (Array.isArray(current)) {
      const index = Number(segment)
      if (!Number.isInteger(index) || index < 0 || index >= current.length) return undefined
      current = current[index]
      continue
    }
    if (!current || typeof current !== "object" || !(segment in current)) return undefined
    current = (current as Record<string, unknown>)[segment]
  }
  return current
}

function parameterValuesEqual(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right)
}

function requiredConditionContractFields(
  condition: AdjustmentRuleCondition,
  prefix = "condition",
): string[] {
  const fields = [`${prefix}.kind`]
  if (
    condition.kind === "horizon_index_in" ||
    condition.kind === "timestamp_hour_in" ||
    condition.kind === "timestamp_month_in"
  ) {
    fields.push(`${prefix}.values`)
  } else if (condition.kind === "timestamp_time_between") {
    fields.push(`${prefix}.start`, `${prefix}.end`)
  } else if (condition.kind === "raw_predicted_original_between") {
    fields.push(`${prefix}.min`, `${prefix}.max`)
  } else if (condition.kind === "exogenous_numeric_between") {
    fields.push(`${prefix}.column`, `${prefix}.min`, `${prefix}.max`)
    if (condition.day_offset !== undefined) fields.push(`${prefix}.day_offset`)
  } else if (condition.kind === "exogenous_text_in" || condition.kind === "exogenous_text_contains") {
    fields.push(`${prefix}.column`, `${prefix}.values`)
    if (condition.day_offset !== undefined) fields.push(`${prefix}.day_offset`)
  } else if (condition.kind === "exogenous_bool_is") {
    fields.push(`${prefix}.column`, `${prefix}.value`)
    if (condition.day_offset !== undefined) fields.push(`${prefix}.day_offset`)
  } else if (condition.kind === "all" || condition.kind === "any") {
    condition.conditions.forEach((child, index) => {
      fields.push(...requiredConditionContractFields(child, `${prefix}.conditions.${index}`))
    })
  }
  return fields
}

function requiredUserPriorContractFields(rule: z.infer<typeof AdjustmentRule>): string[] {
  const fields = [...requiredConditionContractFields(rule.condition), "action.type"]
  if (rule.action.type === "add" || rule.action.type === "multiply") fields.push("action.value")
  if (rule.action.type === "clip") {
    if (rule.action.min !== undefined) fields.push("action.min")
    if (rule.action.max !== undefined) fields.push("action.max")
  }
  if (rule.action.type === "add_profile") fields.push("action.by", "action.values")
  if (rule.bounds.max_delta_ratio !== undefined) fields.push("bounds.max_delta_ratio")
  if (rule.bounds.max_delta_original !== undefined) fields.push("bounds.max_delta_original")
  if (rule.bounds.min_delta_original !== undefined) fields.push("bounds.min_delta_original")
  return fields
}

function isNumericParameterValue(value: unknown): boolean {
  return typeof value === "number" || (Array.isArray(value) && value.every((item) => typeof item === "number"))
}

function numericValues(value: unknown): number[] {
  if (typeof value === "number") return [value]
  if (Array.isArray(value) && value.every((item) => typeof item === "number")) return value
  return []
}

function textContainsNumericValues(text: string, value: unknown): boolean {
  const expected = numericValues(value)
  if (expected.length === 0) return false
  const observed = [...text.matchAll(/[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?/gi)]
    .map((match) => Number(match[0]))
    .filter(Number.isFinite)
  return expected.every((item) => observed.some((candidate) => Object.is(candidate, item) || candidate === item))
}

function textContainsSemanticValue(text: string, value: unknown): boolean {
  const normalized = text.toLocaleLowerCase("en-US")
  if (typeof value === "string") return normalized.includes(value.trim().toLocaleLowerCase("en-US"))
  if (Array.isArray(value) && value.every((item) => typeof item === "string")) {
    return value.every((item) => normalized.includes(item.trim().toLocaleLowerCase("en-US")))
  }
  if (typeof value === "boolean") {
    const tokens = value ? ["true", "yes", "是", "为真"] : ["false", "no", "否", "为假"]
    return tokens.some((token) => normalized.includes(token))
  }
  return false
}

function requiresLiteralSemanticEvidence(field: string, value: unknown): boolean {
  return (
    field.endsWith(".column") ||
    field.endsWith(".start") ||
    field.endsWith(".end") ||
    (field.endsWith(".value") && typeof value === "boolean") ||
    (field.endsWith(".values") && Array.isArray(value) && value.every((item) => typeof item === "string"))
  )
}

function sourceDirection(sourceExcerpt: string): "up" | "down" | "unknown" {
  const withoutNegatedDirections = sourceExcerpt
    .replace(/(?:\u4e0d\u5141\u8bb8|\u7981\u6b62|\u4e0d\u5f97|\u4e0d\u80fd|never|do\s+not|don't|must\s+not)\s*(?:\u4e0b\u8c03|\u964d\u4f4e|\u51cf\u5c11|decrease|reduce|lower)/gi, "")
    .replace(/(?:\u4e0d\u5141\u8bb8|\u7981\u6b62|\u4e0d\u5f97|\u4e0d\u80fd|never|do\s+not|don't|must\s+not)\s*(?:\u4e0a\u8c03|\u63d0\u9ad8|\u589e\u52a0|increase|raise|uplift)/gi, "")
  const up = /(?:\u4e0a\u8c03|\u8c03\u9ad8|\u63d0\u9ad8|\u589e\u52a0|\u62ac\u5347|increase|raise|uplift|upward)/i.test(
    withoutNegatedDirections,
  )
  const down = /(?:\u4e0b\u8c03|\u8c03\u4f4e|\u964d\u4f4e|\u51cf\u5c11|decrease|reduce|lower|downward)/i.test(
    withoutNegatedDirections,
  )
  if (up === down) return "unknown"
  return up ? "up" : "down"
}

function assertRuleDirectionMatchesSource(rule: z.infer<typeof AdjustmentRule>, sourceExcerpt: string): void {
  const direction = sourceDirection(sourceExcerpt)
  if (direction === "unknown") return
  const action = rule.action
  const movesUp =
    (action.type === "add" && action.value > 0) ||
    (action.type === "multiply" && action.value > 1) ||
    (action.type === "add_profile" && action.values.every((value) => value >= 0) && action.values.some((value) => value > 0))
  const movesDown =
    (action.type === "add" && action.value < 0) ||
    (action.type === "multiply" && action.value < 1) ||
    (action.type === "add_profile" && action.values.every((value) => value <= 0) && action.values.some((value) => value < 0))
  if ((direction === "up" && !movesUp) || (direction === "down" && !movesDown)) {
    throw new Error(`Rule ${rule.id} action direction rewrites the mapped user-prior source excerpt`)
  }
  if (direction === "up" && rule.bounds.min_delta_original !== undefined && rule.bounds.min_delta_original < 0) {
    throw new Error(`Rule ${rule.id} permits downward movement despite an up-only user prior`)
  }
  if (direction === "down" && rule.bounds.min_delta_original !== undefined && rule.bounds.min_delta_original > 0) {
    throw new Error(`Rule ${rule.id} permits upward-only movement despite a down user prior`)
  }
}

async function validateUserPriorContract(input: {
  contract: z.infer<typeof AdjustmentUserPriorContract>
  rules: Array<z.infer<typeof AdjustmentRule>>
  task: Record<string, unknown>
}): Promise<void> {
  const evidence = await readVerifiedUserPriorEvidence(input.contract, input.task)
  const rulesById = new Map(input.rules.map((rule) => [rule.id, rule]))
  if (rulesById.size !== input.rules.length) throw new Error("User-prior mirror rule ids must be unique")

  const mappings = new Map<string, string>()
  for (const mapping of input.contract.rule_mappings) {
    if (mappings.has(mapping.rule_id)) throw new Error(`Duplicate user-prior rule mapping for ${mapping.rule_id}`)
    mappings.set(mapping.rule_id, mapping.source_excerpt)
  }

  const completions = new Map<string, z.infer<typeof AdjustmentUserPriorContract>["parameter_completions"][number]>()
  for (const completion of input.contract.parameter_completions) {
    const key = parameterKey(completion.rule_id, completion.field)
    if (completions.has(key)) throw new Error(`Duplicate user-prior parameter completion ${completion.rule_id}/${completion.field}`)
    const rule = rulesById.get(completion.rule_id)
    if (!rule) throw new Error(`Parameter completion references unknown rule ${completion.rule_id}`)
    const actualValue = readParameterField(rule, completion.field)
    if (!parameterValuesEqual(actualValue, completion.value)) {
      throw new Error(`Parameter completion ${completion.rule_id}/${completion.field} does not match the executable rule`)
    }
    if (
      completion.source === "explicit_prompt" &&
      requiresLiteralSemanticEvidence(completion.field, completion.value) &&
      !textContainsSemanticValue(evidence.source_text, completion.value)
    ) {
      throw new Error(
        `Explicit-prompt semantic parameter ${completion.rule_id}/${completion.field} is not present in the frozen user Prompt`,
      )
    }
    if (completion.source === "readonly_reference") {
      const reference = evidence.references.find(
        (item) => item.requested_path === completion.evidence_ref || item.resolved_path === completion.evidence_ref,
      )
      if (
        reference &&
        isNumericParameterValue(completion.value) &&
        !textContainsNumericValues(reference.content, completion.value)
      ) {
        throw new Error(
          `Readonly parameter completion ${completion.rule_id}/${completion.field} is not present in the verified reference`,
        )
      }
      if (
        reference &&
        requiresLiteralSemanticEvidence(completion.field, completion.value) &&
        !textContainsSemanticValue(reference.content, completion.value)
      ) {
        throw new Error(
          `Readonly semantic parameter ${completion.rule_id}/${completion.field} is not present in the verified reference`,
        )
      }
    }
    if (completion.source === "validation_front_fit") {
      if (!isNumericParameterValue(completion.value)) {
        throw new Error(
          `Validation-front matching may fill only missing numeric values; ${completion.rule_id}/${completion.field} is non-numeric`,
        )
      }
      if (!rule.evidence_refs.includes(completion.evidence_ref ?? "")) {
        throw new Error(`Validation-front completion ${completion.rule_id}/${completion.field} lacks canonical fit evidence`)
      }
    }
    if (completion.source === "llm_inferred" && !isNumericParameterValue(completion.value)) {
      throw new Error(
        `LLM may fill only missing numeric values; ${completion.rule_id}/${completion.field} is non-numeric`,
      )
    }
    completions.set(key, completion)
  }

  const unresolved = new Set<string>()
  for (const locator of input.contract.unresolved_parameters_before_llm) {
    const key = parameterKey(locator.rule_id, locator.field)
    if (unresolved.has(key)) throw new Error(`Duplicate unresolved user-prior parameter ${locator.rule_id}/${locator.field}`)
    unresolved.add(key)
    if (completions.get(key)?.source !== "llm_inferred") {
      throw new Error(`Unresolved parameter ${locator.rule_id}/${locator.field} must be completed by llm_inferred`)
    }
  }
  for (const rule of input.rules) {
    const requiredFields = requiredUserPriorContractFields(rule)
    if (input.contract.validation_front_match === "none") {
      for (const field of requiredFields) {
        const source = completions.get(parameterKey(rule.id, field))?.source
        if (source === "validation_front_fit") {
          throw new Error(`No-match user prior cannot claim validation-front fitting for ${rule.id}/${field}`)
        }
      }
    }
    const sourceExcerpt = mappings.get(rule.id)
    if (sourceExcerpt) assertRuleDirectionMatchesSource(rule, sourceExcerpt)
  }
}

async function countCandidatesForSourceSet(
  sourceModelSetId: string,
  origin: z.infer<typeof AdjustmentCandidateOrigin>,
): Promise<number> {
  const ids = await listCandidates()
  let count = 0
  for (const id of ids) {
    const parsed = AdjustmentCandidate.safeParse(
      await readJsonFile(path.join(candidateDir(id), "candidate.json")).catch(() => null),
    )
    const effectiveOrigin =
      parsed.success && (parsed.data.origin === "user_prior_mirror" || parsed.data.llm_provenance.user_prior_mirror === true)
        ? "user_prior_mirror"
        : "llm_exploratory"
    if (
      parsed.success &&
      parsed.data.source_model_set_id === sourceModelSetId &&
      effectiveOrigin === origin
    ) {
      count += 1
    }
  }
  return count
}

async function readActiveManifest(): Promise<z.infer<typeof AdjustmentSkillManifest> | null> {
  const manifestPath = path.join(adjustmentRoot(), "skills", "active", "manifest.json")
  if (!(await Filesystem.exists(manifestPath))) return null
  return AdjustmentSkillManifest.parse(await readJsonFile(manifestPath))
}

async function assertCandidateWriteAllowed(parentSkillVersion: string | null): Promise<void> {
  const state = await readPhaseState()
  if (state.phase === "post-forecast") {
    throw new Error("forecast_adjustment_candidate write is not allowed after entering post-forecast")
  }
  if (!state.adjustment.source_model_set_locked || !state.adjustment.fit_runs_ready) {
    throw new Error("forecast_adjustment_candidate write requires a frozen source set and completed fit runs")
  }
  if (state.adjustment.forecaster_locked && (state.adjustment.active_skill_version || state.adjustment.no_effective_skill)) {
    throw new Error("forecast_adjustment_candidate write is not allowed after the adjustment decision is locked")
  }

  const activeManifest = await readActiveManifest()
  if (activeManifest && activeManifest.status === "active" && parentSkillVersion !== activeManifest.version) {
    throw new Error(
      `candidate parent_skill_version ${parentSkillVersion ?? "null"} must match active skill ${activeManifest.version}`,
    )
  }
}

async function assertUserPriorInspectionAllowed(): Promise<void> {
  const state = await readPhaseState()
  if (state.phase !== "forecast") {
    throw new Error("forecast_adjustment_candidate inspect_user_prior is only allowed during forecast")
  }
  if (!state.adjustment.enabled) {
    throw new Error("forecast_adjustment_candidate inspect_user_prior requires generative prediction reasoning to be enabled")
  }
  if (state.adjustment.forecaster_locked && (state.adjustment.active_skill_version || state.adjustment.no_effective_skill)) {
    throw new Error("forecast_adjustment_candidate inspect_user_prior is not allowed after the adjustment decision is locked")
  }
}

const FORBIDDEN_EXOGENOUS_COLUMNS = new Set(["actual", "target", "residual", "error", "metric", "mse", "mae", "wape", "mase"])

function validateRuleConditionSemantics(
  condition: AdjustmentRuleCondition,
  task: Record<string, unknown> | null,
  location = "condition",
): void {
  if (condition.kind === "raw_predicted_original_between" && condition.min > condition.max) {
    throw new Error(`${location} raw_predicted_original_between requires min <= max`)
  }
  if (condition.kind === "exogenous_numeric_between") {
    if (condition.min > condition.max) {
      throw new Error(`${location} exogenous_numeric_between requires min <= max`)
    }
    const lowerColumn = condition.column.toLocaleLowerCase("en-US")
    if (
      lowerColumn === String(task?.target_col ?? "").toLocaleLowerCase("en-US") ||
      lowerColumn === String(task?.time_col ?? "").toLocaleLowerCase("en-US") ||
      FORBIDDEN_EXOGENOUS_COLUMNS.has(lowerColumn)
    ) {
      throw new Error(`${location} exogenous_numeric_between cannot target time, target, actual, residual, or metric columns`)
    }
  }
  if (
    condition.kind === "exogenous_text_in" ||
    condition.kind === "exogenous_text_contains" ||
    condition.kind === "exogenous_bool_is"
  ) {
    const lowerColumn = condition.column.toLocaleLowerCase("en-US")
    if (
      lowerColumn === String(task?.target_col ?? "").toLocaleLowerCase("en-US") ||
      lowerColumn === String(task?.time_col ?? "").toLocaleLowerCase("en-US") ||
      FORBIDDEN_EXOGENOUS_COLUMNS.has(lowerColumn)
    ) {
      throw new Error(`${location} ${condition.kind} cannot target time, target, actual, residual, or metric columns`)
    }
  }
  if (condition.kind === "all" || condition.kind === "any") {
    condition.conditions.forEach((child, index) => validateRuleConditionSemantics(child, task, `${location}.${index}`))
  }
}

function validateRuleActionSemantics(rule: AdjustmentRule): void {
  if (rule.action.type === "clip") {
    if (rule.action.min === undefined && rule.action.max === undefined) {
      throw new Error(`Rule ${rule.id} clip action requires min or max`)
    }
    if (
      rule.action.min !== undefined &&
      rule.action.max !== undefined &&
      rule.action.min > rule.action.max
    ) {
      throw new Error(`Rule ${rule.id} clip action requires min <= max`)
    }
  }
  if (
    rule.bounds.min_delta_original !== undefined &&
    rule.bounds.max_delta_original !== undefined &&
    rule.bounds.min_delta_original > rule.bounds.max_delta_original
  ) {
    throw new Error(`Rule ${rule.id} bounds require min_delta_original <= max_delta_original`)
  }
}

async function readTaskForCandidateValidation(): Promise<Record<string, unknown> | null> {
  try {
    return z.record(z.string(), z.unknown()).parse(await readJsonFile(path.join(forecastRoot(), "task.json")))
  } catch {
    return null
  }
}

export const ForecastAdjustmentCandidateTool = Tool.define("forecast_adjustment_candidate", {
  description: DESCRIPTION,
  parameters,
  async execute(params) {
    await ensureAdjustmentDirs()
    let result: unknown

    switch (params.subcommand) {
      case "list": {
        result = { candidates: await listCandidates() }
        break
      }

      case "inspect_user_prior": {
        await assertUserPriorInspectionAllowed()
        const evidence = await inspectUserPrior(params.reference_paths ?? [])
        result = {
          inspected: true,
          evidence_path: userPriorEvidenceRelativePath(evidence.artifact_sha256),
          evidence,
          execution_performed: false,
        }
        break
      }

      case "read": {
        if (!params.candidate_id) throw new Error("forecast_adjustment_candidate read requires candidate_id")
        const dir = candidateDir(params.candidate_id)
        result = {
          candidate: await readJsonFile(path.join(dir, "candidate.json")),
          skill_markdown: await readFile(path.join(dir, "SKILL.md"), "utf-8").catch(() => null),
        }
        break
      }

      case "write": {
        if (!params.source_model_set_id) {
          throw new Error("forecast_adjustment_candidate write requires source_model_set_id")
        }
        if (!params.rules) {
          throw new Error("forecast_adjustment_candidate write requires rules")
        }
        if (!params.skill_markdown) {
          throw new Error("forecast_adjustment_candidate write requires skill_markdown")
        }
        const parentSkillVersion = params.parent_skill_version ?? null
        await assertCandidateWriteAllowed(parentSkillVersion)
        const origin =
          params.origin ??
          (params.llm_provenance?.user_prior_mirror === true ? "user_prior_mirror" : "llm_exploratory")
        if (origin === "user_prior_mirror" && !params.user_prior_contract) {
          throw new Error("user_prior_mirror candidate write requires user_prior_contract from inspect_user_prior")
        }
        if (origin === "llm_exploratory" && params.user_prior_contract) {
          throw new Error("llm_exploratory candidates cannot attach a user_prior_contract")
        }
        if (origin === "user_prior_mirror" && params.llm_provenance?.user_prior_mirror !== true) {
          throw new Error("user_prior_mirror candidate write requires llm_provenance.user_prior_mirror=true")
        }
        const policy = await readPolicy()
        const task = await readTaskForCandidateValidation()
        params.rules.forEach((rule) => {
          validateRuleConditionSemantics(rule.condition, task)
          validateRuleActionSemantics(rule)
        })
        if (origin === "user_prior_mirror") {
          if (!task) throw new Error("user_prior_mirror candidate write requires a readable frozen task.json")
          await validateUserPriorContract({ contract: params.user_prior_contract!, rules: params.rules, task })
        }
        const sourceSet = await readSourceSet()
        if (sourceSet.source_model_set_id !== params.source_model_set_id) {
          throw new Error(
            `candidate source_model_set_id ${params.source_model_set_id} does not match frozen source set ${sourceSet.source_model_set_id}`,
          )
        }
        for (const rule of params.rules) {
          const ruleMaxDelta = rule.bounds.max_delta_ratio
          if (ruleMaxDelta !== undefined && ruleMaxDelta > policy.max_point_delta_ratio) {
            throw new Error(
              `Rule ${rule.id} max_delta_ratio=${ruleMaxDelta} exceeds policy.max_point_delta_ratio=${policy.max_point_delta_ratio}`,
            )
          }
          const evidenceItems = await readFitDiagnosticsEvidence(rule.evidence_refs, sourceSet.source_model_set_id)
          evidenceItems.forEach((evidence, index) => {
            if (origin === "user_prior_mirror") {
              validateUserPriorConditionCapability(rule.condition, evidence, `rule ${rule.id} evidence_refs[${index}]`)
            } else {
              validateConditionEvidence(rule.condition, evidence, `rule ${rule.id} evidence_refs[${index}]`)
            }
          })
        }
        const existingCandidateCount = await countCandidatesForSourceSet(sourceSet.source_model_set_id, origin)
        const candidateLimit = origin === "user_prior_mirror" ? 1 : policy.max_candidate_iterations
        if (existingCandidateCount >= candidateLimit) {
          throw new Error(
            `forecast_adjustment_candidate write would exceed the independent ${origin} budget ` +
              `of ${candidateLimit} candidate(s) for the frozen source set`,
          )
        }
        const base = {
          source_model_set_id: params.source_model_set_id,
          parent_skill_version: parentSkillVersion,
          origin,
          user_prior_contract: params.user_prior_contract ?? null,
          rules: params.rules,
          skill_markdown: params.skill_markdown,
          llm_provenance: params.llm_provenance ?? {},
        }
        const candidateId = params.candidate_id ?? `candidate-${computeSpecHash(base).slice(0, 16)}`
        if (await Filesystem.exists(candidateDir(candidateId))) {
          throw new Error(`forecast_adjustment_candidate write would overwrite existing candidate_id ${candidateId}`)
        }
        const candidate = AdjustmentCandidate.parse({
          candidate_id: candidateId,
          status: "candidate",
          created_at: new Date().toISOString(),
          ...base,
        })
        const dir = candidateDir(candidate.candidate_id)
        await mkdir(dir, { recursive: true })
        await writeJsonFile(path.join(dir, "candidate.json"), candidate)
        await writeFile(path.join(dir, "SKILL.md"), candidate.skill_markdown.trim() + "\n", "utf-8")
        await writeJsonFile(path.join(dir, "llm-provenance.json"), candidate.llm_provenance)
        result = {
          written: true,
          candidate_id: candidate.candidate_id,
          candidate_path: path.join(dir, "candidate.json"),
        }
        break
      }
    }

    return {
      title: `Forecast adjustment candidate: ${params.subcommand}`,
      metadata: { subcommand: params.subcommand, candidate_id: params.candidate_id },
      output: JSON.stringify(result),
    }
  },
})
