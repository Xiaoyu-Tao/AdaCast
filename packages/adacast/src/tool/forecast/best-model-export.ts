import { copyFile, mkdir, readdir, readFile, rename, rm } from "fs/promises"
import { randomUUID } from "node:crypto"
import path from "path"
import matter from "gray-matter"
import z from "zod"
import { Instance } from "@/project/instance"
import { Filesystem } from "@/util/filesystem"
import { Tool } from "../tool"
import DESCRIPTION from "./best-model-export.txt"
import {
  normalizeForecastExecutionContract,
  normalizeValidationExecutionContract,
  validationSpecMatchesPostForecastTestSpec,
} from "./execution-contract"
import { normalizeForecastStateFrontmatter } from "./forecast-progress"
import { validateWindowReflectionRecordArtifact } from "./forecast-window-reflection-artifact"
import { computeSpecHash } from "./hash"
import { computeFileSha256, readBoundModelArtifactIdentity } from "./model-artifact"
import { foundationModelReference, statisticalModelArtifact } from "./model-family"
import {
  AdjustmentEvalResult,
  AdjustmentSkillManifest,
  BestModelArtifact,
  BestModelManifest,
  EvalResult,
  ForecastPhaseState,
  TaskConfig,
} from "./types"

type BestModelArtifactValue = z.infer<typeof BestModelArtifact>
type BestModelManifestValue = z.infer<typeof BestModelManifest>

type BestModelExportMetadata = {
  status: "success" | "blocked"
  export_path: string
  manifest_path: string
  missing_artifacts: string[]
  warnings: string[]
}

const parameters = z.object({
  validation_run_id: z.string().describe("Validation run id selected during the model-selection phase"),
  test_run_id: z.string().describe("Held-out test run id selected as the global test-best result"),
  model_family: z.string().optional().describe("Selected model family, when known"),
  selection_reason: z.string().optional().describe("Critic rationale for choosing this test-best run"),
})

const REQUIRED_TEST_ARTIFACTS = [
  { source: "spec.json", target: "metadata/spec.json" },
  { source: "eval.json", target: "metadata/eval.json" },
  { source: "train_scale.json", target: "metadata/train_scale.json" },
  { source: "predictions.csv", target: "predictions/predictions.csv" },
  { source: "actual.npy", target: "predictions/actual.npy" },
  { source: "pred.npy", target: "predictions/pred.npy" },
  { source: "train.log", target: "logs/train.log" },
  { source: "model/reuse.json", target: "metadata/model-reuse.json" },
]

function runIdIsSafe(runId: string): boolean {
  return /^[A-Za-z0-9_-]+$/.test(runId)
}

function forecastPath(...parts: string[]): string {
  return path.posix.join(".forecast", ...parts)
}

async function readJsonOrMissing<T>(file: string, missing: string[]): Promise<T | null> {
  try {
    return await Filesystem.readJson<T>(file)
  } catch {
    missing.push(file)
    return null
  }
}

async function findMissingFiles(files: string[]): Promise<string[]> {
  const results = await Promise.all(files.map(async (file) => ((await Filesystem.exists(file)) ? null : file)))
  return results.filter((file): file is string => file !== null)
}

async function copyPackageFile(source: string, destination: string): Promise<void> {
  await mkdir(path.dirname(destination), { recursive: true })
  await copyFile(source, destination)
}

async function findAdjustedTestArtifact(
  forecastRoot: string,
  testRunId: string,
): Promise<{
  candidateId: string
  root: string
  evalPath: string
  predictionsPath: string
  auditPath: string
  auditPassed: boolean | null
  skillManifestPath: string | null
  skillPath: string | null
  evalJson: Record<string, unknown>
} | null> {
  const testRoot = path.join(forecastRoot, "adjustments", "test", testRunId)
  const rawState = await readFile(path.join(forecastRoot, "STATE.md"), "utf-8").catch(() => null)
  if (!rawState) return null
  const state = ForecastPhaseState.safeParse(
    normalizeForecastStateFrontmatter(matter(rawState).data as Record<string, unknown>),
  )
  if (!state.success) {
    return null
  }
  const applicationRecord = state.data.adjustment.test_application_records.find(
    (record) => record.test_run_id === testRunId,
  )
  const candidateId =
    applicationRecord?.adjustment_locked && applicationRecord.adjustment_evaluated
      ? applicationRecord.adjustment_candidate_id
      : state.data.adjustment.test_application_records.length === 0 &&
          state.data.adjustment.test_adjustment_locked &&
          state.data.adjustment.test_adjustment_run_id === testRunId
        ? state.data.adjustment.test_adjustment_candidate_id
        : null
  if (!candidateId) return null

  const skillManifestPath = path.join(forecastRoot, "adjustments", "skills", "active", "manifest.json")
  const activeManifest = AdjustmentSkillManifest.safeParse(
    await Filesystem.readJson(skillManifestPath).catch(() => null),
  )
  if (!activeManifest.success || activeManifest.data.status !== "locked") return null
  if (activeManifest.data.version !== state.data.adjustment.active_skill_version) return null
  if (activeManifest.data.candidate_id !== candidateId) return null

  const skillPath = path.join(forecastRoot, "adjustments", "skills", "active", "SKILL.md")
  const entries = await readdir(testRoot, { withFileTypes: true }).catch(() => [])
  for (const entry of entries.toSorted((a, b) => a.name.localeCompare(b.name))) {
    if (!entry.isDirectory()) continue
    if (entry.name !== activeManifest.data.candidate_id) continue
    const root = path.join(testRoot, entry.name)
    const evalPath = path.join(root, "adjusted_eval.json")
    const predictionsPath = path.join(root, "adjusted_predictions.csv")
    const auditPath = path.join(root, "leakage-audit.jsonl")
    if (!(await Filesystem.exists(evalPath)) || !(await Filesystem.exists(predictionsPath))) continue
    const evalJson = await Filesystem.readJson<Record<string, unknown>>(evalPath).catch(() => null)
    const parsedEval = AdjustmentEvalResult.safeParse(evalJson)
    if (
      !parsedEval.success ||
      parsedEval.data.run_id !== testRunId ||
      parsedEval.data.split !== "test" ||
      parsedEval.data.candidate_id !== activeManifest.data.candidate_id
    ) {
      continue
    }
    return {
      candidateId: entry.name,
      root,
      evalPath,
      predictionsPath,
      auditPath,
      auditPassed: await leakageAuditPassed(auditPath),
      skillManifestPath,
      skillPath: (await Filesystem.exists(skillPath)) ? skillPath : null,
      evalJson: parsedEval.data,
    }
  }
  return null
}

async function findWindowReflectionArtifact(
  forecastRoot: string,
  testRunId: string,
  validationRunId: string,
): Promise<{
  predictionsPath: string
  evalPath: string
  auditPath: string
  auditPassed: boolean
  protocolPath: string | null
  record: z.infer<typeof ForecastPhaseState>["reflection"]["test_records"][number]
  evalJson: Record<string, unknown>
} | null> {
  const rawState = await readFile(path.join(forecastRoot, "STATE.md"), "utf-8").catch(() => null)
  if (!rawState) return null
  const state = ForecastPhaseState.safeParse(
    normalizeForecastStateFrontmatter(matter(rawState).data as Record<string, unknown>),
  )
  if (!state.success) return null
  const record = state.data.reflection.test_records.find(
    (item) => item.test_run_id === testRunId && item.validation_run_id === validationRunId,
  )
  if (!record?.result_path || !record.final_metrics || !record.baseline_metrics) return null
  if (record.baseline_kind === "raw_anchor" && record.adjustment_candidate_id !== null) return null
  if (record.baseline_kind === "skill_adjusted") {
    const matchingAdjustment = state.data.adjustment.test_application_records.some(
      (item) =>
        item.test_run_id === testRunId &&
        item.validation_run_id === validationRunId &&
        item.decision === "apply" &&
        item.adjustment_locked &&
        item.adjustment_evaluated &&
        item.adjustment_candidate_id === record.adjustment_candidate_id,
    )
    if (!record.adjustment_candidate_id || !matchingAdjustment) return null
  }
  const validated = await validateWindowReflectionRecordArtifact(forecastRoot, record, "test")
  if (!validated) return null
  return {
    predictionsPath: validated.predictionsPath,
    evalPath: validated.evalPath,
    auditPath: validated.auditPath,
    auditPassed: validated.auditPassed,
    protocolPath: validated.protocolPath,
    record,
    evalJson: validated.artifact,
  }
}

async function leakageAuditPassed(auditPath: string): Promise<boolean | null> {
  if (!(await Filesystem.exists(auditPath))) return null
  try {
    const raw = await Bun.file(auditPath).text()
    const events = raw
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line) as { passed?: unknown })
    if (events.length === 0) return null
    return events.every((event) => event.passed === true)
  } catch {
    return false
  }
}

async function testApplicationDecisionAudit(
  forecastRoot: string,
  testRunId: string,
): Promise<{
  confirmedBeforeMetrics: boolean
  actor: "user" | "llm_auto" | "auto_policy" | null
  reason: string | null
} | null> {
  try {
    const raw = await readFile(path.join(forecastRoot, "STATE.md"), "utf-8")
    const state = ForecastPhaseState.safeParse(
      normalizeForecastStateFrontmatter(matter(raw).data as Record<string, unknown>),
    )
    if (!state.success) return null
    const applicationRecord = state.data.adjustment.test_application_records.find(
      (record) => record.test_run_id === testRunId,
    )
    if (applicationRecord) {
      if (!applicationRecord.adjustment_locked || !applicationRecord.adjustment_evaluated) return null
      return {
        confirmedBeforeMetrics: Boolean(applicationRecord.decision_at && applicationRecord.decision === "apply"),
        actor: applicationRecord.decision_actor,
        reason: applicationRecord.decision_reason,
      }
    }
    if (!state.data.adjustment.test_adjustment_locked || state.data.adjustment.test_adjustment_run_id !== testRunId) {
      return null
    }

    return {
      confirmedBeforeMetrics: Boolean(state.data.adjustment.critic_user_confirmed),
      actor: state.data.adjustment.test_application_decision_actor,
      reason: state.data.adjustment.test_application_decision_reason,
    }
  } catch {
    return null
  }
}

async function replaceDirectory(source: string, destination: string, warnings: string[]): Promise<void> {
  const backup = `${destination}.previous-${randomUUID()}`
  const hadPrevious = await Filesystem.exists(destination)

  if (hadPrevious) {
    await rename(destination, backup)
  }

  try {
    await rename(source, destination)
  } catch (error) {
    if (hadPrevious) {
      try {
        await rename(backup, destination)
      } catch (restoreError) {
        throw new Error(
          `Could not install package and could not restore previous package from ${backup}: ${
            restoreError instanceof Error ? restoreError.message : String(restoreError)
          }`,
          { cause: error },
        )
      }
    }
    throw error
  }

  if (hadPrevious) {
    try {
      await rm(backup, { recursive: true, force: true })
    } catch (error) {
      warnings.push(
        `Installed new best-model package, but could not remove previous package backup ${backup}: ${
          error instanceof Error ? error.message : String(error)
        }`,
      )
    }
  }
}

function buildUsageMarkdown(manifest: BestModelManifestValue): string {
  const artifact = manifest.artifacts[0]
  const adjustedContents = manifest.prediction_outputs?.adjusted?.enabled
    ? [
        "- `adjusted/adjusted_predictions.csv` - Predictions produced by the locked generative prediction reasoning skill.",
        "- `adjusted/adjusted_eval.json` - Adjusted test metrics and raw-vs-adjusted deltas.",
        "- `adjusted/skill/` - Adjustment skill manifest and SKILL.md, when available.",
      ]
    : []
  const reflectionContents = manifest.prediction_outputs?.window_reflection?.enabled
    ? [
        "- `reflection/final_predictions.csv` - Complete delivered predictions after independent checks, accepted repairs, and whole-window fallbacks.",
        "- `reflection/reflection_eval.json` - Pre-reflection versus final metrics and trigger/fallback counts.",
        "- `reflection/leakage-audit.json` - Blind-context leakage audit, when available.",
      ]
    : []
  const modelNote =
    artifact?.kind === "checkpoint"
      ? `- Model checkpoint: \`${artifact.path}\` (${artifact.format})`
      : artifact?.kind === "arima_results"
        ? `- Fitted ARIMA results: \`${artifact.path}\` (${artifact.format})`
        : artifact?.kind === "statistical_results"
          ? `- Fitted statistical-model results: \`${artifact.path}\` (${artifact.format})`
          : artifact?.kind === "external_reference"
            ? artifact.revision
              ? `- External model reference: \`${artifact.reference}\` at immutable revision \`${artifact.revision}\``
              : `- External model reference: \`${artifact.reference}\``
            : `- No local checkpoint: see \`${artifact?.path ?? "model/NO_CHECKPOINT.md"}\``

  return [
    "# AdaCast Best Model Package",
    "",
    `Generated: ${manifest.generated_at}`,
    `Model: ${manifest.model}`,
    `Model family: ${manifest.model_family ?? "unknown"}`,
    `Validation run: ${manifest.validation_run_id}`,
    `Test run: ${manifest.test_run_id}`,
    "",
    "## Contents",
    "",
    "- `manifest.json` - Traceability, metrics, task configuration, and model artifact metadata.",
    "- `metadata/spec.json` - Selected global test-best experiment spec.",
    "- `metadata/eval.json` - Selected global test-best fixed-evaluator metrics.",
    "- `metadata/train_scale.json` - Training-scale metadata used for scaled metrics.",
    "- `metadata/model-reuse.json` - Proof that test inference reused the selected validation model without training.",
    "- `predictions/` - Test-set predictions and NumPy arrays from the selected run.",
    ...adjustedContents,
    ...reflectionContents,
    "- `logs/train.log` - Runner log for the selected test run.",
    modelNote,
    "",
    "Use this directory as the stable handoff package for inspecting or reusing the selected forecast result.",
    "",
  ].join("\n")
}

async function validatePackage(root: string, artifact: BestModelArtifactValue): Promise<string[]> {
  const modelFiles =
    artifact.kind === "checkpoint"
      ? [path.join(root, "model", "checkpoint.pth")]
      : artifact.kind === "arima_results"
        ? [path.join(root, "model", "arima-results.pkl")]
        : artifact.kind === "statistical_results"
          ? [path.join(root, "model", "statistical-results.pkl")]
          : artifact.kind === "external_reference"
            ? [path.join(root, "model", "artifact.json")]
            : [path.join(root, "model", "NO_CHECKPOINT.md"), path.join(root, "model", "artifact.json")]
  const missing = await findMissingFiles([
    path.join(root, "manifest.json"),
    path.join(root, "USAGE.md"),
    path.join(root, "metadata", "spec.json"),
    path.join(root, "metadata", "eval.json"),
    path.join(root, "metadata", "train_scale.json"),
    path.join(root, "predictions", "predictions.csv"),
    path.join(root, "predictions", "actual.npy"),
    path.join(root, "predictions", "pred.npy"),
    path.join(root, "logs", "train.log"),
    path.join(root, "metadata", "model-reuse.json"),
    ...modelFiles,
  ])
  if (missing.length > 0) return missing
  if (artifact.kind === "checkpoint" || artifact.kind === "arima_results" || artifact.kind === "statistical_results") {
    const artifactPath = path.join(root, artifact.path)
    if ((await computeFileSha256(artifactPath)) !== artifact.sha256) {
      return [artifactPath]
    }
  }
  return []
}

function blockedResult(title: string, metadata: BestModelExportMetadata, reason: string) {
  return {
    title,
    metadata,
    output: JSON.stringify({ ...metadata, reason }),
  }
}

export const BestModelExportTool = Tool.define<typeof parameters, BestModelExportMetadata>("export_best_model", {
  description: DESCRIPTION,
  parameters,
  async execute(params) {
    const forecastRoot = path.join(Instance.directory, ".forecast")
    const exportDir = path.join(forecastRoot, "best-model")
    const manifestPath = path.join(exportDir, "manifest.json")
    const exportPath = forecastPath("best-model")
    const manifestOutputPath = forecastPath("best-model", "manifest.json")
    const warnings: string[] = []
    const missingArtifacts: string[] = []
    const blockedMetadata = (): BestModelExportMetadata => ({
      status: "blocked",
      export_path: exportPath,
      manifest_path: manifestOutputPath,
      missing_artifacts: missingArtifacts,
      warnings,
    })

    if (!runIdIsSafe(params.validation_run_id)) {
      missingArtifacts.push(`invalid validation_run_id: ${params.validation_run_id}`)
    }
    if (!runIdIsSafe(params.test_run_id)) {
      missingArtifacts.push(`invalid test_run_id: ${params.test_run_id}`)
    }
    if (missingArtifacts.length > 0) {
      return blockedResult("Best model export blocked", blockedMetadata(), "Run ids must be simple directory names")
    }

    const validationRunDir = path.join(forecastRoot, "runs", params.validation_run_id)
    const testRunDir = path.join(forecastRoot, "runs", params.test_run_id)
    const taskPath = path.join(forecastRoot, "task.json")
    const validationSpecPath = path.join(validationRunDir, "spec.json")
    const validationEvalPath = path.join(validationRunDir, "eval.json")
    const testSpecPath = path.join(testRunDir, "spec.json")
    const testEvalPath = path.join(testRunDir, "eval.json")
    const testReusePath = path.join(testRunDir, "model", "reuse.json")

    missingArtifacts.push(
      ...(await findMissingFiles([
        taskPath,
        validationSpecPath,
        validationEvalPath,
        ...REQUIRED_TEST_ARTIFACTS.map((artifact) => path.join(testRunDir, artifact.source)),
      ])),
    )
    if (missingArtifacts.length > 0) {
      return blockedResult("Best model export blocked", blockedMetadata(), "Required source artifacts are missing")
    }

    const taskRaw = await readJsonOrMissing<unknown>(taskPath, missingArtifacts)
    const validationSpec = await readJsonOrMissing<Record<string, unknown>>(validationSpecPath, missingArtifacts)
    const validationEval = await readJsonOrMissing<unknown>(validationEvalPath, missingArtifacts)
    const testSpec = await readJsonOrMissing<Record<string, unknown>>(testSpecPath, missingArtifacts)
    const testEval = await readJsonOrMissing<unknown>(testEvalPath, missingArtifacts)
    const testReuse = await readJsonOrMissing<Record<string, unknown>>(testReusePath, missingArtifacts)
    const parsedTask = TaskConfig.safeParse(taskRaw)
    if (!parsedTask.success) {
      missingArtifacts.push(taskPath)
      warnings.push("task.json is present but does not match the forecast task schema")
    }
    if (!validationSpec || !validationEval || !testSpec || !testEval || !testReuse || !parsedTask.success) {
      return blockedResult("Best model export blocked", blockedMetadata(), "Required source artifacts are corrupt")
    }
    const parsedValidationEval = EvalResult.safeParse(validationEval)
    const parsedTestEval = EvalResult.safeParse(testEval)
    if (!parsedValidationEval.success || !parsedTestEval.success) {
      missingArtifacts.push(validationEvalPath, testEvalPath)
      return blockedResult(
        "Best model export blocked",
        blockedMetadata(),
        "Validation or test evaluation artifacts are corrupt",
      )
    }
    const validationSpecHash = computeSpecHash(validationSpec)
    const testSpecHash = computeSpecHash(testSpec)
    const model = String(testSpec.model ?? "unknown")
    const statisticalArtifact = statisticalModelArtifact(model)
    let sourceIdentity
    try {
      sourceIdentity = await readBoundModelArtifactIdentity(validationRunDir, model, parsedValidationEval.data)
    } catch (error) {
      missingArtifacts.push(
        path.join(
          validationRunDir,
          "model",
          model
            .trim()
            .toLocaleLowerCase("en-US")
            .replace(/[\s_-]+/g, "") === "userprior"
            ? "prior-source.json"
            : foundationModelReference(model)
              ? "foundation.json"
              : statisticalArtifact
                ? statisticalArtifact.filename
                : "checkpoint.pth",
        ),
      )
      warnings.push(error instanceof Error ? error.message : String(error))
      return blockedResult(
        "Best model export blocked",
        blockedMetadata(),
        "Selected validation model artifact identity is invalid",
      )
    }
    let testIdentity
    try {
      testIdentity = await readBoundModelArtifactIdentity(testRunDir, model, parsedTestEval.data)
    } catch (error) {
      missingArtifacts.push(testEvalPath, testReusePath)
      warnings.push(error instanceof Error ? error.message : String(error))
      return blockedResult(
        "Best model export blocked",
        blockedMetadata(),
        "Selected test evaluation artifact binding is invalid",
      )
    }
    let normalizedValidationSpec
    let normalizedTestSpec
    try {
      normalizedValidationSpec = normalizeValidationExecutionContract(validationSpec)
      normalizedTestSpec = normalizeForecastExecutionContract(testSpec)
      if (normalizedTestSpec.phase !== "post-forecast" || normalizedTestSpec.eval_split !== "test") {
        throw new Error("Expected a post-forecast test execution contract")
      }
    } catch (error) {
      missingArtifacts.push(validationSpecPath, testSpecPath)
      warnings.push(error instanceof Error ? error.message : String(error))
      return blockedResult(
        "Best model export blocked",
        blockedMetadata(),
        "Validation or test execution contract is invalid",
      )
    }
    const validRunPair =
      parsedValidationEval.data.run_id === params.validation_run_id &&
      parsedTestEval.data.run_id === params.test_run_id &&
      parsedValidationEval.data.spec_hash === validationSpecHash &&
      parsedTestEval.data.spec_hash === testSpecHash &&
      parsedValidationEval.data.eval_split === "val" &&
      parsedTestEval.data.eval_split === "test" &&
      normalizedValidationSpec.phase === "forecast" &&
      normalizedValidationSpec.eval_split === "val" &&
      normalizedTestSpec.phase === "post-forecast" &&
      normalizedTestSpec.eval_split === "test" &&
      testReuse.source_run_id === params.validation_run_id &&
      testReuse.training_performed === false &&
      testReuse.evaluated_split === "test" &&
      testReuse.source_artifact_kind === sourceIdentity.kind &&
      testReuse.source_artifact_sha256 === sourceIdentity.sha256 &&
      testIdentity.kind === sourceIdentity.kind &&
      testIdentity.sha256 === sourceIdentity.sha256 &&
      validationSpecMatchesPostForecastTestSpec(validationSpec, testSpec)
    if (!validRunPair) {
      missingArtifacts.push(validationSpecPath, validationEvalPath, testSpecPath, testEvalPath)
      return blockedResult(
        "Best model export blocked",
        blockedMetadata(),
        "Validation and test run artifacts do not match",
      )
    }

    const modelFamily = params.model_family ?? null
    const checkpointSource = path.join(validationRunDir, "model", "checkpoint.pth")
    const arimaSource = path.join(validationRunDir, "model", "arima-results.pkl")
    const statisticalSource = path.join(validationRunDir, "model", "statistical-results.pkl")
    const hasCheckpoint = await Filesystem.exists(checkpointSource)
    const hasArimaResults = await Filesystem.exists(arimaSource)
    const hasStatisticalResults = await Filesystem.exists(statisticalSource)
    const foundationReference = foundationModelReference(model)
    if (sourceIdentity.kind === "torch_state_dict" && !hasCheckpoint) {
      missingArtifacts.push(checkpointSource)
      return blockedResult(
        "Best model export blocked",
        blockedMetadata(),
        "Selected validation model checkpoint is missing",
      )
    }
    if (sourceIdentity.kind === "statsmodels_arima_results" && !hasArimaResults) {
      missingArtifacts.push(arimaSource)
      return blockedResult(
        "Best model export blocked",
        blockedMetadata(),
        "Selected validation ARIMA results are missing",
      )
    }
    if (sourceIdentity.kind === "statsmodels_statistical_results" && !hasStatisticalResults) {
      missingArtifacts.push(statisticalSource)
      return blockedResult(
        "Best model export blocked",
        blockedMetadata(),
        "Selected validation statistical-model results are missing",
      )
    }
    const artifact: BestModelArtifactValue =
      sourceIdentity.kind === "external_prediction_prior"
        ? BestModelArtifact.parse({
            kind: "external_reference",
            reference: "user_prediction_prior",
            reason: "The selected forecast is an imported user-submitted prediction prior.",
            model,
            model_family: modelFamily,
            external_model_id: "user_prediction_prior",
            sha256: sourceIdentity.sha256,
          })
        : statisticalArtifact?.kind === "statsmodels_arima_results" && hasArimaResults
          ? BestModelArtifact.parse({
              kind: "arima_results",
              path: "model/arima-results.pkl",
              format: "statsmodels_arima_results",
              sha256: sourceIdentity.sha256,
              model,
              model_family: modelFamily,
            })
          : statisticalArtifact?.kind === "statsmodels_statistical_results" && hasStatisticalResults
            ? BestModelArtifact.parse({
                kind: "statistical_results",
                path: "model/statistical-results.pkl",
                format: "statsmodels_results_pickle",
                sha256: sourceIdentity.sha256,
                model,
                model_family: modelFamily,
              })
            : foundationReference
              ? BestModelArtifact.parse({
                  kind: "external_reference",
                  reference: foundationReference,
                  reason:
                    "The selected foundation model is reused from its pretrained repository for zero-shot inference.",
                  model,
                  model_family: modelFamily,
                  external_model_id: foundationReference,
                  revision: sourceIdentity.revision,
                  sha256: sourceIdentity.sha256,
                })
              : hasCheckpoint
                ? BestModelArtifact.parse({
                    kind: "checkpoint",
                    path: "model/checkpoint.pth",
                    format: "torch_state_dict",
                    sha256: sourceIdentity.sha256,
                    model,
                    model_family: modelFamily,
                  })
                : BestModelArtifact.parse({
                    kind: "no_checkpoint",
                    path: "model/NO_CHECKPOINT.md",
                    reason: "The selected validation run did not contain a reusable fitted model artifact.",
                    model,
                    model_family: modelFamily,
                  })

    if (artifact.kind === "no_checkpoint") {
      warnings.push(
        "Selected validation run has no reusable fitted model artifact; wrote explicit no-checkpoint marker",
      )
    }

    const manifest = BestModelManifest.parse({
      generated_at: new Date().toISOString(),
      model,
      model_family: modelFamily,
      validation_run_id: params.validation_run_id,
      test_run_id: params.test_run_id,
      validation_metrics: parsedValidationEval.data.metrics,
      test_metrics: parsedTestEval.data.metrics,
      task: parsedTask.data,
      spec_hash: parsedTestEval.data.spec_hash,
      validation_spec_hash: parsedValidationEval.data.spec_hash,
      selection_reason: params.selection_reason ?? null,
      source_paths: {
        validation_run: forecastPath("runs", params.validation_run_id),
        test_run: forecastPath("runs", params.test_run_id),
      },
      artifacts: [artifact],
      prediction_outputs: {
        raw: {
          predictions: "predictions/predictions.csv",
          metrics: parsedTestEval.data.metrics,
        },
      },
    })

    const tempDir = path.join(forecastRoot, `.best-model.tmp-${Date.now()}-${randomUUID()}`)
    await rm(tempDir, { recursive: true, force: true })
    let finalManifestForSummary = manifest

    try {
      for (const artifactFile of REQUIRED_TEST_ARTIFACTS) {
        await copyPackageFile(path.join(testRunDir, artifactFile.source), path.join(tempDir, artifactFile.target))
      }

      if (artifact.kind === "checkpoint") {
        await copyPackageFile(checkpointSource, path.join(tempDir, "model", "checkpoint.pth"))
      } else if (artifact.kind === "arima_results") {
        await copyPackageFile(arimaSource, path.join(tempDir, "model", "arima-results.pkl"))
      } else if (artifact.kind === "statistical_results") {
        await copyPackageFile(statisticalSource, path.join(tempDir, "model", "statistical-results.pkl"))
      } else if (artifact.kind === "external_reference") {
        await Filesystem.writeJson(path.join(tempDir, "model", "artifact.json"), artifact)
      } else {
        await Filesystem.write(
          path.join(tempDir, "model", "NO_CHECKPOINT.md"),
          [
            "# No Local Checkpoint",
            "",
            `Model: ${model}`,
            `Model family: ${modelFamily ?? "unknown"}`,
            "",
            "The selected validation run did not provide a reusable fitted model artifact.",
            "Use `manifest.json` and `model/artifact.json` to inspect the explicit model artifact status.",
            "",
          ].join("\n"),
        )
        await Filesystem.writeJson(path.join(tempDir, "model", "artifact.json"), artifact)
      }

      const adjusted = await findAdjustedTestArtifact(forecastRoot, params.test_run_id)
      let adjustedManifestPatch: Record<string, unknown> | null = null
      if (adjusted) {
        await copyPackageFile(adjusted.predictionsPath, path.join(tempDir, "adjusted", "adjusted_predictions.csv"))
        await copyPackageFile(adjusted.evalPath, path.join(tempDir, "adjusted", "adjusted_eval.json"))
        if (await Filesystem.exists(adjusted.auditPath)) {
          await copyPackageFile(adjusted.auditPath, path.join(tempDir, "adjusted", "leakage-audit.jsonl"))
        }
        if (adjusted.skillManifestPath) {
          await copyPackageFile(adjusted.skillManifestPath, path.join(tempDir, "adjusted", "skill", "manifest.json"))
        }
        if (adjusted.skillPath) {
          await copyPackageFile(adjusted.skillPath, path.join(tempDir, "adjusted", "skill", "SKILL.md"))
        }
        const adjustedEval = adjusted.evalJson as {
          adjusted_metrics?: Record<string, number | null>
          delta_vs_raw?: Record<string, number | null>
        }
        const activeManifest = adjusted.skillManifestPath
          ? await Filesystem.readJson<{ version?: string }>(adjusted.skillManifestPath).catch(() => null)
          : null
        const decisionAudit = await testApplicationDecisionAudit(forecastRoot, params.test_run_id)
        adjustedManifestPatch = {
          prediction_outputs: {
            ...manifest.prediction_outputs,
            adjusted: {
              enabled: true,
              predictions: "adjusted/adjusted_predictions.csv",
              metrics: adjustedEval.adjusted_metrics ?? null,
              delta_vs_raw: adjustedEval.delta_vs_raw ?? null,
              skill_version: activeManifest?.version ?? null,
              leakage_audit_passed: adjusted.auditPassed,
              confirmed_before_test_metrics: decisionAudit?.confirmedBeforeMetrics ?? null,
              decision_actor: decisionAudit?.actor ?? null,
              decision_reason: decisionAudit?.reason ?? null,
            },
          },
        }
      }

      const reflection = await findWindowReflectionArtifact(forecastRoot, params.test_run_id, params.validation_run_id)
      let reflectionOutput: Record<string, unknown> | null = null
      const reflectionBaselineIsSafe = reflection
        ? reflection.record.baseline_kind === "raw_anchor"
          ? reflection.record.adjustment_candidate_id === null && adjusted === null
          : adjusted !== null &&
            adjusted.candidateId === reflection.record.adjustment_candidate_id &&
            adjusted.auditPassed === true
        : false
      if (reflection && !reflectionBaselineIsSafe) {
        warnings.push(
          "Window reflection output was not exported because its pre-reflection baseline provenance was incomplete or unsafe",
        )
      } else if (reflection && !reflection.auditPassed) {
        warnings.push("Window reflection output was not exported because its blind-context leakage audit did not pass")
      } else if (reflection) {
        await copyPackageFile(reflection.predictionsPath, path.join(tempDir, "reflection", "final_predictions.csv"))
        await copyPackageFile(reflection.evalPath, path.join(tempDir, "reflection", "reflection_eval.json"))
        if (await Filesystem.exists(reflection.auditPath)) {
          await copyPackageFile(reflection.auditPath, path.join(tempDir, "reflection", "leakage-audit.json"))
        }
        if (reflection.protocolPath) {
          await copyPackageFile(reflection.protocolPath, path.join(tempDir, "reflection", "llm-protocol.json"))
        }
        const evalDeltaMse = Number(reflection.evalJson.delta_mse)
        reflectionOutput = {
          enabled: true,
          predictions: "reflection/final_predictions.csv",
          metrics: reflection.record.final_metrics,
          baseline_metrics: reflection.record.baseline_metrics,
          delta_mse: Number.isFinite(evalDeltaMse)
            ? evalDeltaMse
            : Number(reflection.record.final_metrics!.mse) - Number(reflection.record.baseline_metrics!.mse),
          baseline_kind: reflection.record.baseline_kind,
          validation_status: reflection.record.validation_status,
          llm_refinement_enabled: reflection.record.llm_refinement_enabled,
          trigger_count: reflection.record.trigger_count,
          skill_match_count: reflection.record.skill_match_count,
          llm_required_count: reflection.record.llm_required_count,
          corrected_count: reflection.record.corrected_count,
          fallback_count: reflection.record.fallback_count,
          deterministic_fallback_count: reflection.record.deterministic_fallback_count,
          raw_fallback_count: reflection.record.raw_fallback_count,
          llm_protocol: reflection.protocolPath ? "reflection/llm-protocol.json" : null,
          leakage_audit_passed: true,
        }
      }

      const adjustedPredictionOutputs = adjustedManifestPatch?.prediction_outputs as
        | BestModelManifestValue["prediction_outputs"]
        | undefined
      const finalManifest = BestModelManifest.parse({
        ...manifest,
        prediction_outputs: {
          ...(adjustedPredictionOutputs ?? manifest.prediction_outputs),
          ...(reflectionOutput ? { window_reflection: reflectionOutput } : {}),
        },
      })
      finalManifestForSummary = finalManifest

      await Filesystem.writeJson(path.join(tempDir, "manifest.json"), finalManifest)
      await Filesystem.write(path.join(tempDir, "USAGE.md"), buildUsageMarkdown(finalManifest))

      const packageMissing = await validatePackage(tempDir, artifact)
      if (packageMissing.length > 0) {
        missingArtifacts.push(...packageMissing)
        return blockedResult("Best model export blocked", blockedMetadata(), "Temporary package validation failed")
      }

      await mkdir(forecastRoot, { recursive: true })
      await replaceDirectory(tempDir, exportDir, warnings)
    } catch (error) {
      await rm(tempDir, { recursive: true, force: true })
      warnings.push(error instanceof Error ? error.message : String(error))
      return blockedResult("Best model export blocked", blockedMetadata(), "Export package could not be written")
    }

    const metadata: BestModelExportMetadata = {
      status: "success",
      export_path: exportPath,
      manifest_path: manifestOutputPath,
      missing_artifacts: [],
      warnings,
    }
    const summary = {
      ...metadata,
      model,
      model_family: modelFamily,
      validation_run_id: params.validation_run_id,
      test_run_id: params.test_run_id,
      artifacts: finalManifestForSummary.artifacts,
      prediction_outputs: finalManifestForSummary.prediction_outputs,
    }

    return {
      title: `Best model export: ${params.test_run_id.slice(0, 12)}`,
      metadata,
      output: JSON.stringify(summary),
    }
  },
})
