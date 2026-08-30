import path from "path"
import z from "zod"
import { Instance } from "@/project/instance"
import { Filesystem } from "@/util/filesystem"
import { Tool } from "../tool"
import { normalizeValidationExecutionContract } from "./execution-contract"
import DESCRIPTION from "./forecast-adjustment-fit-runs.txt"
import { computeSpecHash } from "./hash"
import { resolveForecastModelEnvironment } from "./model-environment"
import { readBoundModelArtifactIdentity } from "./model-artifact"
import { resolveForecastPythonProject } from "./python-project"
import {
  AdjustmentFitRunManifest,
  AdjustmentFitRunMember,
  AdjustmentSourceModelSet,
  EvalResult,
  FIXED_ADJUSTMENT_FIT_HOLDOUT_RATIO,
  RunnerOutput,
} from "./types"
import {
  adjustmentRoot,
  ensureAdjustmentDirs,
  forecastPath,
  forecastRoot,
  projectRelative,
  readJsonFile,
  readPhaseState,
  runIdIsSafe,
  writeJsonFile,
  writePhaseState,
} from "./forecast-adjustment-common"

const parameters = z.object({
  subcommand: z.enum(["prepare", "run", "read"]),
  families: z.array(z.string()).optional().describe("Optional family filter from source-model-set members"),
  timeout_ms: z
    .number()
    .int()
    .positive()
    .max(24 * 60 * 60 * 1000)
    .optional()
    .describe("Per-fit-run timeout"),
})

const DEFAULT_TIMEOUT_MS = 30 * 60 * 1000
const FOUNDATION_MODEL_NAMES = new Set(["Chronos", "Chronos2", "TimesFM", "Moirai", "Sundial", "TiRex", "TimeMoE"])
const FOUNDATION_MODEL_NAME_BY_CASEFOLD = new Map(
  [...FOUNDATION_MODEL_NAMES].map((name) => [name.toLocaleLowerCase("en-US"), name]),
)

type FitMember = z.infer<typeof AdjustmentFitRunMember>
type RunnerOutputValue = z.infer<typeof RunnerOutput>

function manifestPath(): string {
  return path.join(adjustmentRoot(), "fit-runs", "manifest.json")
}

function sourceSetPath(): string {
  return path.join(adjustmentRoot(), "source-model-set.json")
}

function safeSegment(value: string): string {
  const cleaned = value.replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "")
  return (cleaned || "member").slice(0, 80)
}

function isFoundationModelName(model: string): boolean {
  return FOUNDATION_MODEL_NAME_BY_CASEFOLD.has(model.toLocaleLowerCase("en-US"))
}

function projectRelativeToAbsolute(relativePath: string): string {
  return path.join(Instance.directory, relativePath)
}

function fitSpecFromValidationSpec(validationSpec: Record<string, unknown>): Record<string, unknown> {
  const normalizedValidationSpec = normalizeValidationExecutionContract(validationSpec)
  const { window_features: _windowFeatures, ...spec } = normalizedValidationSpec
  const holdoutRatio = Number(normalizedValidationSpec.adjustment_fit_holdout_ratio)
  if (!Number.isFinite(holdoutRatio) || Math.abs(holdoutRatio - FIXED_ADJUSTMENT_FIT_HOLDOUT_RATIO) > 1e-12) {
    throw new Error("Adjustment source spec does not use the fixed validation-front evidence contract")
  }

  return {
    ...spec,
    eval_split: "adjustment_fit",
  }
}

function aggregateStatus(members: FitMember[]): z.infer<typeof AdjustmentFitRunManifest>["status"] {
  if (members.every((member) => member.status === "prepared")) return "prepared"
  const isComplete = (member: FitMember) =>
    (member.status === "success" || member.status === "cached") &&
    Boolean(member.predictions_path) &&
    Boolean(member.eval_path)
  if (members.every(isComplete)) return "complete"
  if (members.some(isComplete)) return "partial"
  return "failed"
}

function cleanEnv(extra: Record<string, string>): Record<string, string> {
  const base: Record<string, string> = {}
  for (const [key, value] of Object.entries(process.env)) {
    if (typeof value === "string") base[key] = value
  }
  return { ...base, ...extra }
}

async function spawnJson(
  command: string[],
  timeoutMs: number,
  env: Record<string, string>,
): Promise<{
  stdout: string
  stderr: string
  exitCode: number
  timedOut: boolean
}> {
  const proc = Bun.spawn(command, {
    cwd: Instance.directory,
    env,
    stdout: "pipe",
    stderr: "pipe",
  })
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    try {
      proc.kill("SIGTERM")
    } catch {
      // Process already exited.
    }
  }, timeoutMs)

  try {
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
      proc.exited.catch(() => 1),
    ])
    return { stdout, stderr, exitCode, timedOut }
  } finally {
    clearTimeout(timer)
  }
}

async function readSourceSet(): Promise<z.infer<typeof AdjustmentSourceModelSet>> {
  return AdjustmentSourceModelSet.parse(await readJsonFile(sourceSetPath()))
}

async function readMatchingManifest(families?: string[]): Promise<z.infer<typeof AdjustmentFitRunManifest> | null> {
  const existing = await readJsonFile(manifestPath())
    .then((value) => AdjustmentFitRunManifest.parse(value))
    .catch(() => null)
  if (!existing) return null

  const sourceSet = await readSourceSet()
  if (existing.source_model_set_id !== sourceSet.source_model_set_id) return null

  const allowedFamilies = families ? new Set(families) : null
  const expectedRunIds = sourceSet.members
    .filter((member) => !allowedFamilies || allowedFamilies.has(member.family))
    .map((member) => member.validation_run_id)
    .sort()
  const existingRunIds = existing.members.map((member) => member.validation_run_id).sort()
  if (JSON.stringify(existingRunIds) !== JSON.stringify(expectedRunIds)) return null
  return existing
}

async function prepareManifest(families?: string[]): Promise<z.infer<typeof AdjustmentFitRunManifest>> {
  const sourceSet = await readSourceSet()
  const allowedFamilies = families ? new Set(families) : null
  const members: FitMember[] = []
  const now = new Date().toISOString()

  for (const source of sourceSet.members) {
    if (allowedFamilies && !allowedFamilies.has(source.family)) continue
    if (!runIdIsSafe(source.validation_run_id)) {
      throw new Error(`Invalid validation_run_id in source model set: ${source.validation_run_id}`)
    }

    const validationSpecPath = path.join(forecastRoot(), "runs", source.validation_run_id, "spec.json")
    const validationSpec = await readJsonFile<Record<string, unknown>>(validationSpecPath)
    const validationSpecHash = computeSpecHash(validationSpec)
    if (validationSpecHash !== source.spec_hash) {
      throw new Error(
        `Source spec hash mismatch for ${source.validation_run_id}: source-set=${source.spec_hash}, spec.json=${validationSpecHash}`,
      )
    }

    const fitSpec = fitSpecFromValidationSpec(validationSpec)
    const fitRunId = computeSpecHash(fitSpec)
    const familySegment = safeSegment(source.family)
    const specFile = `${familySegment}-${source.validation_run_id}.json`
    const specPath = path.join(adjustmentRoot(), "fit-runs", "specs", specFile)
    await writeJsonFile(specPath, fitSpec)

    members.push(
      AdjustmentFitRunMember.parse({
        family: source.family,
        validation_run_id: source.validation_run_id,
        model: source.model,
        validation_spec_hash: validationSpecHash,
        fit_spec_hash: fitRunId,
        fit_run_id: fitRunId,
        status: "prepared",
        fit_method: "runner_reuse_validation_front_half",
        spec_path: projectRelative(specPath),
        fit_run_path: forecastPath("adjustments", "fit-runs", "runs", fitRunId),
        predictions_path: null,
        eval_path: null,
        metrics: null,
        error: null,
      }),
    )
  }

  if (members.length === 0) {
    throw new Error("No source model members matched the fit-run family filter")
  }

  const manifest = AdjustmentFitRunManifest.parse({
    source_model_set_id: sourceSet.source_model_set_id,
    created_at: now,
    updated_at: now,
    status: "prepared",
    members,
  })
  await writeJsonFile(manifestPath(), manifest)

  const state = await readPhaseState()
  await writePhaseState(
    {
      ...state,
      adjustment: {
        ...state.adjustment,
        fit_run_manifest_path: forecastPath("adjustments", "fit-runs", "manifest.json"),
        fit_runs_ready: false,
      },
    },
    `## Adjustment Fit Runs Prepared\n\nPrepared ${members.length} validation-front fit replay spec(s).`,
  )

  return manifest
}

async function runMember(member: FitMember, timeoutMs: number): Promise<FitMember> {
  const runDir = path.join(adjustmentRoot(), "fit-runs", "runs", member.fit_run_id)
  const predictionsPath = path.join(runDir, "predictions.csv")
  const evalPath = path.join(runDir, "eval.json")
  const reusePath = path.join(runDir, "model", "reuse.json")
  const sourceRunDir = path.join(forecastRoot(), "runs", member.validation_run_id)

  if (
    (await Filesystem.exists(evalPath)) &&
    (await Filesystem.exists(predictionsPath)) &&
    (await Filesystem.exists(reusePath))
  ) {
    const evalResult = EvalResult.parse(await readJsonFile(evalPath))
    const reuse = await readJsonFile<Record<string, unknown>>(reusePath)
    const bindingValid = await readBoundModelArtifactIdentity(runDir, member.model, evalResult, sourceRunDir)
      .then(() => true)
      .catch(() => false)
    if (
      evalResult.run_id === member.fit_run_id &&
      evalResult.spec_hash === member.fit_run_id &&
      reuse.source_run_id === member.validation_run_id &&
      reuse.training_performed === false &&
      bindingValid
    ) {
      return AdjustmentFitRunMember.parse({
        ...member,
        status: "cached",
        predictions_path: projectRelative(predictionsPath),
        eval_path: projectRelative(evalPath),
        metrics: evalResult.metrics,
        error: null,
      })
    }
  }

  const pythonDir = await resolveForecastPythonProject(Instance.directory)
  const modelEnvironment = resolveForecastModelEnvironment({ directory: Instance.directory, env: process.env })
  const env = cleanEnv(modelEnvironment.env)
  const evaluatorHashStore = path.join(forecastRoot(), "evaluator.sha256")
  const specPath = projectRelativeToAbsolute(member.spec_path)

  const command = [
    "uv",
    "run",
    "--project",
    pythonDir,
    ...(isFoundationModelName(member.model) ? ["--extra", "heavy-models"] : []),
    "python",
    "-m",
    "adacast_ml.runner",
    "--spec",
    specPath,
    "--run-id",
    member.fit_run_id,
    "--out-dir",
    path.join(adjustmentRoot(), "fit-runs", "runs"),
    "--evaluator-hash-store",
    evaluatorHashStore,
    "--reuse-model-from",
    sourceRunDir,
  ]

  const result = await spawnJson(command, timeoutMs, env)
  if (result.timedOut) {
    return AdjustmentFitRunMember.parse({ ...member, status: "timeout", error: "timeout" })
  }

  let output: RunnerOutputValue
  try {
    output = RunnerOutput.parse(JSON.parse(result.stdout.trim()))
  } catch {
    return AdjustmentFitRunMember.parse({
      ...member,
      status: "ipc_error",
      error: result.stderr.split("\n").filter(Boolean).slice(-5).join("\n") || "runner_stdout_invalid",
    })
  }

  if (output.status === "error") {
    return AdjustmentFitRunMember.parse({
      ...member,
      status: "crash",
      error: output.error ?? output.error_type ?? "runner_error",
    })
  }

  if (!/^[A-Fa-f0-9]{64}$/.test(output.run_id)) {
    return AdjustmentFitRunMember.parse({
      ...member,
      status: "ipc_error",
      error: `runner returned invalid fit run_id: ${output.run_id}`,
    })
  }

  const actualRunId = output.run_id
  const actualRunDir = path.join(adjustmentRoot(), "fit-runs", "runs", actualRunId)
  const actualPredictionsPath = path.join(actualRunDir, "predictions.csv")
  const actualEvalPath = path.join(actualRunDir, "eval.json")
  const actualReusePath = path.join(actualRunDir, "model", "reuse.json")
  const hasPredictions = await Filesystem.exists(actualPredictionsPath)
  const hasEval = await Filesystem.exists(actualEvalPath)
  const hasReuse = await Filesystem.exists(actualReusePath)
  const completedEval = hasEval ? EvalResult.parse(await readJsonFile(actualEvalPath)) : null
  const reuse = hasReuse ? await readJsonFile<Record<string, unknown>>(actualReusePath) : null
  const bindingValid =
    completedEval && hasPredictions && hasReuse
      ? await readBoundModelArtifactIdentity(actualRunDir, member.model, completedEval, sourceRunDir)
          .then(() => true)
          .catch(() => false)
      : false
  if (
    !hasPredictions ||
    !hasEval ||
    !hasReuse ||
    reuse?.source_run_id !== member.validation_run_id ||
    reuse?.training_performed !== false ||
    !bindingValid
  ) {
    const missing = [
      ...(!hasPredictions ? [projectRelative(actualPredictionsPath)] : []),
      ...(!hasEval ? [projectRelative(actualEvalPath)] : []),
      ...(!hasReuse ? [projectRelative(actualReusePath)] : []),
    ]
    if (hasReuse && reuse?.source_run_id !== member.validation_run_id) {
      missing.push("reuse source_run_id mismatch")
    }
    if (hasReuse && reuse?.training_performed !== false) {
      missing.push("reuse provenance does not prove training_performed=false")
    }
    if (completedEval && !bindingValid) {
      missing.push("eval.json is not bound to the current reused model and prediction artifacts")
    }
    return AdjustmentFitRunMember.parse({
      ...member,
      fit_run_id: actualRunId,
      fit_run_path: forecastPath("adjustments", "fit-runs", "runs", actualRunId),
      status: "ipc_error",
      predictions_path: hasPredictions ? projectRelative(actualPredictionsPath) : null,
      eval_path: hasEval ? projectRelative(actualEvalPath) : null,
      metrics: completedEval?.metrics ?? output.metrics ?? null,
      error: `runner completed without required adjustment fit artifact(s): ${missing.join(", ")}`,
    })
  }
  if (completedEval && (completedEval.run_id !== actualRunId || completedEval.spec_hash !== actualRunId)) {
    return AdjustmentFitRunMember.parse({
      ...member,
      fit_run_id: actualRunId,
      fit_run_path: forecastPath("adjustments", "fit-runs", "runs", actualRunId),
      status: "ipc_error",
      predictions_path: projectRelative(actualPredictionsPath),
      eval_path: projectRelative(actualEvalPath),
      metrics: completedEval.metrics,
      error: "runner fit eval identity does not match the returned run_id",
    })
  }

  return AdjustmentFitRunMember.parse({
    ...member,
    fit_run_id: actualRunId,
    fit_run_path: forecastPath("adjustments", "fit-runs", "runs", actualRunId),
    status: output.status === "cached" ? "cached" : "success",
    predictions_path: projectRelative(actualPredictionsPath),
    eval_path: projectRelative(actualEvalPath),
    metrics: completedEval?.metrics ?? output.metrics ?? null,
    error: null,
  })
}

async function runPreparedManifest(input: z.infer<typeof AdjustmentFitRunManifest>, timeoutMs: number) {
  const members: FitMember[] = []
  for (const member of input.members) {
    members.push(await runMember(member, timeoutMs))
  }
  const now = new Date().toISOString()
  const manifest = AdjustmentFitRunManifest.parse({
    ...input,
    updated_at: now,
    status: aggregateStatus(members),
    members,
  })
  await writeJsonFile(manifestPath(), manifest)

  const state = await readPhaseState()
  await writePhaseState(
    {
      ...state,
      adjustment: {
        ...state.adjustment,
        fit_run_manifest_path: forecastPath("adjustments", "fit-runs", "manifest.json"),
        fit_runs_ready: manifest.status === "complete",
      },
    },
    `## Adjustment Fit Runs ${manifest.status === "complete" ? "Completed" : "Updated"}\n\nStatus: ${manifest.status}.`,
  )
  return manifest
}

export const ForecastAdjustmentFitRunsTool = Tool.define("forecast_adjustment_fit_runs", {
  description: DESCRIPTION,
  parameters,
  async execute(params) {
    await ensureAdjustmentDirs()
    let result: unknown

    switch (params.subcommand) {
      case "read": {
        result = await readJsonFile(manifestPath()).catch(() => null)
        break
      }

      case "prepare": {
        result = await prepareManifest(params.families)
        break
      }

      case "run": {
        const existing = await readMatchingManifest(params.families)
        const prepared = existing ?? (await prepareManifest(params.families))
        result = await runPreparedManifest(prepared, params.timeout_ms ?? DEFAULT_TIMEOUT_MS)
        break
      }
    }

    return {
      title: `Forecast adjustment fit runs: ${params.subcommand}`,
      metadata: { subcommand: params.subcommand },
      output: JSON.stringify(result),
    }
  },
})
