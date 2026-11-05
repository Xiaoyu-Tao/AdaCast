import path from "node:path"
import z from "zod"
import { Filesystem } from "./runtime"
import { Tool } from "./runtime"
import { validationSpecMatchesPostForecastTestSpec } from "./execution-contract"
import DESCRIPTION from "./forecast-adjustment-apply.txt"
import {
  AdjustmentCandidate,
  AdjustmentEvalResult,
  AdjustmentFitRunManifest,
  AdjustmentPolicy,
  AdjustmentSkillManifest,
  AdjustmentSourceModelSet,
  EvalResult,
  parseStoredAdjustmentPolicy,
} from "./types"
import { readBoundModelArtifactIdentity } from "./model-artifact"
import {
  adjustmentRoot,
  forecastRoot,
  projectPath,
  projectRelative,
  readJsonFile,
  readPhaseState,
  runAdjustmentPython,
  runIdIsSafe,
  writePhaseState,
} from "./forecast-adjustment-common"

const parameters = z.object({
  run_id: z.string().describe("Raw source run id to adjust"),
  candidate_id: z.string().describe("Candidate id under .forecast/adjustments/candidates"),
  split: z.enum(["fit", "val", "test"]).describe("Split label for adjusted artifacts"),
})

function splitDir(split: "fit" | "val" | "test"): string {
  return split === "val" ? "validation" : split
}

async function readPolicy(policyPath: string): Promise<z.infer<typeof AdjustmentPolicy>> {
  try {
    return parseStoredAdjustmentPolicy(await readJsonFile(policyPath))
  } catch {
    return AdjustmentPolicy.parse({})
  }
}

function projectOrAbsolute(file: string): string {
  return path.isAbsolute(file) ? file : projectPath(file)
}

async function assertLockedSkillWasAcceptedForRawTestSpec(
  activeManifest: z.infer<typeof AdjustmentSkillManifest>,
  runId: string,
): Promise<void> {
  const runRoot = path.join(forecastRoot(), "runs", runId)
  const rawTestSpec = z.record(z.string(), z.unknown()).parse(await readJsonFile(path.join(runRoot, "spec.json")))
  if (rawTestSpec.phase !== "post-forecast" || rawTestSpec.eval_split !== "test") {
    throw new Error("Test adjustment requires run_id to reference a post-forecast test run")
  }

  const rawEval = EvalResult.parse(await readJsonFile(path.join(runRoot, "eval.json")))
  if (rawEval.run_id !== runId || rawEval.eval_split !== "test") {
    throw new Error("Test adjustment requires matching raw test eval.json")
  }
  await readBoundModelArtifactIdentity(runRoot, String(rawTestSpec.model ?? ""), rawEval)

  for (const validationPath of activeManifest.validation_results) {
    const validation = AdjustmentEvalResult.parse(await readJsonFile(projectOrAbsolute(validationPath)))
    if (validation.split !== "val" || validation.candidate_id !== activeManifest.candidate_id) {
      throw new Error("Locked active skill validation result does not match the locked candidate")
    }
    if (!validation.acceptance_decision?.accepted) {
      throw new Error("Locked active skill validation result is not accepted")
    }
    if (!runIdIsSafe(validation.run_id)) continue
    const validationSpec = z
      .record(z.string(), z.unknown())
      .parse(await readJsonFile(path.join(forecastRoot(), "runs", validation.run_id, "spec.json")))
    if (validationSpecMatchesPostForecastTestSpec(validationSpec, rawTestSpec)) return
  }

  throw new Error(
    "Locked adjustment skill was not validation-accepted on this test model spec; only matching models in the current test collection may be adjusted",
  )
}

function testApplicationForRun(state: Awaited<ReturnType<typeof readPhaseState>>, runId: string) {
  const record = state.adjustment.test_application_records.find((item) => item.test_run_id === runId)
  if (record) return record
  if (state.adjustment.test_application_run_id !== runId) return null
  return {
    test_run_id: runId,
    validation_run_id: "",
    model: null,
    decision: state.adjustment.test_application_decision,
    decision_at: state.adjustment.test_application_decision_at,
    decision_actor: state.adjustment.test_application_decision_actor,
    decision_reason: state.adjustment.test_application_decision_reason,
    adjustment_locked: state.adjustment.test_adjustment_locked && state.adjustment.test_adjustment_run_id === runId,
    adjustment_candidate_id:
      state.adjustment.test_adjustment_run_id === runId ? state.adjustment.test_adjustment_candidate_id : null,
    adjustment_evaluated: false,
    adjusted_eval_path: null,
  }
}

async function assertTestApplyAllowed(runId: string, candidateId: string, policyPath: string): Promise<void> {
  const policy = await readPolicy(policyPath)
  const state = await readPhaseState()
  if (!policy.lock_before_test) {
    throw new Error("Test adjustment cannot disable lock_before_test in the deterministic adjustment protocol")
  }
  if (
    !state.adjustment.forecaster_locked ||
    !state.adjustment.active_skill_version ||
    state.adjustment.no_effective_skill
  ) {
    throw new Error("Test adjustment requires a locked active adjustment skill before adjusted test metrics")
  }
  const application = testApplicationForRun(state, runId)
  if (!application || application.decision !== "apply") {
    throw new Error("Test adjustment requires an apply decision for this eligible test run")
  }
  const legacyConfirmed =
    state.adjustment.test_application_records.length === 0 && Boolean(state.adjustment.critic_user_confirmed)
  if (!application.decision_at || (!application.decision_actor && !legacyConfirmed)) {
    throw new Error(
      "Test adjustment requires a recorded forecast_adjustment_decision test application approval before adjusted test metrics",
    )
  }

  const manifestPath = path.join(adjustmentRoot(), "skills", "active", "manifest.json")
  const activeManifest = AdjustmentSkillManifest.parse(await readJsonFile(manifestPath))
  if (activeManifest.status !== "locked") {
    throw new Error("Test adjustment requires the active adjustment skill manifest to be locked")
  }
  if (activeManifest.version !== state.adjustment.active_skill_version) {
    throw new Error(
      `Test adjustment active skill version ${state.adjustment.active_skill_version} does not match active manifest ${activeManifest.version}`,
    )
  }
  if (activeManifest.candidate_id !== candidateId) {
    throw new Error(
      `Test adjustment candidate_id ${candidateId} does not match locked active skill candidate_id ${activeManifest.candidate_id}`,
    )
  }
  await assertLockedSkillWasAcceptedForRawTestSpec(activeManifest, runId)
}

async function assertValidationApplyAllowed(runId: string, candidateId: string): Promise<void> {
  const state = await readPhaseState()
  if (state.phase !== "forecast") {
    throw new Error("Validation adjustment apply is only allowed during the forecast phase")
  }

  const candidate = AdjustmentCandidate.parse(
    await readJsonFile(path.join(adjustmentRoot(), "candidates", candidateId, "candidate.json")),
  )
  const sourceSet = AdjustmentSourceModelSet.parse(
    await readJsonFile(path.join(adjustmentRoot(), "source-model-set.json")),
  )
  if (candidate.source_model_set_id !== sourceSet.source_model_set_id) {
    throw new Error("Validation adjustment apply requires a candidate from the frozen source model set")
  }
  if (!sourceSet.members.some((member) => member.validation_run_id === runId)) {
    throw new Error("Validation adjustment apply requires a validation run from the frozen source model set")
  }
}

async function assertFitApplyAllowed(runId: string, candidateId: string): Promise<void> {
  const state = await readPhaseState()
  if (state.phase !== "forecast") {
    throw new Error("Fit adjustment apply is only allowed during the forecast phase")
  }
  if (!state.adjustment.source_model_set_locked || !state.adjustment.fit_runs_ready) {
    throw new Error("Fit adjustment apply requires a frozen source set and completed fit runs")
  }

  const candidate = AdjustmentCandidate.parse(
    await readJsonFile(path.join(adjustmentRoot(), "candidates", candidateId, "candidate.json")),
  )
  const sourceSet = AdjustmentSourceModelSet.parse(
    await readJsonFile(path.join(adjustmentRoot(), "source-model-set.json")),
  )
  const fitManifest = AdjustmentFitRunManifest.parse(
    await readJsonFile(path.join(adjustmentRoot(), "fit-runs", "manifest.json")),
  )
  if (candidate.source_model_set_id !== sourceSet.source_model_set_id) {
    throw new Error("Fit adjustment apply requires a candidate from the frozen source model set")
  }
  if (fitManifest.source_model_set_id !== sourceSet.source_model_set_id) {
    throw new Error("Fit adjustment apply requires fit runs derived from the frozen source model set")
  }
  const member = fitManifest.members.find((item) => item.fit_run_id === runId)
  if (!member || (member.status !== "success" && member.status !== "cached") || !member.predictions_path) {
    throw new Error("Fit adjustment apply requires a completed fit run from the frozen source model set")
  }
}

function rawRunRoot(split: "fit" | "val" | "test", runId: string): string {
  return split === "fit"
    ? path.join(adjustmentRoot(), "fit-runs", "runs", runId)
    : path.join(forecastRoot(), "runs", runId)
}

export const ForecastAdjustmentApplyTool = Tool.define("forecast_adjustment_apply", {
  description: DESCRIPTION,
  parameters,
  async execute(params) {
    if (!runIdIsSafe(params.run_id)) throw new Error(`Invalid run_id "${params.run_id}"`)
    if (!runIdIsSafe(params.candidate_id)) throw new Error(`Invalid candidate_id "${params.candidate_id}"`)

    const root = forecastRoot()
    const rawPredictions = path.join(rawRunRoot(params.split, params.run_id), "predictions.csv")
    const taskPath = path.join(root, "task.json")
    const candidatePath = path.join(adjustmentRoot(), "candidates", params.candidate_id, "candidate.json")
    const policyPath = path.join(adjustmentRoot(), "policy.json")
    const outDir = path.join(adjustmentRoot(), splitDir(params.split), params.run_id, params.candidate_id)

    const missing = []
    for (const file of [rawPredictions, taskPath, candidatePath]) {
      if (!(await Filesystem.exists(file))) missing.push(projectRelative(file))
    }
    if (missing.length > 0) {
      throw new Error(`Required adjustment apply artifact(s) missing: ${missing.join(", ")}`)
    }

    if (params.split === "test") {
      await assertTestApplyAllowed(params.run_id, params.candidate_id, policyPath)
    } else if (params.split === "val") {
      await assertValidationApplyAllowed(params.run_id, params.candidate_id)
    } else if (params.split === "fit") {
      await assertFitApplyAllowed(params.run_id, params.candidate_id)
    }

    const output = await runAdjustmentPython("adacast_lite_ml.engine.adjustment.apply", [
      "--task-json",
      taskPath,
      "--predictions-csv",
      rawPredictions,
      "--candidate-json",
      candidatePath,
      "--out-dir",
      outDir,
      "--source-run-id",
      params.run_id,
      "--split",
      params.split,
      ...((await Filesystem.exists(policyPath)) ? ["--policy-json", policyPath] : []),
    ])

    if (params.split === "test") {
      const state = await readPhaseState()
      const records = state.adjustment.test_application_records.map((record) =>
        record.test_run_id === params.run_id
          ? {
              ...record,
              adjustment_locked: true,
              adjustment_candidate_id: params.candidate_id,
              adjustment_evaluated: false,
              adjusted_eval_path: null,
            }
          : record,
      )
      await writePhaseState(
        {
          ...state,
          adjustment: {
            ...state.adjustment,
            test_application_records: records,
            test_adjustment_locked: true,
            test_adjustment_run_id: params.run_id,
            test_adjustment_candidate_id: params.candidate_id,
          },
        },
        `## Test Adjustment Values Locked\n\nAdjusted values for ${params.run_id} / ${params.candidate_id} were locked before adjusted test metrics.`,
      )
    }

    return {
      title: `Forecast adjustment apply: ${params.run_id.slice(0, 12)}`,
      metadata: { run_id: params.run_id, candidate_id: params.candidate_id, split: params.split },
      output: JSON.stringify(output),
    }
  },
})
