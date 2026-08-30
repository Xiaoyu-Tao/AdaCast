import path from "path"
import z from "zod"
import { Filesystem } from "@/util/filesystem"
import { Tool } from "../tool"
import DESCRIPTION from "./forecast-adjustment-evaluate.txt"
import {
  AdjustmentCandidate,
  AdjustmentEvalResult,
  AdjustmentFitRunManifest,
  AdjustmentSkillManifest,
  AdjustmentSourceModelSet,
  EvalResult,
} from "./types"
import { readBoundModelArtifactIdentity } from "./model-artifact"
import {
  adjustmentRoot,
  forecastRoot,
  projectRelative,
  readJsonFile,
  readPhaseState,
  runAdjustmentPython,
  runIdIsSafe,
  writeJsonFile,
  writePhaseState,
} from "./forecast-adjustment-common"

const parameters = z.object({
  run_id: z.string().describe("Raw source run id"),
  candidate_id: z.string().describe("Candidate id under .forecast/adjustments/candidates"),
  split: z.enum(["fit", "val", "test"]).describe("Split label for adjusted artifacts"),
})

function splitDir(split: "fit" | "val" | "test"): string {
  return split === "val" ? "validation" : split
}

async function assertTestEvaluateAllowed(runId: string, candidateId: string): Promise<void> {
  const state = await readPhaseState()
  const record = state.adjustment.test_application_records.find((item) => item.test_run_id === runId)
  const recordLocked =
    record?.adjustment_locked === true && record.adjustment_candidate_id === candidateId && record.decision === "apply"
  const legacyLocked =
    state.adjustment.test_application_records.length === 0 &&
    state.adjustment.test_adjustment_locked &&
    state.adjustment.test_adjustment_run_id === runId &&
    state.adjustment.test_adjustment_candidate_id === candidateId
  if (!recordLocked && !legacyLocked) {
    throw new Error("Adjusted test evaluation requires matching locked test adjustment values")
  }

  const manifestPath = path.join(adjustmentRoot(), "skills", "active", "manifest.json")
  const activeManifest = AdjustmentSkillManifest.parse(await readJsonFile(manifestPath))
  if (
    activeManifest.status !== "locked" ||
    activeManifest.version !== state.adjustment.active_skill_version ||
    activeManifest.candidate_id !== candidateId
  ) {
    throw new Error("Adjusted test evaluation requires the matching locked active adjustment skill")
  }
}

async function assertValidationEvaluateAllowed(runId: string, candidateId: string): Promise<void> {
  const state = await readPhaseState()
  if (state.phase !== "forecast") {
    throw new Error("Validation adjustment evaluation is only allowed during the forecast phase")
  }

  const candidate = AdjustmentCandidate.parse(
    await readJsonFile(path.join(adjustmentRoot(), "candidates", candidateId, "candidate.json")),
  )
  const sourceSet = AdjustmentSourceModelSet.parse(
    await readJsonFile(path.join(adjustmentRoot(), "source-model-set.json")),
  )
  if (candidate.source_model_set_id !== sourceSet.source_model_set_id) {
    throw new Error("Validation adjustment evaluation requires a candidate from the frozen source model set")
  }
  if (!sourceSet.members.some((member) => member.validation_run_id === runId)) {
    throw new Error("Validation adjustment evaluation requires a validation run from the frozen source model set")
  }
}

async function assertFitEvaluateAllowed(runId: string, candidateId: string): Promise<void> {
  const state = await readPhaseState()
  if (state.phase !== "forecast") {
    throw new Error("Fit adjustment evaluation is only allowed during the forecast phase")
  }
  if (!state.adjustment.source_model_set_locked || !state.adjustment.fit_runs_ready) {
    throw new Error("Fit adjustment evaluation requires a frozen source set and completed fit runs")
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
    throw new Error("Fit adjustment evaluation requires a candidate from the frozen source model set")
  }
  if (fitManifest.source_model_set_id !== sourceSet.source_model_set_id) {
    throw new Error("Fit adjustment evaluation requires fit runs derived from the frozen source model set")
  }
  const member = fitManifest.members.find((item) => item.fit_run_id === runId)
  if (
    !member ||
    (member.status !== "success" && member.status !== "cached") ||
    !member.predictions_path ||
    !member.eval_path
  ) {
    throw new Error("Fit adjustment evaluation requires a completed fit run from the frozen source model set")
  }
}

function rawRunRoot(split: "fit" | "val" | "test", runId: string): string {
  return split === "fit"
    ? path.join(adjustmentRoot(), "fit-runs", "runs", runId)
    : path.join(forecastRoot(), "runs", runId)
}

async function assertRawEvaluationBound(split: "fit" | "val" | "test", runId: string, rawRun: string): Promise<void> {
  const spec = z.record(z.string(), z.unknown()).parse(await readJsonFile(path.join(rawRun, "spec.json")))
  const evalResult = EvalResult.parse(await readJsonFile(path.join(rawRun, "eval.json")))
  const sourceRunDir =
    split === "fit"
      ? path.join(forecastRoot(), "runs", String(evalResult.artifact_binding?.model_artifact.source_run_id ?? ""))
      : undefined
  await readBoundModelArtifactIdentity(rawRun, String(spec.model ?? ""), evalResult, sourceRunDir)
  if (evalResult.run_id !== runId || evalResult.eval_split !== (split === "fit" ? "adjustment_fit" : split)) {
    throw new Error("Raw evaluation identity does not match the requested adjustment split")
  }
}

export const ForecastAdjustmentEvaluateTool = Tool.define("forecast_adjustment_evaluate", {
  description: DESCRIPTION,
  parameters,
  async execute(params) {
    if (!runIdIsSafe(params.run_id)) throw new Error(`Invalid run_id "${params.run_id}"`)
    if (!runIdIsSafe(params.candidate_id)) throw new Error(`Invalid candidate_id "${params.candidate_id}"`)

    const outDir = path.join(adjustmentRoot(), splitDir(params.split), params.run_id, params.candidate_id)
    const adjustedPredictions = path.join(outDir, "adjusted_predictions.csv")
    const adjustedEval = path.join(outDir, "adjusted_eval.json")
    const applyResults = path.join(outDir, "apply-results.json")
    const rawRun = rawRunRoot(params.split, params.run_id)
    const rawEval = path.join(rawRun, "eval.json")
    const trainScale = path.join(rawRun, "train_scale.json")
    const policyPath = path.join(adjustmentRoot(), "policy.json")
    const candidatePath = path.join(adjustmentRoot(), "candidates", params.candidate_id, "candidate.json")

    if (!(await Filesystem.exists(adjustedPredictions))) {
      throw new Error(`Missing adjusted predictions: ${projectRelative(adjustedPredictions)}`)
    }
    if (!(await Filesystem.exists(candidatePath))) {
      throw new Error(`Missing adjustment candidate: ${projectRelative(candidatePath)}`)
    }
    if (!(await Filesystem.exists(applyResults))) {
      throw new Error(`Missing adjustment apply results: ${projectRelative(applyResults)}`)
    }
    if (params.split === "test") {
      await assertTestEvaluateAllowed(params.run_id, params.candidate_id)
    } else if (params.split === "val") {
      await assertValidationEvaluateAllowed(params.run_id, params.candidate_id)
    } else if (params.split === "fit") {
      await assertFitEvaluateAllowed(params.run_id, params.candidate_id)
    }
    await assertRawEvaluationBound(params.split, params.run_id, rawRun)

    const output = await runAdjustmentPython("adacast_ml.adjustment.evaluate", [
      "--adjusted-predictions-csv",
      adjustedPredictions,
      "--out",
      adjustedEval,
      "--run-id",
      params.run_id,
      "--candidate-id",
      params.candidate_id,
      "--split",
      params.split,
      "--candidate-json",
      candidatePath,
      "--apply-results-json",
      applyResults,
      ...((await Filesystem.exists(rawEval)) ? ["--raw-eval-json", rawEval] : []),
      ...((await Filesystem.exists(trainScale)) ? ["--train-scale-json", trainScale] : []),
      ...((await Filesystem.exists(policyPath)) ? ["--policy-json", policyPath] : []),
    ])
    const parsed = AdjustmentEvalResult.parse(output)
    await writeJsonFile(adjustedEval, parsed)
    if (params.split === "test") {
      const state = await readPhaseState()
      const records = state.adjustment.test_application_records.map((record) =>
        record.test_run_id === params.run_id
          ? {
              ...record,
              adjustment_evaluated: true,
              adjusted_eval_path: projectRelative(adjustedEval),
            }
          : record,
      )
      await writePhaseState(
        {
          ...state,
          adjustment: {
            ...state.adjustment,
            test_application_records: records,
          },
        },
        `## Test Adjustment Evaluated\n\nAdjusted test metrics recorded for ${params.run_id} / ${params.candidate_id}.`,
      )
    }

    return {
      title: `Forecast adjustment eval: ${params.run_id.slice(0, 12)}`,
      metadata: {
        run_id: params.run_id,
        candidate_id: params.candidate_id,
        split: params.split,
        accepted: parsed.acceptance_decision?.accepted,
      },
      output: JSON.stringify(parsed),
    }
  },
})
