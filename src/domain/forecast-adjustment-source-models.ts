import { readdir, readFile } from "node:fs/promises"
import path from "node:path"
import z from "zod"
import { Filesystem } from "./runtime"
import { Tool } from "./runtime"
import { normalizeValidationExecutionContract } from "./execution-contract"
import DESCRIPTION from "./forecast-adjustment-source-models.txt"
import { deduplicateExperimentsByRunId } from "./forecast-history"
import { AdjustmentSourceModelSet, EvalResult, ExperimentRecord, FIXED_ADJUSTMENT_FIT_HOLDOUT_RATIO } from "./types"
import { computeSpecHash } from "./hash"
import { readBoundModelArtifactIdentity } from "./model-artifact"
import { inferForecastModelFamily, statisticalModelArtifact } from "./model-family"
import {
  adjustmentRoot,
  ensureAdjustmentDirs,
  forecastPath,
  forecastRoot,
  readJsonFile,
  readPhaseState,
  runIdIsSafe,
  writeJsonFile,
  writePhaseState,
} from "./forecast-adjustment-common"

const parameters = z.object({
  subcommand: z.enum(["freeze", "read"]),
})

const FOUNDATION_MODEL_NAMES = new Set(
  ["Chronos", "Chronos2", "TimesFM", "Moirai", "Sundial", "TiREx", "TimeMoE"].map((name) =>
    name.toLocaleLowerCase("en-US"),
  ),
)

function sourceSetPath(): string {
  return path.join(adjustmentRoot(), "source-model-set.json")
}

async function readHistory(): Promise<Array<z.infer<typeof ExperimentRecord>>> {
  const historyPath = path.join(forecastRoot(), "history.jsonl")
  const raw = await readFile(historyPath, "utf-8").catch(() => "")
  const records: Array<z.infer<typeof ExperimentRecord>> = []
  for (const line of raw.split("\n").filter(Boolean)) {
    try {
      const parsed = ExperimentRecord.safeParse(JSON.parse(line))
      if (parsed.success) records.push(parsed.data)
    } catch {
      // Skip corrupt lines so one partial write does not block source freezing.
    }
  }
  return deduplicateExperimentsByRunId(records)
}

function isSelectableHistoryRecord(record: z.infer<typeof ExperimentRecord>): boolean {
  if (record.status !== "success" || record.metrics?.mse == null) return false
  if (record.decision === "discard") return false
  try {
    normalizeValidationExecutionContract(record.spec)
    return true
  } catch {
    return false
  }
}

type CompleteValidationArtifact = {
  run_id: string
  spec_hash: string
  metrics: z.infer<typeof EvalResult>["metrics"]
  artifacts: {
    spec_path: string
    eval_path: string
    predictions_path: string
  }
}

async function readCompleteValidationArtifact(
  runId: string,
  expectedSpecHash: string,
): Promise<CompleteValidationArtifact | null> {
  if (!runIdIsSafe(runId)) return null

  const runDir = path.join(forecastRoot(), "runs", runId)
  const specPath = path.join(runDir, "spec.json")
  const evalPath = path.join(runDir, "eval.json")
  const predictionsPath = path.join(runDir, "predictions.csv")

  if (
    !(await Filesystem.exists(specPath)) ||
    !(await Filesystem.exists(evalPath)) ||
    !(await Filesystem.exists(predictionsPath))
  ) {
    return null
  }

  const spec = await readJsonFile<Record<string, unknown>>(specPath).catch(() => null)
  if (!spec) return null
  try {
    normalizeValidationExecutionContract(spec)
  } catch {
    return null
  }
  const holdoutRatio = Number(spec.adjustment_fit_holdout_ratio)
  if (!Number.isFinite(holdoutRatio) || Math.abs(holdoutRatio - FIXED_ADJUSTMENT_FIT_HOLDOUT_RATIO) > 1e-12) {
    return null
  }

  const modelName = String(spec.model ?? "").toLocaleLowerCase("en-US")
  const statisticalArtifact = statisticalModelArtifact(String(spec.model ?? ""))
  const reusableArtifactPath =
    statisticalArtifact
      ? path.join(runDir, "model", statisticalArtifact.filename)
      : ["naive","mean","seasonalnaive","linear","userprior"].includes(modelName) ? path.join(runDir,"model/model.json")
      : FOUNDATION_MODEL_NAMES.has(modelName)
        ? null
        : path.join(runDir, "model", "checkpoint.pth")
  if (reusableArtifactPath && !(await Filesystem.exists(reusableArtifactPath))) return null

  const specHash = computeSpecHash(spec)
  if (specHash !== expectedSpecHash) return null

  const parsedEval = EvalResult.safeParse(await readJsonFile(evalPath).catch(() => null))
  if (!parsedEval.success) return null
  if (parsedEval.data.eval_split !== "val") return null
  // Legacy Python runs may use a different Unicode JSON escaping dialect for
  // their directory/eval hash. Accept them only when the stored spec is
  // semantically identical to the history spec and eval identity is internally
  // consistent with the artifact directory.
  if (parsedEval.data.run_id !== runId || parsedEval.data.spec_hash !== runId) return null
  try {
    await readBoundModelArtifactIdentity(runDir, String(spec.model ?? ""), parsedEval.data)
  } catch {
    return null
  }

  return {
    run_id: runId,
    spec_hash: expectedSpecHash,
    metrics: parsedEval.data.metrics,
    artifacts: {
      spec_path: forecastPath("runs", runId, "spec.json"),
      eval_path: forecastPath("runs", runId, "eval.json"),
      predictions_path: forecastPath("runs", runId, "predictions.csv"),
    },
  }
}

async function resolveCompleteValidationArtifact(
  record: z.infer<typeof ExperimentRecord>,
): Promise<CompleteValidationArtifact | null> {
  const expectedSpecHash = computeSpecHash(record.spec)

  const direct = await readCompleteValidationArtifact(record.run_id, expectedSpecHash)
  if (direct) return direct

  const hashed = await readCompleteValidationArtifact(expectedSpecHash, expectedSpecHash)
  if (hashed) return hashed

  const entries = await readdir(path.join(forecastRoot(), "runs"), { withFileTypes: true }).catch(() => [])
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    if (!entry.isDirectory() || entry.name === record.run_id || entry.name === expectedSpecHash) continue
    const resolved = await readCompleteValidationArtifact(entry.name, expectedSpecHash)
    if (resolved) return resolved
  }

  return null
}

async function selectPerFamilyBest(records: Array<z.infer<typeof ExperimentRecord>>) {
  const best = new Map<
    string,
    {
      record: z.infer<typeof ExperimentRecord>
      artifact: CompleteValidationArtifact
    }
  >()
  const skipped: Array<{ run_id: string; model: string; reason: string }> = []

  for (const record of records) {
    if (!isSelectableHistoryRecord(record)) continue

    const artifact = await resolveCompleteValidationArtifact(record)
    if (!artifact) {
      skipped.push({
        run_id: record.run_id,
        model: record.model,
        reason:
          "missing validation run with the fixed validation-front evidence contract, complete predictions, and a reusable model artifact",
      })
      continue
    }

    const family = record.window_features?.model_family ?? inferForecastModelFamily(record.model) ?? "unclassified"
    const current = best.get(family)
    if (!current || artifact.metrics.mse < current.artifact.metrics.mse) {
      best.set(family, { record, artifact })
    }
  }
  return { selected: [...best.entries()], skipped }
}

export const ForecastAdjustmentSourceModelsTool = Tool.define("forecast_adjustment_source_models", {
  description: DESCRIPTION,
  parameters,
  async execute(params) {
    await ensureAdjustmentDirs()
    let result: unknown

    switch (params.subcommand) {
      case "read": {
        result = await readJsonFile(sourceSetPath()).catch(() => null)
        break
      }

      case "freeze": {
        const { selected, skipped } = await selectPerFamilyBest(await readHistory())
        if (selected.length === 0) {
          throw new Error(
            "No reusable successful forecast-phase validation experiments were found. Each source needs the fixed validation-front evidence contract, complete prediction artifacts, and a reusable checkpoint, fitted statistical result, or foundation reference.",
          )
        }

        const createdAt = new Date().toISOString()
        const members = selected.map(([family, { record, artifact }]) => ({
          family,
          validation_run_id: artifact.run_id,
          ...(artifact.run_id !== record.run_id ? { history_run_id: record.run_id } : {}),
          model: record.model,
          spec_hash: artifact.spec_hash,
          validation_metrics: {
            mse: artifact.metrics.mse,
            mae: artifact.metrics.mae,
            wape: artifact.metrics.wape,
            mase: artifact.metrics.mase,
          },
          artifacts: artifact.artifacts,
        }))
        const sourceSetId = computeSpecHash({ created_from: "post_raw_forecast_search", members })
        const sourceSet = AdjustmentSourceModelSet.parse({
          source_model_set_id: sourceSetId,
          created_at: createdAt,
          created_from: "post_raw_forecast_search",
          members,
        })
        await writeJsonFile(sourceSetPath(), sourceSet)

        const state = await readPhaseState()
        const updated = {
          ...state,
          adjustment: {
            ...state.adjustment,
            source_model_set_locked: true,
            source_model_set_path: forecastPath("adjustments", "source-model-set.json"),
            fit_run_manifest_path: null,
            fit_runs_ready: false,
            active_skill_version: null,
            active_skill_path: null,
            forecaster_locked: false,
            no_effective_skill: false,
            no_effective_skill_reason: null,
            test_collection_run_ids: [],
            test_eligible_run_ids: [],
            test_application_records: [],
            critic_user_confirmed: null,
            test_application_decision: "pending" as const,
            test_application_decision_at: null,
            test_application_decision_actor: null,
            test_application_decision_reason: null,
            test_application_run_id: null,
            test_adjustment_locked: false,
            test_adjustment_run_id: null,
            test_adjustment_candidate_id: null,
          },
        }
        await writePhaseState(
          updated,
          [
            "## Adjustment Source Model Set Frozen",
            "",
            `Frozen ${members.length} artifact-complete source model(s) at ${createdAt}.`,
            ...(skipped.length > 0
              ? ["", `Skipped ${skipped.length} history record(s) without complete validation prediction artifacts.`]
              : []),
          ].join("\n"),
        )
        result = { frozen: true, source_model_set: sourceSet, skipped }
        break
      }
    }

    return {
      title: `Forecast adjustment source models: ${params.subcommand}`,
      metadata: { subcommand: params.subcommand },
      output: JSON.stringify(result),
    }
  },
})
