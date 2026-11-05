import path from "node:path"
import z from "zod"
import { Filesystem } from "./runtime"
import { Tool } from "./runtime"
import DESCRIPTION from "./forecast-adjustment-diagnostics.txt"
import { AdjustmentFitRunManifest } from "./types"
import {
  adjustmentRoot,
  forecastRoot,
  projectRelative,
  readJsonFile,
  readPhaseState,
  runAdjustmentPython,
  runIdIsSafe,
} from "./forecast-adjustment-common"

const parameters = z.object({
  run_id: z.string().describe("Completed fit-run id to summarize"),
  split: z.literal("fit").default("fit").describe("Only train-internal fit diagnostics are supported"),
  top_k: z.number().int().positive().max(100).default(20).describe("Maximum high-signal bucket rows to return"),
})

async function assertFitDiagnosticsAllowed(runId: string): Promise<void> {
  const state = await readPhaseState()
  if (state.phase !== "forecast") {
    throw new Error("Adjustment fit diagnostics are only allowed during the forecast phase")
  }
  if (!state.adjustment.source_model_set_locked || !state.adjustment.fit_runs_ready) {
    throw new Error("Adjustment fit diagnostics require a frozen source set and completed fit runs")
  }

  const fitManifest = AdjustmentFitRunManifest.parse(
    await readJsonFile(path.join(adjustmentRoot(), "fit-runs", "manifest.json")),
  )
  const member = fitManifest.members.find((item) => item.fit_run_id === runId)
  if (!member || (member.status !== "success" && member.status !== "cached") || !member.predictions_path) {
    throw new Error("Adjustment fit diagnostics require a completed fit run with predictions.csv")
  }
}

export const ForecastAdjustmentDiagnosticsTool = Tool.define("forecast_adjustment_diagnostics", {
  description: DESCRIPTION,
  parameters,
  async execute(params) {
    if (!runIdIsSafe(params.run_id)) throw new Error(`Invalid run_id "${params.run_id}"`)
    await assertFitDiagnosticsAllowed(params.run_id)

    const predictionsPath = path.join(adjustmentRoot(), "fit-runs", "runs", params.run_id, "predictions.csv")
    const taskPath = path.join(forecastRoot(), "task.json")
    const out = path.join(adjustmentRoot(), "diagnostics", params.split, params.run_id, "diagnostics.json")
    const missing = []
    for (const file of [predictionsPath, taskPath]) {
      if (!(await Filesystem.exists(file))) missing.push(projectRelative(file))
    }
    if (missing.length > 0) {
      throw new Error(`Required adjustment diagnostics artifact(s) missing: ${missing.join(", ")}`)
    }

    const output = await runAdjustmentPython("adacast_lite_ml.engine.adjustment.diagnostics", [
      "--task-json",
      taskPath,
      "--predictions-csv",
      predictionsPath,
      "--out",
      out,
      "--top-k",
      String(params.top_k),
    ])

    return {
      title: `Forecast adjustment diagnostics: ${params.run_id.slice(0, 12)}`,
      metadata: { run_id: params.run_id, split: params.split, diagnostics_path: projectRelative(out) },
      output: JSON.stringify(output),
    }
  },
})
