import path from "path"
import z from "zod"
import { Instance } from "@/project/instance"
import { Filesystem } from "@/util/filesystem"
import { Tool } from "../tool"
import { IPCError } from "./errors"
import DESCRIPTION from "./evaluate-experiment.txt"
import { readBoundModelArtifactIdentity } from "./model-artifact"
import { EvalResult } from "./types"

type EvalResultValue = z.infer<typeof EvalResult>
type EvaluateExperimentMetadata = {
  run_id: string
  status: "success" | "ipc_error"
  improved?: boolean
}

const parameters = z.object({
  run_id: z.string().describe("The run_id (spec hash) of the experiment to evaluate"),
})

function finiteMetric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

export const EvaluateExperimentTool = Tool.define<typeof parameters, EvaluateExperimentMetadata>(
  "evaluate_experiment",
  {
    description: DESCRIPTION,
    parameters,
    async execute(params) {
      const runDir = path.join(Instance.directory, ".forecast", "runs", params.run_id)
      const evalPath = path.join(runDir, "eval.json")

      let evalResult: EvalResultValue
      try {
        evalResult = EvalResult.parse(await Filesystem.readJson(evalPath))
        const spec = await Filesystem.readJson<Record<string, unknown>>(path.join(runDir, "spec.json"))
        await readBoundModelArtifactIdentity(runDir, String(spec.model ?? ""), evalResult)
      } catch {
        const ipcError = new IPCError({
          reason: "eval.json missing, corrupt, or not bound to current run artifacts",
        })
        return {
          title: `Eval error: ${params.run_id.slice(0, 12)}`,
          metadata: { run_id: params.run_id, status: "ipc_error" },
          output: JSON.stringify({
            status: "ipc_error",
            run_id: params.run_id,
            reason: ipcError.data.reason,
          }),
        }
      }

      const bestPath = path.join(Instance.directory, ".forecast", "best.json")
      let bestMetrics: { mse: number; mae: number } | null = null
      try {
        const best = await Filesystem.readJson(bestPath)
        const mse = finiteMetric(best?.metrics?.mse)
        const mae = finiteMetric(best?.metrics?.mae)
        if (mse != null && mae != null) {
          bestMetrics = { mse, mae }
        }
      } catch {
        bestMetrics = null
      }

      const metrics = evalResult.metrics
      const improved = bestMetrics ? metrics.mse < bestMetrics.mse : true
      const delta = bestMetrics
        ? {
            mse_delta: metrics.mse - bestMetrics.mse,
            mae_delta: metrics.mae - bestMetrics.mae,
            mse_improvement_pct:
              bestMetrics.mse > 0
                ? `${(((bestMetrics.mse - metrics.mse) / bestMetrics.mse) * 100).toFixed(2)}%`
                : "N/A",
          }
        : null

      const summary = {
        status: "success" as const,
        run_id: params.run_id,
        metrics: {
          mse: metrics.mse,
          mae: metrics.mae,
          wape: metrics.wape,
          mase: metrics.mase,
        },
        comparison: {
          improved,
          delta,
          has_baseline: bestMetrics !== null,
        },
        eval_split: evalResult.eval_split,
        evaluator_hash: evalResult.evaluator_hash,
      }

      return {
        title: `Eval: ${params.run_id.slice(0, 12)}`,
        metadata: { run_id: params.run_id, status: "success", improved },
        output: JSON.stringify(summary),
      }
    },
  },
)
