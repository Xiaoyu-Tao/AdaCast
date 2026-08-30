import { appendFile, mkdir } from "fs/promises"
import path from "path"
import z from "zod"
import { Instance } from "@/project/instance"
import { Tool } from "../tool"
import DESCRIPTION from "./forecast-reflect.txt"
import { ReflectionRecord } from "./types"

type ForecastReflectMetadata = {
  run_id: string
  decision: "keep" | "discard" | "crash" | "stop"
}

const parameters = z.object({
  run_id: z.string().describe("Spec hash of the experiment this reflection evaluates"),
  effective: z.boolean().describe("Did the last experiment improve on the current best?"),
  reason: z.string().describe("Explanation of the outcome — what worked, what failed, and why"),
  next_direction: z.string().describe("What to try next and why, based on what was learned"),
  decision: z
    .enum(["keep", "discard", "crash", "stop"])
    .describe(
      "Outcome action: keep (best updated), discard (no improvement), crash (runner failure), stop (terminate experiment loop)",
    ),
})

export const ForecastReflectTool = Tool.define<typeof parameters, ForecastReflectMetadata>("forecast_reflect", {
  description: DESCRIPTION,
  parameters,
  async execute(params) {
    const directory = Instance.directory
    const forecastDir = path.join(directory, ".forecast")
    await mkdir(forecastDir, { recursive: true })

    const record = ReflectionRecord.parse({
      type: "reflection",
      run_id: params.run_id,
      effective: params.effective,
      reason: params.reason,
      next_direction: params.next_direction,
      decision: params.decision,
      timestamp: new Date().toISOString(),
    })

    const historyPath = path.join(forecastDir, "history.jsonl")
    await appendFile(historyPath, `${JSON.stringify(record)}\n`, "utf-8")

    return {
      title: `Reflection: ${params.decision} (${params.effective ? "improved" : "no improvement"})`,
      metadata: { run_id: params.run_id, decision: params.decision },
      output: JSON.stringify({
        recorded: true,
        run_id: params.run_id,
        decision: params.decision,
        effective: params.effective,
      }),
    }
  },
})
