import { open, readFile, unlink, writeFile } from "fs/promises"
import path from "path"
import z from "zod"
import { Instance } from "@/project/instance"
import { Filesystem } from "@/util/filesystem"
import { Tool } from "../tool"
import DESCRIPTION from "./forecast-task.txt"
import { TaskFrozenError } from "./errors"
import { freezeForecastSplit } from "./split-contract"
import { TaskConfig } from "./types"
import { readPhaseState, writePhaseState } from "./forecast-adjustment-common"
import { writeFrozenTaskIntegrityLock } from "./forecast-task-integrity"

const parameters = z
  .object({
    dataset_path: z.string().describe("Absolute path to the CSV dataset file"),
    target_col: z.string().describe("Name of the column to forecast"),
    time_col: z.string().describe("Name of the timestamp column"),
    pred_len: z.number().int().positive().describe("Prediction horizon in time steps"),
    forecast_gap: z
      .number()
      .int()
      .nonnegative()
      .optional()
      .describe("Future steps skipped between the history window and evaluated horizon; defaults to 0"),
    seq_len: z.number().int().positive().describe("Input sequence length in time steps"),
    label_len: z.number().int().nonnegative().describe("Label sequence length for decoder models"),
    window_stride: z
      .number()
      .int()
      .positive()
      .optional()
      .describe("Unified sliding-window step for training, validation, and testing; defaults to 1"),
    stride: z
      .number()
      .int()
      .positive()
      .optional()
      .describe("Alias of task-level window_stride when the user explicitly says stride"),
    split_mode: z
      .enum(["ratio", "timestamp"])
      .optional()
      .describe("Use ratio fields or two inclusive forecast-origin timestamp boundaries; inferred when omitted"),
    train_ratio: z.number().min(0).max(1).optional().describe("Fraction of data for training (e.g. 0.7)"),
    val_ratio: z.number().min(0).max(1).optional().describe("Fraction of data for validation (e.g. 0.2)"),
    test_ratio: z.number().min(0).max(1).optional().describe("Fraction of data for testing (e.g. 0.1)"),
    train_end_timestamp: z
      .string()
      .min(1)
      .optional()
      .describe(
        "Inclusive final training history/origin timestamp; use with validation_end_timestamp instead of ratios",
      ),
    validation_end_timestamp: z
      .string()
      .min(1)
      .optional()
      .describe(
        "Inclusive final validation history/origin timestamp; later evaluation targets start after forecast_gap",
      ),
    eval_metrics: z
      .array(z.enum(["mse", "mae", "wape", "mase"]))
      .min(1)
      .describe("Metrics to compute"),
    model_families: z
      .array(z.string())
      .min(1)
      .describe("Model families to include in experiments, e.g. ['DLinear', 'PatchTST', 'Chronos']"),
    business_context: z.string().optional().describe("Business background for the forecast"),
    csv_semantics: z.string().optional().describe("Description of CSV column meanings"),
    region_info: z.string().optional().describe("Geographic region information"),
    seasonality_hints: z.string().optional().describe("Known seasonality patterns"),
    holiday_effects: z.string().optional().describe("Holiday or event effects to consider"),
    anomaly_periods: z.string().optional().describe("Known anomaly periods in the data"),
    adjustment_skill_priors: z
      .string()
      .optional()
      .describe(
        "User-supplied or synthesized expert adjustment rules, including unspecified numeric parameters and explicitly referenced local text/code paths, to mirror independently as validation-gated candidates",
      ),
    reflection_enabled: z.boolean().optional().describe("Independent final window reflection switch; defaults to true"),
  })
  .superRefine((params, ctx) => {
    if (params.window_stride !== undefined && params.stride !== undefined && params.window_stride !== params.stride) {
      ctx.addIssue({
        code: "custom",
        path: ["stride"],
        message: "stride and window_stride must match when both are provided",
      })
    }
  })

type ForecastTaskParameters = z.infer<typeof parameters>

const TASK_FREEZE_PENDING = "task-freeze.pending.json"

const TaskFreezePending = z.object({
  version: z.literal(1),
  state_content: z.string().nullable(),
})

async function rollbackPendingTaskFreeze(forecastDirectory: string): Promise<void> {
  const pendingPath = path.join(forecastDirectory, TASK_FREEZE_PENDING)
  if (!(await Filesystem.exists(pendingPath))) {
    const taskPath = path.join(forecastDirectory, "task.json")
    const lockPath = path.join(forecastDirectory, "task-lock.json")
    if (
      (await Filesystem.exists(taskPath)) &&
      !(await Filesystem.exists(lockPath)) &&
      (await Filesystem.size(taskPath)) === 0
    ) {
      await unlink(taskPath).catch(() => {})
    }
    return
  }

  const pendingRaw = await readFile(pendingPath, "utf-8").catch(() => "null")
  const pending = TaskFreezePending.safeParse(
    (() => {
      try {
        return JSON.parse(pendingRaw)
      } catch {
        return null
      }
    })(),
  )
  await Promise.all([
    unlink(path.join(forecastDirectory, "task.json")).catch(() => {}),
    unlink(path.join(forecastDirectory, "task-lock.json")).catch(() => {}),
  ])
  if (pending.success) {
    const statePath = path.join(forecastDirectory, "STATE.md")
    if (pending.data.state_content === null) await unlink(statePath).catch(() => {})
    else await writeFile(statePath, pending.data.state_content, "utf-8")
  }
  await unlink(pendingPath).catch(() => {})
}

const TIMESTAMP_PATTERN = /\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]\d{1,2}[:：]\d{1,2}(?:[:：]\d{1,2})?)?/g
const TASK_TEXT_FIELDS = [
  "business_context",
  "csv_semantics",
  "region_info",
  "seasonality_hints",
  "holiday_effects",
  "anomaly_periods",
  "adjustment_skill_priors",
] as const satisfies readonly (keyof ForecastTaskParameters)[]

function normalizePromptText(text: string): string {
  return text
    .replace(/\u00a0/g, " ")
    .replace(/\uff1a/g, ":")
    .replace(/\uff1d/g, "=")
    .replace(/\uff0c/g, ",")
    .replace(/\uff1b/g, ";")
    .replace(/[\u2013\u2014]/g, "-")
    .replace(/[ \t]+/g, " ")
}

function collectRecentUserText(ctx: Tool.Context): string {
  const chunks: string[] = []
  for (const message of ctx.messages ?? []) {
    if (message.info.role !== "user") continue
    for (const part of message.parts) {
      if (part.type === "text" && !part.ignored) chunks.push(part.text)
    }
  }
  return normalizePromptText(chunks.join("\n"))
}

function collectTaskParameterText(params: ForecastTaskParameters): string {
  const chunks: string[] = []
  for (const field of TASK_TEXT_FIELDS) {
    const value = params[field]
    if (typeof value === "string" && value.trim()) {
      chunks.push(`${field}:\n${value}`)
    }
  }
  return normalizePromptText(chunks.join("\n"))
}

function collectTaskContractEvidenceText(params: ForecastTaskParameters, ctx: Tool.Context): string {
  return [collectRecentUserText(ctx), collectTaskParameterText(params)].filter(Boolean).join("\n")
}

function assertExplicitAdjustmentSkillPriorCaptured(params: ForecastTaskParameters, ctx: Tool.Context): void {
  if (params.adjustment_skill_priors?.trim()) return
  const userText = collectRecentUserText(ctx)
  const explicitPriorMarkers = [
    /生成式预测推理先验\s*skill/i,
    /生成式预测推理.{0,16}专家(?:先验|经验|规则)/i,
    /(?:generative prediction reasoning|forecast adjustment).{0,16}(?:expert|user)[ -]?prior/i,
  ]
  if (explicitPriorMarkers.some((pattern) => pattern.test(userText))) {
    throw new Error(
      "The user Prompt contains an explicit generative prediction reasoning expert prior, but " +
        "adjustment_skill_priors is missing. Capture the expert-rule guidance before freezing the task.",
    )
  }
}

function timestampIdentity(value: string): string {
  const normalized = normalizePromptText(value).replace("T", " ").trim()
  const match = normalized.match(/^(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:\s+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?$/)
  if (!match) return normalized

  const [, year, month, day, hour = "0", minute = "0", second = "0"] = match
  return [
    year.padStart(4, "0"),
    month.padStart(2, "0"),
    day.padStart(2, "0"),
    hour.padStart(2, "0"),
    minute.padStart(2, "0"),
    second.padStart(2, "0"),
  ].join(":")
}

function uniqueTimestampHints(values: string[]): string[] {
  const seen = new Set<string>()
  const unique: string[] = []
  for (const value of values) {
    const key = timestampIdentity(value)
    if (seen.has(key)) continue
    seen.add(key)
    unique.push(value)
  }
  return unique
}

function extractTimestampSplitHint(
  text: string,
): { train_end_timestamp: string; validation_end_timestamp: string } | undefined {
  const segments = text
    .split(/[\n\u3002\uff1b;]/)
    .map((segment) => segment.trim())
    .filter(Boolean)

  let hint: { train_end_timestamp: string; validation_end_timestamp: string } | undefined
  for (const segment of segments) {
    const hasSplitSignal = /(\bsplit\b|\bboundar|\u5207\u5206|\u5212\u5206|\u65f6\u95f4\u6233)/i.test(segment)
    const hasDatasetSignal =
      /(\btrain\b|\bvalidation\b|\bval\b|\btest\b|\u8bad\u7ec3\u96c6|\u9a8c\u8bc1\u96c6|\u6d4b\u8bd5\u96c6|\u65f6\u95f4\u6233)/i.test(
        segment,
      )
    if (!hasSplitSignal || !hasDatasetSignal) continue

    const timestamps = uniqueTimestampHints(
      Array.from(segment.matchAll(TIMESTAMP_PATTERN), (match) =>
        normalizePromptText(match[0]).replace("T", " ").trim(),
      ),
    )
    if (timestamps.length >= 2) {
      hint = {
        train_end_timestamp: timestamps[0],
        validation_end_timestamp: timestamps[1],
      }
    }
  }
  return hint
}

function extractLastIntegerHint(text: string, patterns: RegExp[]): number | undefined {
  let value: number | undefined
  for (const pattern of patterns) {
    for (const match of text.matchAll(pattern)) {
      const parsed = Number(match[1])
      if (Number.isInteger(parsed)) value = parsed
    }
  }
  return value
}

function applyPromptContractFallback(params: ForecastTaskParameters, ctx: Tool.Context) {
  const evidenceText = collectTaskContractEvidenceText(params, ctx)
  if (!evidenceText) return { params, recovered: [] as string[] }

  const recovered: string[] = []
  const next: ForecastTaskParameters = { ...params }
  const timestampHint = extractTimestampSplitHint(evidenceText)
  if (timestampHint) {
    const missingTimestampBoundary =
      next.train_end_timestamp === undefined || next.validation_end_timestamp === undefined
    const mismatchedTimestampBoundary =
      (next.train_end_timestamp !== undefined &&
        timestampIdentity(next.train_end_timestamp) !== timestampIdentity(timestampHint.train_end_timestamp)) ||
      (next.validation_end_timestamp !== undefined &&
        timestampIdentity(next.validation_end_timestamp) !== timestampIdentity(timestampHint.validation_end_timestamp))
    const mixedTimestampRatio =
      next.train_ratio !== undefined ||
      next.val_ratio !== undefined ||
      next.test_ratio !== undefined ||
      next.split_mode === "ratio"

    if (missingTimestampBoundary || mismatchedTimestampBoundary || mixedTimestampRatio) {
      next.split_mode = "timestamp"
      next.train_end_timestamp = timestampHint.train_end_timestamp
      next.validation_end_timestamp = timestampHint.validation_end_timestamp
      next.train_ratio = undefined
      next.val_ratio = undefined
      next.test_ratio = undefined
      recovered.push("timestamp split boundaries")
    }
  }

  const forecastGap = extractLastIntegerHint(evidenceText, [
    /\bforecast[_\s-]*gap\b\s*[=:]\s*(\d+)/gi,
    /\bgap\b\s*[=:]\s*(\d+)/gi,
    /\u9884\u6d4b\u95f4\u9694\s*[=:]\s*(\d+)/g,
  ])
  if (forecastGap !== undefined && next.forecast_gap !== forecastGap) {
    next.forecast_gap = forecastGap
    recovered.push("forecast_gap")
  }

  const windowStride = extractLastIntegerHint(evidenceText, [
    /\bwindow[_\s-]*stride\b\s*[=:]\s*(\d+)/gi,
    /\bstride\b\s*[=:]\s*(\d+)/gi,
    /\u6b65\u957f\s*[=:]\s*(\d+)/g,
  ])
  if (windowStride !== undefined && next.window_stride !== windowStride) {
    next.window_stride = windowStride
    next.stride = undefined
    recovered.push("window_stride")
  }

  return { params: next, recovered }
}

function formatValidationError(error: z.ZodError) {
  const issues = error.issues.map((issue) => `  - ${issue.path.join(".")}: ${issue.message}`).join("\n")
  return `forecast_task validation failed. Fix these fields and try again:\n${issues}`
}

export const ForecastTaskTool = Tool.define("forecast_task", {
  description: DESCRIPTION,
  parameters,
  formatValidationError,
  async execute(params, ctx) {
    const parsedParams = parameters.parse(params)
    const fallback = applyPromptContractFallback(parsedParams, ctx)
    const sanitizedParams = parameters.parse(fallback.params)
    assertExplicitAdjustmentSkillPriorCaptured(sanitizedParams, ctx)
    const forecastDirectory = path.join(Instance.directory, ".forecast")
    const taskPath = path.join(forecastDirectory, "task.json")

    await Filesystem.write(path.join(forecastDirectory, ".keep"), "")
    await rollbackPendingTaskFreeze(forecastDirectory)

    const frozenSplit = await freezeForecastSplit(
      sanitizedParams,
      sanitizedParams.dataset_path,
      sanitizedParams.time_col,
      Instance.directory,
    )
    const { stride, ...taskParams } = sanitizedParams
    const config = TaskConfig.parse({
      ...taskParams,
      ...frozenSplit,
      window_stride: sanitizedParams.window_stride ?? stride,
      reflection_enabled: sanitizedParams.reflection_enabled ?? true,
      frozen_at: new Date().toISOString(),
    })

    const fileHandle = await open(taskPath, "wx").catch((error: unknown) => {
      if (typeof error === "object" && error !== null && "code" in error && error.code === "EEXIST") {
        throw new TaskFrozenError({ path: taskPath })
      }
      throw error
    })

    const statePath = path.join(forecastDirectory, "STATE.md")
    const stateContent = await readFile(statePath, "utf-8").catch(() => null)
    const pendingPath = path.join(forecastDirectory, TASK_FREEZE_PENDING)
    try {
      await Filesystem.writeJson(pendingPath, { version: 1, state_content: stateContent })
      await fileHandle.writeFile(JSON.stringify(config, null, 2))
      await fileHandle.close()
      await writeFrozenTaskIntegrityLock(taskPath, config.frozen_at)

      if (await Filesystem.exists(statePath)) {
        const state = await readPhaseState()
        await writePhaseState(
          {
            ...state,
            reflection: {
              ...state.reflection,
              enabled: config.reflection_enabled,
            },
          },
          `## Window Reflection Switch\n\nIndependent window reflection was ${config.reflection_enabled ? "enabled" : "disabled"} in the frozen task.`,
        )
      }
      await unlink(pendingPath)
    } catch (error) {
      await fileHandle.close().catch(() => {})
      await rollbackPendingTaskFreeze(forecastDirectory)
      throw error
    }

    const parameterKeys = Object.keys(sanitizedParams) as Array<keyof typeof sanitizedParams>
    const setParameterCount = parameterKeys.filter((key) => sanitizedParams[key] !== undefined).length

    return {
      title: "Forecast task defined",
      metadata: { taskPath },
      output: [
        `Task frozen at ${taskPath}. Parameters locked for this forecasting session. ${parameterKeys.length} parameters recorded (${setParameterCount} set).`,
        ...(fallback.recovered.length > 0
          ? [`Recovered task contract fields from the current task text: ${fallback.recovered.join(", ")}.`]
          : []),
        "Required next action: call forecast_prediction_prior with subcommand ask_start to show the user prediction-prior window before entering pre-forecast.",
      ].join("\n"),
    }
  },
})
