import { deepSortKeys } from "./hash"

export type ForecastExecutionContract = Record<string, unknown> & {
  phase: "forecast" | "post-forecast"
  eval_split: "val" | "adjustment_fit" | "test"
}

export function normalizeForecastExecutionContract(spec: Record<string, unknown>): ForecastExecutionContract {
  const phase = spec.phase ?? "forecast"
  const evalSplit = spec.eval_split ?? "val"
  const valid =
    (phase === "forecast" && (evalSplit === "val" || evalSplit === "adjustment_fit")) ||
    (phase === "post-forecast" && evalSplit === "test")

  if (!valid) {
    throw new Error(
      "Invalid forecast execution contract: expected phase='forecast' with eval_split='val' or " +
        "'adjustment_fit', or phase='post-forecast' with eval_split='test'",
    )
  }
  if ("window_stride" in spec) {
    const windowStride = spec.window_stride
    if (typeof windowStride !== "number" || !Number.isInteger(windowStride) || windowStride <= 0) {
      throw new Error("Invalid forecast execution contract: window_stride must be a positive integer")
    }
  }
  if ("forecast_gap" in spec) {
    const forecastGap = spec.forecast_gap
    if (typeof forecastGap !== "number" || !Number.isInteger(forecastGap) || forecastGap < 0) {
      throw new Error("Invalid forecast execution contract: forecast_gap must be a nonnegative integer")
    }
  }

  return {
    ...spec,
    phase,
    eval_split: evalSplit,
  } as ForecastExecutionContract
}

export function normalizeValidationExecutionContract(
  spec: Record<string, unknown>,
): ForecastExecutionContract & { phase: "forecast"; eval_split: "val" } {
  const normalized = normalizeForecastExecutionContract(spec)
  if (normalized.phase !== "forecast" || normalized.eval_split !== "val") {
    throw new Error("Expected a forecast-phase validation execution contract")
  }
  return normalized as ForecastExecutionContract & { phase: "forecast"; eval_split: "val" }
}

export function normalizeGenerateModelExecutionContract(spec: Record<string, unknown>): ForecastExecutionContract {
  try {
    const normalized = normalizeForecastExecutionContract(spec)
    if (normalized.eval_split !== "adjustment_fit") return normalized
  } catch {
    // Keep generate_model's established public error contract below.
  }

  throw new Error(
    "generate_model only supports phase='forecast' with eval_split='val', or " +
      "phase='post-forecast' with eval_split='test'",
  )
}

export function validationSpecMatchesPostForecastTestSpec(
  validationSpec: Record<string, unknown>,
  testSpec: Record<string, unknown>,
): boolean {
  return validationPostForecastTestSpecDifferences(validationSpec, testSpec)?.length === 0
}

export function validationPostForecastTestSpecDifferences(
  validationSpec: Record<string, unknown>,
  testSpec: Record<string, unknown>,
): string[] | null {
  try {
    const normalizedValidationSpec = normalizeValidationExecutionContract(validationSpec)
    const normalizedTestSpec = normalizeForecastExecutionContract(testSpec)
    if (normalizedTestSpec.phase !== "post-forecast" || normalizedTestSpec.eval_split !== "test") return null

    const comparableValidationSpec: Record<string, unknown> = {
      ...normalizedValidationSpec,
      window_stride: normalizedValidationSpec.window_stride ?? 1,
      forecast_gap: normalizedValidationSpec.forecast_gap ?? 0,
      phase: "post-forecast",
      eval_split: "test",
    }
    const comparableTestSpec: Record<string, unknown> = {
      ...normalizedTestSpec,
      window_stride: normalizedTestSpec.window_stride ?? 1,
      forecast_gap: normalizedTestSpec.forecast_gap ?? 0,
    }
    // This anchor locates the first target inside each evaluation split. It must
    // change between validation and test and is not part of the frozen model contract.
    delete comparableValidationSpec.prediction_grid_target_anchor_index
    delete comparableTestSpec.prediction_grid_target_anchor_index
    const keys = [...new Set([...Object.keys(comparableValidationSpec), ...Object.keys(comparableTestSpec)])].sort()
    return keys.filter(
      (key) =>
        JSON.stringify(deepSortKeys(comparableValidationSpec[key])) !==
        JSON.stringify(deepSortKeys(comparableTestSpec[key])),
    )
  } catch {
    return null
  }
}
