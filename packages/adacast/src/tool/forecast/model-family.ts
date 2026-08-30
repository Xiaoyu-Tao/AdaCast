export type ForecastModelFamily = "statistical" | "deep_learning" | "foundation"

const STATISTICAL_MODELS = new Map([
  ["arima", "ARIMA"],
  ["autoarima", "AutoARIMA"],
  ["ets", "ETS"],
  ["exponentialsmoothing", "ExponentialSmoothing"],
  ["simpleexponentialsmoothing", "SimpleExponentialSmoothing"],
  ["holt", "Holt"],
  ["holtwinters", "HoltWinters"],
  ["theta", "Theta"],
])

const FOUNDATION_MODEL_REFERENCES = new Map([
  ["chronos", "amazon/chronos-bolt-base"],
  ["chronos2", "amazon/chronos-2"],
  ["timesfm", "google/timesfm-2.5-200m-pytorch"],
  ["moirai", "Salesforce/moirai-2.0-R-small"],
  ["sundial", "thuml/sundial-base-128m"],
  ["tirex", "NX-AI/TiRex"],
  ["timemoe", "Maple728/TimeMoE-50M"],
])

export function inferForecastModelFamily(model: string): ForecastModelFamily | undefined {
  const normalized = model
    .trim()
    .toLocaleLowerCase("en-US")
    .replace(/[\s_-]+/g, "")
  if (!normalized) return undefined
  if (STATISTICAL_MODELS.has(normalized)) return "statistical"
  if (FOUNDATION_MODEL_REFERENCES.has(normalized)) return "foundation"
  return "deep_learning"
}

export function statisticalModelName(model: string): string | undefined {
  const normalized = model
    .trim()
    .toLocaleLowerCase("en-US")
    .replace(/[\s_-]+/g, "")
  return STATISTICAL_MODELS.get(normalized)
}

export function isStatisticalForecastModel(model: string): boolean {
  return statisticalModelName(model) !== undefined
}

export function statisticalModelArtifact(model: string):
  | {
      canonicalName: string
      filename: "arima-results.pkl" | "statistical-results.pkl"
      kind: "statsmodels_arima_results" | "statsmodels_statistical_results"
      metadataKind: "arima_results" | "statistical_results"
      format: "statsmodels_arima_results" | "statsmodels_results_pickle"
    }
  | undefined {
  const canonicalName = statisticalModelName(model)
  if (!canonicalName) return undefined
  const isArima = canonicalName === "ARIMA"
  return {
    canonicalName,
    filename: isArima ? "arima-results.pkl" : "statistical-results.pkl",
    kind: isArima ? "statsmodels_arima_results" : "statsmodels_statistical_results",
    metadataKind: isArima ? "arima_results" : "statistical_results",
    format: isArima ? "statsmodels_arima_results" : "statsmodels_results_pickle",
  }
}

export function foundationModelReference(model: string): string | undefined {
  const normalized = model
    .trim()
    .toLocaleLowerCase("en-US")
    .replace(/[\s_-]+/g, "")
  return FOUNDATION_MODEL_REFERENCES.get(normalized)
}
