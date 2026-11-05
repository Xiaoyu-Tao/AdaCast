export type ForecastModelFamily = "statistical" | "deep_learning" | "foundation" | "baseline" | "user_prior"

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
  if (normalized === "userprior") return "user_prior"
  if (["naive","mean","seasonalnaive","linear"].includes(normalized)) return "baseline"
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


// Independent registry, checked against the numerical engine by contract tests.
export const FORECAST_MODEL_NAMES = ["ARIMA", "AutoARIMA", "Autoformer", "Chronos", "Chronos2", "Crossformer", "DLinear", "ETS", "ETSformer", "ExponentialSmoothing", "FEDformer", "FiLM", "FreTS", "Holt", "HoltWinters", "Informer", "KANAD", "Koopa", "LightTS", "Linear", "MICN", "MSGNet", "Mamba", "MambaSimple", "MambaSingleLayer", "Mean", "Moirai", "MultiPatchFormer", "Naive", "Nonstationary_Transformer", "PAttn", "PatchTST", "Pyraformer", "Reformer", "SCINet", "SeasonalNaive", "SegRNN", "SimpleExponentialSmoothing", "Sundial", "TSMixer", "TemporalFusionTransformer", "Theta", "TiDE", "TiRex", "TimeFilter", "TimeMixer", "TimeMoE", "TimeXer", "TimesFM", "TimesNet", "Transformer", "UserPrior", "WPMixer", "iTransformer"] as const
export function canonicalForecastModelName(model:string):string {
  const key=model.toLocaleLowerCase("en-US").replace(/[^a-z0-9]/g,"")
  const name=FORECAST_MODEL_NAMES.find(n=>n.toLocaleLowerCase("en-US").replace(/[^a-z0-9]/g,"")===key)
  if(!name)throw new Error("Unknown forecasting model: "+model)
  return name
}
