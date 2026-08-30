import { describe, expect, test } from "bun:test"
import {
  inferForecastModelFamily,
  isStatisticalForecastModel,
  statisticalModelArtifact,
  statisticalModelName,
} from "./model-family"

describe("statistical forecast model registry", () => {
  test("normalizes every supported statsmodels-backed name", () => {
    expect(statisticalModelName("auto_arima")).toBe("AutoARIMA")
    expect(statisticalModelName("simple-exponential-smoothing")).toBe("SimpleExponentialSmoothing")
    expect(statisticalModelName("HOLT WINTERS")).toBe("HoltWinters")
    expect(isStatisticalForecastModel("Theta")).toBe(true)
    expect(inferForecastModelFamily("ETS")).toBe("statistical")
  })

  test("does not advertise unsupported external statistical packages", () => {
    expect(isStatisticalForecastModel("SARIMA")).toBe(false)
    expect(statisticalModelArtifact("SARIMA")).toBeUndefined()
    expect(isStatisticalForecastModel("Prophet")).toBe(false)
    expect(isStatisticalForecastModel("TBATS")).toBe(false)
  })

  test("uses the legacy ARIMA artifact and a generic artifact for added models", () => {
    expect(statisticalModelArtifact("ARIMA")).toMatchObject({
      filename: "arima-results.pkl",
      kind: "statsmodels_arima_results",
    })
    expect(statisticalModelArtifact("ETS")).toMatchObject({
      filename: "statistical-results.pkl",
      kind: "statsmodels_statistical_results",
      format: "statsmodels_results_pickle",
    })
  })
})
