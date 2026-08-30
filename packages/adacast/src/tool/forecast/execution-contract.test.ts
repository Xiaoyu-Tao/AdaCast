import { describe, expect, test } from "bun:test"
import {
  normalizeForecastExecutionContract,
  normalizeGenerateModelExecutionContract,
  normalizeValidationExecutionContract,
  validationPostForecastTestSpecDifferences,
  validationSpecMatchesPostForecastTestSpec,
} from "./execution-contract"

describe("forecast execution contract", () => {
  test("normalizes legacy validation specs without mutating the original artifact", () => {
    const legacy = { model: "Chronos", eval_split: "val" }

    const normalized = normalizeValidationExecutionContract(legacy)

    expect(normalized).toEqual({ model: "Chronos", phase: "forecast", eval_split: "val" })
    expect(legacy).toEqual({ model: "Chronos", eval_split: "val" })
  })

  test("rejects an invalid task-level window stride", () => {
    expect(() =>
      normalizeForecastExecutionContract({
        model: "DLinear",
        phase: "forecast",
        eval_split: "val",
        window_stride: 0,
      }),
    ).toThrow("window_stride must be a positive integer")
  })

  test("rejects an invalid forecast gap", () => {
    expect(() =>
      normalizeForecastExecutionContract({
        model: "DLinear",
        phase: "forecast",
        eval_split: "val",
        forecast_gap: -1,
      }),
    ).toThrow("forecast_gap must be a nonnegative integer")
  })

  test("normalizes validation-front fit replay as a forecast-only execution", () => {
    expect(
      normalizeForecastExecutionContract({
        model: "DLinear",
        eval_split: "adjustment_fit",
      }),
    ).toMatchObject({
      phase: "forecast",
      eval_split: "adjustment_fit",
    })
  })

  test("generate_model rejects adjustment-fit executions", () => {
    expect(() =>
      normalizeGenerateModelExecutionContract({
        model: "DLinear",
        phase: "forecast",
        eval_split: "adjustment_fit",
      }),
    ).toThrow("generate_model only supports")
  })

  test("rejects cross-phase split combinations", () => {
    expect(() =>
      normalizeForecastExecutionContract({
        model: "DLinear",
        phase: "forecast",
        eval_split: "test",
      }),
    ).toThrow("Invalid forecast execution contract")
  })

  test("matches a legacy validation spec to its normalized post-forecast test spec", () => {
    const validationSpec = { model: "Chronos", eval_split: "val" }
    const testSpec = { model: "Chronos", phase: "post-forecast", eval_split: "test" }

    expect(validationSpecMatchesPostForecastTestSpec(validationSpec, testSpec)).toBe(true)
    expect(validationSpecMatchesPostForecastTestSpec(validationSpec, { ...testSpec, window_stride: 1 })).toBe(true)
    expect(validationSpecMatchesPostForecastTestSpec(validationSpec, { ...testSpec, forecast_gap: 0 })).toBe(true)
    expect(validationSpecMatchesPostForecastTestSpec(validationSpec, { ...testSpec, forecast_gap: 1 })).toBe(false)
    expect(validationSpecMatchesPostForecastTestSpec(validationSpec, { ...testSpec, model: "Moirai" })).toBe(false)
  })

  test("ignores split-specific prediction-grid anchors when matching validation to test", () => {
    const validationSpec = {
      model: "UserPrior",
      phase: "forecast",
      eval_split: "val",
      prediction_grid_target_anchor_index: 48_144,
    }
    const testSpec = {
      model: "UserPrior",
      phase: "post-forecast",
      eval_split: "test",
      prediction_grid_target_anchor_index: 49_560,
    }

    expect(validationSpecMatchesPostForecastTestSpec(validationSpec, testSpec)).toBe(true)
    expect(validationPostForecastTestSpecDifferences(validationSpec, testSpec)).toEqual([])
  })

  test("reports the exact frozen fields that prevent validation-to-test reuse", () => {
    const validationSpec = {
      model: "UserPrior",
      phase: "forecast",
      eval_split: "val",
      adjustment_skill_priors: "immutable expert rule",
      reflection_enabled: true,
    }
    const testSpec = {
      model: "UserPrior",
      phase: "post-forecast",
      eval_split: "test",
      reflection_enabled: false,
    }

    expect(validationPostForecastTestSpecDifferences(validationSpec, testSpec)).toEqual([
      "adjustment_skill_priors",
      "reflection_enabled",
    ])
  })
})
