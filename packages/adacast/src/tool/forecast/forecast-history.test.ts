import { describe, expect, test } from "bun:test"
import { deduplicateExperimentsByRunId, experimentAttemptIndex } from "./forecast-history"

describe("deduplicateExperimentsByRunId", () => {
  test("keeps the authoritative first record for each deterministic run id", () => {
    const records = [
      { run_id: "run-a", spec: { dataset_path: "data.csv" } },
      { run_id: "run-a", spec: { model: "manual-summary" } },
      { run_id: "run-b", spec: { dataset_path: "data.csv" } },
    ]

    expect(deduplicateExperimentsByRunId(records)).toEqual([records[0], records[2]])
  })

  test("uses an explicitly newer retry attempt as the authoritative record", () => {
    const records = [
      { run_id: "run-a", attempt_index: 1, status: "crash" },
      { run_id: "run-b", status: "success" },
      { run_id: "run-a", attempt_index: 2, status: "success" },
    ]

    expect(deduplicateExperimentsByRunId(records)).toEqual([records[2], records[1]])
    expect(experimentAttemptIndex(records[1])).toBe(1)
  })
})
