import { describe, expect, test } from "bun:test"
import { mkdir, writeFile } from "fs/promises"
import matter from "gray-matter"
import path from "path"
import { tmpdir } from "../../../test/fixture/fixture"
import { formatForecastStatus, readForecastStatus } from "./status"
import type { ExperimentRecord, ForecastPhaseState } from "./types"

const task = {
  dataset_path: "/tmp/dataset.csv",
  target_col: "value",
  time_col: "timestamp",
  pred_len: 24,
  seq_len: 48,
  label_len: 24,
  train_ratio: 0.7,
  val_ratio: 0.2,
  test_ratio: 0.1,
  eval_metrics: ["mse", "mae", "wape", "mase"],
  model_families: ["DLinear", "ARIMA"],
  frozen_at: "2026-04-30T00:00:00.000Z",
}

function forecastState(overrides: Partial<ForecastPhaseState> = {}): ForecastPhaseState {
  return {
    phase: "init",
    skills_confirmed: false,
    skills_confirmed_at: null,
    pre_forecast_completed: false,
    pre_forecast_completed_at: null,
    task_frozen_at: null,
    transitions: [],
    auto_mode: {
      enabled: false,
      active_agent: null,
      state: "waiting",
      pause_reason: null,
      auto_approve_permissions: false,
      auto_approve_skills: false,
      handoffs: [],
    },
    consecutive_no_improvement: 0,
    consecutive_crashes: 0,
    stop_counter_last_run_id: null,
    stop_counter_last_attempt_index: null,
    stop_reason: null,
    budget_initialized: false,
    adjustment: {
      enabled: false,
      mode: "off",
      strategy: null,
      user_decision: "pending",
      user_decision_at: null,
      user_decision_actor: null,
      user_decision_reason: null,
      policy_path: null,
      source_model_set_locked: false,
      source_model_set_path: null,
      fit_run_manifest_path: null,
      fit_runs_ready: false,
      active_skill_version: null,
      active_skill_path: null,
      forecaster_locked: false,
      no_effective_skill: false,
      no_effective_skill_reason: null,
      test_collection_run_ids: [],
      test_eligible_run_ids: [],
      test_application_records: [],
      critic_user_confirmed: null,
      test_application_decision: "pending",
      test_application_decision_at: null,
      test_application_decision_actor: null,
      test_application_decision_reason: null,
      test_application_run_id: null,
      test_adjustment_locked: false,
      test_adjustment_run_id: null,
      test_adjustment_candidate_id: null,
    },
    reflection: {
      enabled: true,
      configured: false,
      policy_path: null,
      policy_sha256: null,
      controller_version: "1",
      max_reflection_rounds: 3,
      max_format_retries: 2,
      validation_records: [],
      test_records: [],
    },
    prediction_prior: {
      enabled: false,
      user_decision: "pending",
      user_decision_at: null,
      user_decision_actor: null,
      user_decision_reason: null,
      prior_dataset_path: null,
      prior_dataset_sha256: null,
      manifest_path: null,
      source_model_set_id: null,
      validation_run_id: null,
      validation_metrics: null,
      fit_run_id: null,
      fit_metrics: null,
      test_run_id: null,
      test_metrics: null,
      test_materialized: false,
    },
    ...overrides,
  }
}

function experiment(input: {
  run_id: string
  model?: string
  status?: ExperimentRecord["status"]
  mse?: number
}): ExperimentRecord {
  const mse = input.mse ?? 1

  return {
    run_id: input.run_id,
    model: input.model ?? "DLinear",
    status: input.status ?? "success",
    metrics:
      input.status && input.status !== "success"
        ? null
        : {
            mse,
            mae: mse / 2,
            wape: mse / 3,
            mase: mse / 4,
          },
    spec: { model: input.model ?? "DLinear" },
    decision: input.status && input.status !== "success" ? "crash" : "keep",
    elapsed_seconds: 12,
    timestamp: "2026-04-30T00:00:00.000Z",
  }
}

async function writeJson(file: string, value: unknown) {
  await mkdir(path.dirname(file), { recursive: true })
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`, "utf-8")
}

async function writeState(directory: string, state: ForecastPhaseState) {
  const forecastRoot = path.join(directory, ".forecast")
  await mkdir(forecastRoot, { recursive: true })
  await writeFile(path.join(forecastRoot, "STATE.md"), matter.stringify("", state), "utf-8")
}

async function appendHistory(directory: string, records: ExperimentRecord[]) {
  await writeFile(
    path.join(directory, ".forecast", "history.jsonl"),
    records.map((record) => JSON.stringify(record)).join("\n") + "\n",
    "utf-8",
  )
}

async function writeManifest(directory: string) {
  await writeJson(path.join(directory, ".forecast", "best-model", "manifest.json"), {
    generated_at: "2026-04-30T00:00:00.000Z",
    model: "DLinear",
    model_family: "deep_learning",
    validation_run_id: "validation-run",
    test_run_id: "test-run",
    validation_metrics: {
      mse: 2,
      mae: 1,
      wape: 0.5,
      mase: 0.25,
    },
    test_metrics: {
      mse: 1,
      mae: 0.5,
      wape: 0.25,
      mase: 0.125,
    },
    task,
    spec_hash: "test-spec-hash",
    validation_spec_hash: "validation-spec-hash",
    selection_reason: "lowest test mse",
    source_paths: {
      validation_run: ".forecast/runs/validation-run",
      test_run: ".forecast/runs/test-run",
    },
    artifacts: [
      {
        kind: "checkpoint",
        path: "model/checkpoint.pth",
        format: "torch_state_dict",
        sha256: "a".repeat(64),
        model: "DLinear",
        model_family: "deep_learning",
      },
    ],
  })
}

describe("forecast status", () => {
  test("missing .forecast returns workspace false and formats without throwing", async () => {
    await using tmp = await tmpdir()

    const status = await readForecastStatus(tmp.path)
    const text = formatForecastStatus(status)

    expect(status.workspace.exists).toBe(false)
    expect(status.next_step).toBe("Start the forecast workflow.")
    expect(text).toContain("Workflow")
    expect(text).toContain("phase: unknown")
  })

  test("STATE.md and task.json populate phase and task summary", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path, forecastState({ phase: "forecast", task_frozen_at: task.frozen_at }))
    await writeJson(path.join(tmp.path, ".forecast", "task.json"), task)

    const status = await readForecastStatus(tmp.path)

    expect(status.workflow.phase).toBe("forecast")
    expect(status.task.dataset).toBe(task.dataset_path)
    expect(status.task.target).toBe(task.target_col)
    expect(status.task.horizon).toBe(task.pred_len)
    expect(status.task.forecast_gap).toBe(0)
    expect(status.task.model_horizon).toBe(task.pred_len)
    expect(status.task.model_families).toEqual(["DLinear", "ARIMA"])
  })

  test("adjustment status exposes whether decisions came from the Auto Mode LLM", async () => {
    await using tmp = await tmpdir()
    const state = forecastState({ phase: "forecast" })
    state.adjustment.user_decision = "enabled_a"
    state.adjustment.user_decision_actor = "llm_auto"
    state.adjustment.user_decision_reason =
      "A lightweight correction is proportionate to the remaining structured bias."
    await writeState(tmp.path, state)

    const status = await readForecastStatus(tmp.path)
    const text = formatForecastStatus(status)

    expect(status.adjustment.user_decision_actor).toBe("llm_auto")
    expect(status.adjustment.user_decision_reason).toContain("lightweight correction")
    expect(text).toContain("Generative prediction reasoning")
    expect(text).toContain("start decision actor: llm_auto")
    expect(text).toContain("start decision reason:")
  })

  test("history.jsonl counts success, crash, timeout, and ipc_error", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path, forecastState({ phase: "forecast" }))
    await appendHistory(tmp.path, [
      experiment({ run_id: "success-run" }),
      experiment({ run_id: "crash-run", status: "crash" }),
      experiment({ run_id: "timeout-run", status: "timeout" }),
      experiment({ run_id: "ipc-run", status: "ipc_error" }),
    ])

    const status = await readForecastStatus(tmp.path)

    expect(status.experiments.completed).toBe(4)
    expect(status.experiments.success).toBe(1)
    expect(status.experiments.failed).toBe(3)
    expect(status.experiments.crash).toBe(1)
    expect(status.experiments.timeout).toBe(1)
    expect(status.experiments.ipc_error).toBe(1)
  })

  test("best.json populates validation best", async () => {
    await using tmp = await tmpdir()
    const best = experiment({ run_id: "best-run", model: "ARIMA", mse: 0.42 })
    await writeState(tmp.path, forecastState({ phase: "forecast" }))
    await writeJson(path.join(tmp.path, ".forecast", "best.json"), best)

    const status = await readForecastStatus(tmp.path)

    expect(status.validation_best).toMatchObject({
      ready: true,
      source: "best.json",
      model: "ARIMA",
      run_id: "best-run",
      mse: 0.42,
      mae: 0.21,
    })
  })

  test("validation best falls back to lowest MSE in history", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path, forecastState({ phase: "forecast" }))
    await appendHistory(tmp.path, [
      experiment({ run_id: "worse-run", model: "ARIMA", mse: 10 }),
      experiment({ run_id: "best-run", model: "DLinear", mse: 2 }),
    ])

    const status = await readForecastStatus(tmp.path)

    expect(status.validation_best.source).toBe("history")
    expect(status.validation_best.run_id).toBe("best-run")
    expect(status.validation_best.mse).toBe(2)
  })

  test("valid best-model manifest shows package ready and test best", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path, forecastState({ phase: "post-forecast" }))
    await writeManifest(tmp.path)

    const status = await readForecastStatus(tmp.path)

    expect(status.outputs.best_model_package).toBe("ready")
    expect(status.test_best).toMatchObject({
      ready: true,
      source: "manifest",
      model: "DLinear",
      run_id: "test-run",
      mse: 1,
      mae: 0.5,
    })
  })

  test("corrupt manifest marks package invalid and records warning", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path, forecastState({ phase: "post-forecast" }))
    await mkdir(path.join(tmp.path, ".forecast", "best-model"), { recursive: true })
    await writeFile(path.join(tmp.path, ".forecast", "best-model", "manifest.json"), "{", "utf-8")

    const status = await readForecastStatus(tmp.path)

    expect(status.outputs.best_model_package).toBe("invalid")
    expect(status.test_best.ready).toBe(false)
    expect(status.warnings.some((warning) => warning.includes("manifest.json"))).toBe(true)
  })

  test("json output shape is stable and parseable", async () => {
    await using tmp = await tmpdir()
    await writeState(tmp.path, forecastState({ phase: "init" }))

    const status = await readForecastStatus(tmp.path)
    const parsed = JSON.parse(JSON.stringify(status))

    expect(Object.keys(parsed)).toEqual([
      "workspace",
      "workflow",
      "task",
      "experiments",
      "validation_best",
      "test_best",
      "adjustment",
      "outputs",
      "warnings",
      "next_step",
    ])
  })
})
