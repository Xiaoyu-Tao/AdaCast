import { describe, expect, test } from "bun:test"
import { mkdir, writeFile } from "fs/promises"
import { createHash } from "node:crypto"
import matter from "gray-matter"
import path from "path"
import { tmpdir } from "../../../test/fixture/fixture"
import {
  defaultForecastProgress,
  deriveForecastProgress,
  extractForecastLiveEvents,
  formatBestMetric,
  parseForecastHistory,
  readForecastProgress,
} from "./forecast-progress"
import type { ExperimentRecord, ForecastPhaseState } from "./types"

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex")
}

type StateOverrides = Omit<Partial<ForecastPhaseState>, "auto_mode"> & {
  auto_mode?: Partial<ForecastPhaseState["auto_mode"]>
}

const forecastTask = {
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
  model_families: ["DLinear"],
  frozen_at: "2026-04-21T00:00:00.000Z",
}

function forecastState(overrides: StateOverrides = {}): ForecastPhaseState {
  const base: ForecastPhaseState = {
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
  }

  return {
    ...base,
    ...overrides,
    auto_mode: {
      ...base.auto_mode,
      ...(overrides.auto_mode ?? {}),
    },
  }
}

function lockedAdjustment(overrides: Partial<ForecastPhaseState["adjustment"]> = {}): ForecastPhaseState["adjustment"] {
  return {
    ...forecastState().adjustment,
    enabled: true,
    mode: "rule_skill",
    strategy: "b_comprehensive",
    user_decision: "enabled_b",
    user_decision_at: "2026-04-21T00:00:00.000Z",
    user_decision_actor: "user",
    user_decision_reason: "User selected B in the generative prediction reasoning decision window.",
    policy_path: ".forecast/adjustments/policy.json",
    source_model_set_locked: true,
    source_model_set_path: ".forecast/adjustments/source-model-set.json",
    fit_run_manifest_path: ".forecast/adjustments/fit-runs/manifest.json",
    fit_runs_ready: true,
    active_skill_version: "v001",
    active_skill_path: ".forecast/adjustments/skills/active/SKILL.md",
    forecaster_locked: true,
    ...overrides,
  }
}

function experiment(input: { run_id: string; model?: string; mse?: number }): ExperimentRecord {
  const mse = input.mse ?? 1

  return {
    run_id: input.run_id,
    model: input.model ?? "DLinear",
    status: "success",
    metrics: {
      mse,
      mae: mse / 2,
      wape: mse / 3,
      mase: mse / 4,
    },
    spec: {
      model: input.model ?? "DLinear",
      pred_len: 24,
    },
    decision: "keep",
    elapsed_seconds: 12,
    timestamp: "2026-04-21T00:00:00.000Z",
  }
}

function artifacts(overrides: Partial<Parameters<typeof deriveForecastProgress>[0]["artifacts"]> = {}) {
  return {
    hasTask: false,
    hasPreForecastReport: false,
    testEvaluationCount: 0,
    hasWindowViz: false,
    hasTrajectoryViz: false,
    hasBestModelExport: false,
    finalReportState: "missing" as const,
    historicalExplanationsState: "missing" as const,
    ...overrides,
  }
}

function planner(snapshot: ReturnType<typeof deriveForecastProgress>) {
  return snapshot.rows.find((row) => row.id === "planner")
}

function forecaster(snapshot: ReturnType<typeof deriveForecastProgress>) {
  return snapshot.rows.find((row) => row.id === "forecaster")
}

function critic(snapshot: ReturnType<typeof deriveForecastProgress>) {
  return snapshot.rows.find((row) => row.id === "critic")
}

function step(row: NonNullable<ReturnType<typeof planner>>, label: string) {
  return row.steps.find((item) => item.label === label)
}

async function writeState(directory: string, state: ForecastPhaseState) {
  const forecastRoot = path.join(directory, ".forecast")
  await mkdir(forecastRoot, { recursive: true })
  await writeFile(path.join(forecastRoot, "STATE.md"), matter.stringify("", state), "utf-8")
}

async function writeJson(file: string, value: unknown) {
  await mkdir(path.dirname(file), { recursive: true })
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`, "utf-8")
}

describe("Planner -> Forecaster -> Critic forecast progress", () => {
  test("missing state returns a visible Manual planner checklist", async () => {
    await using tmp = await tmpdir()

    const snapshot = await readForecastProgress(tmp.path)

    expect(snapshot).toEqual(defaultForecastProgress())
    expect(snapshot.modeLabel).toBe("Manual")
    expect(snapshot.summary).toBe("Manual: Planner not started")
    expect(snapshot.rows.map((row) => row.id)).toEqual(["planner", "forecaster", "critic"])
    expect(planner(snapshot)?.active).toBe(true)
    expect(planner(snapshot)?.steps.map((item) => item.label)).toEqual([
      "Clarifying",
      "Task frozen",
      "Pre-forecast",
      "Skills confirmed",
    ])
  })

  test("pre-forecast state marks frozen planner steps complete and pre-forecast active", () => {
    const snapshot = deriveForecastProgress({
      state: forecastState({
        phase: "pre-forecast",
        task_frozen_at: "2026-04-21T00:00:00.000Z",
      }),
      history: [],
      best: null,
      artifacts: artifacts(),
    })

    const row = planner(snapshot)

    expect(step(row!, "Clarifying")?.status).toBe("complete")
    expect(step(row!, "Task frozen")?.status).toBe("complete")
    expect(step(row!, "Pre-forecast")?.status).toBe("active")
    expect(step(row!, "Skills confirmed")?.status).toBe("pending")
  })

  test("auto mode forecaster progress includes budget, remaining count, and best metric", () => {
    const best = experiment({ run_id: "best", model: "DLinear", mse: 1.23456 })
    const history = parseForecastHistory(
      [
        experiment({ run_id: "run-1", mse: 3 }),
        experiment({ run_id: "run-2", mse: 2 }),
        experiment({ run_id: "run-3", mse: 1.5 }),
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    )

    const snapshot = deriveForecastProgress({
      state: forecastState({
        phase: "forecast",
        auto_mode: {
          enabled: true,
          active_agent: "forecaster",
          state: "running",
        },
        budget_initialized: true,
        budget_thresholds: {
          max_experiments: 5,
          no_improvement_rounds: 3,
          crash_threshold: 3,
        },
      }),
      history,
      best,
      artifacts: artifacts(),
    })

    expect(formatBestMetric(best)).toBe("DLinear mse=1.235")
    expect(snapshot.summary).toBe("Auto Mode: Forecaster running")
    expect(forecaster(snapshot)?.detail).toContain("3/5 experiments")
    expect(forecaster(snapshot)?.detail).toContain("2 remaining")
    expect(forecaster(snapshot)?.detail).toContain("DLinear mse=1.235")
  })

  test("retry attempts do not consume extra progress budget", () => {
    const failed = {
      ...experiment({ run_id: "retry-run", mse: 2 }),
      attempt_index: 1,
      status: "crash" as const,
      metrics: null,
      decision: "crash" as const,
      failure_reason: "invalid artifact",
    }
    const recovered = {
      ...experiment({ run_id: "retry-run", mse: 1 }),
      attempt_index: 2,
    }
    const snapshot = deriveForecastProgress({
      state: forecastState({
        phase: "forecast",
        budget_initialized: true,
        budget_thresholds: {
          max_experiments: 5,
          no_improvement_rounds: 3,
          crash_threshold: 3,
        },
      }),
      history: [failed, recovered],
      best: recovered,
      artifacts: artifacts(),
    })

    expect(forecaster(snapshot)?.detail).toContain("1/5 experiments")
    expect(forecaster(snapshot)?.detail).toContain("4 remaining")
  })

  test("paused auto mode marks the active forecaster row warning and exposes pause reason", () => {
    const pauseReason = "no_improvement after 3 rounds; expert feedback required"
    const snapshot = deriveForecastProgress({
      state: forecastState({
        phase: "forecast",
        auto_mode: {
          enabled: true,
          active_agent: "forecaster",
          state: "paused",
          pause_reason: pauseReason,
        },
      }),
      history: [],
      best: null,
      artifacts: artifacts(),
    })

    expect(snapshot.pauseReason).toBe(pauseReason)
    expect(forecaster(snapshot)?.active).toBe(true)
    expect(forecaster(snapshot)?.status).toBe("warning")
    expect(forecaster(snapshot)?.detail).toContain(pauseReason)
  })

  test("forecaster stop reasons distinguish stopped and warning states", () => {
    const cases = [
      ["budget_exhausted", "stopped"],
      ["crash_threshold", "stopped"],
      ["no_improvement", "warning"],
    ] as const

    for (const [stopReason, status] of cases) {
      const snapshot = deriveForecastProgress({
        state: forecastState({
          phase: "forecast",
          stop_reason: stopReason,
        }),
        history: [],
        best: null,
        artifacts: artifacts(),
      })

      expect(forecaster(snapshot)?.status).toBe(status)
    }
  })

  test("post-forecast without critic artifacts starts at test-set evaluation", () => {
    const snapshot = deriveForecastProgress({
      state: forecastState({
        phase: "post-forecast",
      }),
      history: [],
      best: null,
      artifacts: artifacts(),
    })

    const row = critic(snapshot)

    expect(row?.active).toBe(true)
    expect(step(row!, "Evaluating raw test set")?.status).toBe("active")
    expect(step(row!, "Generating visualizations")?.status).toBe("pending")
    expect(step(row!, "Distillation")?.status).toBe("pending")
    expect(step(row!, "Writing final report")?.status).toBe("pending")
  })

  test("post-forecast with locked adjustment inserts adjustment step before visualizations", () => {
    const snapshot = deriveForecastProgress({
      state: forecastState({
        phase: "post-forecast",
        adjustment: {
          enabled: true,
          mode: "rule_skill",
          strategy: "b_comprehensive",
          user_decision: "enabled_b",
          user_decision_at: "2026-04-21T00:00:00.000Z",
          user_decision_actor: "user",
          user_decision_reason: "User selected B in the generative prediction reasoning decision window.",
          policy_path: ".forecast/adjustments/policy.json",
          source_model_set_locked: true,
          source_model_set_path: ".forecast/adjustments/source-model-set.json",
          fit_run_manifest_path: ".forecast/adjustments/fit-runs/manifest.json",
          fit_runs_ready: true,
          active_skill_version: "v001",
          active_skill_path: ".forecast/adjustments/skills/active/SKILL.md",
          forecaster_locked: true,
          no_effective_skill: false,
          no_effective_skill_reason: null,
          test_collection_run_ids: [],
          test_eligible_run_ids: [],
          test_application_records: [],
          critic_user_confirmed: "2026-04-21T00:00:00.000Z",
          test_application_decision: "apply",
          test_application_decision_at: "2026-04-21T00:00:00.000Z",
          test_application_decision_actor: "user",
          test_application_decision_reason: "User chose to keep the locked skill for the test stage.",
          test_application_run_id: "run-test-best",
          test_adjustment_locked: false,
          test_adjustment_run_id: null,
          test_adjustment_candidate_id: null,
        },
      }),
      history: [],
      best: null,
      artifacts: artifacts({ testEvaluationCount: 1 }),
    })

    const row = critic(snapshot)

    expect(step(row!, "Evaluating raw test set")?.status).toBe("complete")
    expect(step(row!, "Applying generative prediction reasoning")?.status).toBe("active")
    expect(step(row!, "Generating visualizations")?.status).toBe("pending")
  })

  test("configured reflection is an independent critic step before visualizations", () => {
    const configuredReflection = {
      ...forecastState().reflection,
      configured: true,
      policy_path: ".forecast/reflection/policy.json",
    }
    const activeSnapshot = deriveForecastProgress({
      state: forecastState({ phase: "post-forecast", reflection: configuredReflection }),
      history: [],
      best: null,
      artifacts: artifacts({ testEvaluationCount: 1 }),
    })

    expect(step(critic(activeSnapshot)!, "Running window reflection")?.status).toBe("active")
    expect(step(critic(activeSnapshot)!, "Generating visualizations")?.status).toBe("pending")

    const completeSnapshot = deriveForecastProgress({
      state: forecastState({
        phase: "post-forecast",
        reflection: {
          ...configuredReflection,
          test_records: [
            {
              test_run_id: "test-run",
              validation_run_id: "validation-run",
              baseline_kind: "raw_anchor",
              adjustment_candidate_id: null,
              validation_status: "enabled_no_trigger",
              llm_refinement_enabled: true,
              trigger_count: 0,
              skill_match_count: 0,
              llm_required_count: 0,
              corrected_count: 0,
              fallback_count: 0,
              deterministic_fallback_count: 0,
              raw_fallback_count: 0,
              baseline_metrics: null,
              final_metrics: null,
              result_path: ".forecast/reflection/test/test-run/raw/reflection_eval.json",
              completed_at: "2026-04-21T00:00:00.000Z",
            },
          ],
        },
      }),
      history: [],
      best: null,
      artifacts: artifacts({ testEvaluationCount: 1 }),
    })

    expect(step(critic(completeSnapshot)!, "Running window reflection")?.status).toBe("complete")
    expect(step(critic(completeSnapshot)!, "Generating visualizations")?.status).toBe("active")
  })

  test("critic adjustment remains active until every eligible test model is evaluated or skipped", () => {
    const firstRecord = {
      test_run_id: "run-dlinear",
      validation_run_id: "val-dlinear",
      model: "DLinear",
      decision: "apply" as const,
      decision_at: "2026-04-21T00:00:00.000Z",
      decision_actor: "auto_policy" as const,
      decision_reason: "Exact validation match.",
      adjustment_locked: true,
      adjustment_candidate_id: "candidate-test",
      adjustment_evaluated: true,
      adjusted_eval_path: ".forecast/adjustments/test/run-dlinear/candidate-test/adjusted_eval.json",
    }
    const pendingRecord = {
      test_run_id: "run-segrnn",
      validation_run_id: "val-segrnn",
      model: "SegRNN",
      decision: "pending" as const,
      decision_at: null,
      decision_actor: null,
      decision_reason: null,
      adjustment_locked: false,
      adjustment_candidate_id: null,
      adjustment_evaluated: false,
      adjusted_eval_path: null,
    }
    const activeSnapshot = deriveForecastProgress({
      state: forecastState({
        phase: "post-forecast",
        adjustment: lockedAdjustment({
          test_collection_run_ids: ["run-dlinear", "run-segrnn"],
          test_eligible_run_ids: ["run-dlinear", "run-segrnn"],
          test_application_records: [firstRecord, pendingRecord],
          test_adjustment_locked: true,
          test_adjustment_run_id: "run-dlinear",
          test_adjustment_candidate_id: "candidate-test",
        }),
      }),
      history: [],
      best: null,
      artifacts: artifacts({ testEvaluationCount: 2 }),
    })
    expect(step(critic(activeSnapshot)!, "Applying generative prediction reasoning")?.status).toBe("active")
    expect(step(critic(activeSnapshot)!, "Generating visualizations")?.status).toBe("pending")

    const completeSnapshot = deriveForecastProgress({
      state: forecastState({
        phase: "post-forecast",
        adjustment: lockedAdjustment({
          test_collection_run_ids: ["run-dlinear", "run-segrnn"],
          test_eligible_run_ids: ["run-dlinear", "run-segrnn"],
          test_application_records: [
            firstRecord,
            {
              ...pendingRecord,
              decision: "skip",
              decision_at: "2026-04-21T00:00:01.000Z",
              decision_actor: "user",
              decision_reason: "User skipped this model.",
            },
          ],
          test_adjustment_locked: true,
          test_adjustment_run_id: "run-dlinear",
          test_adjustment_candidate_id: "candidate-test",
        }),
      }),
      history: [],
      best: null,
      artifacts: artifacts({ testEvaluationCount: 2 }),
    })
    expect(step(critic(completeSnapshot)!, "Applying generative prediction reasoning")?.status).toBe("complete")
    expect(step(critic(completeSnapshot)!, "Generating visualizations")?.status).toBe("active")
  })

  test("post-forecast artifact mapping advances critic progress to final report writing", async () => {
    await using tmp = await tmpdir()
    const forecastRoot = path.join(tmp.path, ".forecast")

    await writeState(
      tmp.path,
      forecastState({
        phase: "post-forecast",
        task_frozen_at: "2026-04-21T00:00:00.000Z",
        pre_forecast_completed: true,
        pre_forecast_completed_at: "2026-04-21T00:01:00.000Z",
        skills_confirmed: true,
        skills_confirmed_at: "2026-04-21T00:02:00.000Z",
      }),
    )
    const checkpoint = "checkpoint"
    const predictions = "timestamp,actual,predicted\n0,1,1\n"
    const actual = "actual"
    const pred = "pred"
    const trainScale = '{"train_scale":1}'
    await writeJson(path.join(forecastRoot, "runs", "validation-run", "model", "metadata.json"), {
      kind: "checkpoint",
      path: "model/checkpoint.pth",
      sha256: sha256(checkpoint),
      model: "DLinear",
      eval_split: "val",
      phase: "forecast",
      trained_on: "train",
      format: "torch_state_dict",
    })
    await writeFile(path.join(forecastRoot, "runs", "validation-run", "model", "checkpoint.pth"), checkpoint, "utf-8")
    await writeJson(path.join(forecastRoot, "runs", "test-run", "spec.json"), {
      model: "DLinear",
      phase: "post-forecast",
      eval_split: "test",
    })
    await writeFile(path.join(forecastRoot, "runs", "test-run", "predictions.csv"), predictions, "utf-8")
    await writeFile(path.join(forecastRoot, "runs", "test-run", "actual.npy"), actual, "utf-8")
    await writeFile(path.join(forecastRoot, "runs", "test-run", "pred.npy"), pred, "utf-8")
    await writeFile(path.join(forecastRoot, "runs", "test-run", "train_scale.json"), trainScale, "utf-8")
    await writeJson(path.join(forecastRoot, "runs", "test-run", "model", "reuse.json"), {
      source_run_id: "validation-run",
      source_artifact_kind: "torch_state_dict",
      source_artifact_sha256: sha256(checkpoint),
      training_performed: false,
      evaluated_split: "test",
    })
    await writeJson(path.join(forecastRoot, "runs", "test-run", "eval.json"), {
      run_id: "test-run",
      evaluator_hash: "e".repeat(64),
      spec_hash: "test-spec-hash",
      metrics: {
        mse: 1,
        mae: 0.5,
        wape: 0.25,
        mase: 0.125,
      },
      eval_split: "test",
      timestamp: "2026-04-21T00:00:00.000Z",
      artifact_binding: {
        model_artifact: {
          kind: "torch_state_dict",
          sha256: sha256(checkpoint),
          source_run_id: "validation-run",
          foundation_revision: null,
        },
        prediction_artifacts: {
          predictions_sha256: sha256(predictions),
          actual_sha256: sha256(actual),
          pred_sha256: sha256(pred),
          train_scale_sha256: sha256(trainScale),
        },
      },
    })
    await mkdir(path.join(forecastRoot, "viz"), { recursive: true })
    await writeFile(path.join(forecastRoot, "viz", "window-level.png"), "png", "utf-8")
    await writeFile(path.join(forecastRoot, "viz", "trajectory.png"), "png", "utf-8")
    await writeJson(path.join(forecastRoot, "best-model", "manifest.json"), {
      generated_at: "2026-04-21T00:00:00.000Z",
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
      task: forecastTask,
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
    await mkdir(path.join(forecastRoot, "reports"), { recursive: true })
    await writeFile(
      path.join(forecastRoot, "reports", "final-report.md"),
      "_Pending: Critic agent will populate this report._\n",
      "utf-8",
    )

    const snapshot = await readForecastProgress(tmp.path)
    const row = critic(snapshot)

    expect(row?.active).toBe(true)
    expect(step(row!, "Evaluating raw test set")?.status).toBe("complete")
    expect(step(row!, "Generating visualizations")?.status).toBe("complete")
    expect(step(row!, "Distillation")?.status).toBe("complete")
    expect(step(row!, "Writing final report")?.status).toBe("active")
  })

  test("corrupt test evaluation and best-model artifacts do not complete critic steps", async () => {
    await using tmp = await tmpdir()
    const forecastRoot = path.join(tmp.path, ".forecast")

    await writeState(tmp.path, forecastState({ phase: "post-forecast" }))
    await writeJson(path.join(forecastRoot, "runs", "test-run", "spec.json"), {
      phase: "post-forecast",
      eval_split: "test",
    })
    await writeJson(path.join(forecastRoot, "runs", "test-run", "eval.json"), { status: "success" })
    await mkdir(path.join(forecastRoot, "viz"), { recursive: true })
    await writeFile(path.join(forecastRoot, "viz", "window-level.png"), "png", "utf-8")
    await writeFile(path.join(forecastRoot, "viz", "trajectory.png"), "png", "utf-8")
    await writeJson(path.join(forecastRoot, "best-model", "manifest.json"), { model: "DLinear" })

    const snapshot = await readForecastProgress(tmp.path)
    const row = critic(snapshot)

    expect(step(row!, "Evaluating raw test set")?.status).toBe("active")
    expect(step(row!, "Distillation")?.status).toBe("active")
    expect(step(row!, "Writing final report")?.status).toBe("pending")
  })

  test("snapshot avoids stale live-event placeholders when no live events are provided", () => {
    const snapshot = deriveForecastProgress({
      state: forecastState({
        phase: "forecast",
        auto_mode: {
          enabled: true,
          active_agent: "forecaster",
          state: "running",
        },
      }),
      history: [],
      best: null,
      artifacts: artifacts(),
    })

    const serialized = JSON.stringify(snapshot).toLowerCase()

    expect(serialized).not.toContain("unknown")
    expect(serialized).not.toContain("stale")
    expect(serialized).not.toContain("syncing")
  })

  test("extractForecastLiveEvents maps relevant running tool parts newest first", () => {
    const events = extractForecastLiveEvents([
      {
        type: "text",
      },
      {
        type: "tool",
        tool: "generate_model",
        state: {
          status: "running",
          title: "Training DLinear",
          input: { spec: { model: "DLinear" } },
          metadata: { runId: "run-generate", status: "running", progress: { type: "train_start" } },
          time: { start: 100 },
        },
      },
      {
        type: "tool",
        tool: "evaluate_experiment",
        state: {
          status: "running",
          input: { run_id: "run-eval" },
          metadata: { run_id: "run-eval", status: "running" },
          time: { start: 110 },
        },
      },
      {
        type: "tool",
        tool: "bash",
        state: {
          status: "running",
          title: "window_viz",
          input: {
            command: "uv run python -m adacast_ml.critic.window_viz --out .forecast/viz/window-level.png",
          },
          time: { start: 120 },
        },
      },
      {
        type: "tool",
        tool: "forecast_window_reflection",
        state: {
          status: "running",
          input: { subcommand: "run_test", run_id: "run-eval" },
          time: { start: 1.5 },
        },
      },
      {
        type: "tool",
        tool: "write",
        state: {
          status: "running",
          input: { path: ".forecast/reports/final-report.md" },
          time: { start: 130 },
        },
      },
      {
        type: "tool",
        tool: "export_best_model",
        state: {
          status: "running",
          input: { validation_run_id: "run-val", test_run_id: "run-test" },
          metadata: { status: "running" },
          time: { start: 125 },
        },
      },
      {
        type: "tool",
        tool: "read",
        state: {
          status: "running",
          input: {},
          time: { start: 140 },
        },
      },
    ])

    expect(events.map((event) => event.tool)).toEqual([
      "write",
      "export_best_model",
      "bash",
      "evaluate_experiment",
      "generate_model",
      "forecast_window_reflection",
    ])
    expect(events[1]).toMatchObject({ tool: "export_best_model", stage: "critic-distill" })
    expect(events[2]).toMatchObject({ tool: "bash", stage: "critic-viz" })
    expect(events[3]).toMatchObject({
      tool: "evaluate_experiment",
      stage: "critic-evaluate",
      status: "running",
      runId: "run-eval",
    })
    expect(events[4]).toMatchObject({
      tool: "generate_model",
      stage: "forecast",
      status: "running",
      runId: "run-generate",
      progressType: "train_start",
      title: "Training DLinear",
    })
    expect(events[5]).toMatchObject({
      tool: "forecast_window_reflection",
      stage: "critic-reflection",
      status: "running",
    })
  })

  test("live events overlay forecaster detail without replacing durable experiment counts", () => {
    const snapshot = deriveForecastProgress({
      state: forecastState({
        phase: "forecast",
        auto_mode: {
          enabled: true,
          active_agent: "forecaster",
          state: "running",
        },
        budget_thresholds: {
          max_experiments: 5,
          no_improvement_rounds: 3,
          crash_threshold: 3,
        },
      }),
      history: [experiment({ run_id: "run-1", mse: 3 }), experiment({ run_id: "run-2", mse: 2 })],
      best: experiment({ run_id: "run-best", model: "DLinear", mse: 1.23456 }),
      artifacts: artifacts(),
      liveEvents: [
        {
          tool: "generate_model",
          stage: "forecast",
          status: "running",
          runId: "run-live",
          progressType: "train_start",
        },
      ],
    })

    expect(forecaster(snapshot)?.detail).toBe(
      "2/5 experiments, 3 remaining, best DLinear mse=1.235, running train_start",
    )
  })

  test("live events can advance critic substeps only during post-forecast workflow", () => {
    const liveEvents = extractForecastLiveEvents([
      {
        type: "tool",
        tool: "evaluate_experiment",
        state: {
          status: "running",
          input: { run_id: "run-eval" },
          metadata: { run_id: "run-eval", status: "running" },
          time: { start: 1 },
        },
      },
      {
        type: "tool",
        tool: "bash",
        state: {
          status: "running",
          input: {
            command: "uv run python -m adacast_ml.critic.trajectory_viz --out .forecast/viz/trajectory.png",
          },
          output: "generated .forecast/viz/trajectory.png",
          time: { start: 2 },
        },
      },
      {
        type: "tool",
        tool: "write",
        state: {
          status: "running",
          input: { path: ".forecast/reports/final-report.md" },
          time: { start: 3 },
        },
      },
      {
        type: "tool",
        tool: "export_best_model",
        state: {
          status: "running",
          input: { validation_run_id: "run-val", test_run_id: "run-eval" },
          metadata: { status: "running" },
          time: { start: 2.5 },
        },
      },
    ])
    const forecastSnapshot = deriveForecastProgress({
      state: forecastState({ phase: "forecast" }),
      history: [],
      best: null,
      artifacts: artifacts(),
      liveEvents,
    })
    const postForecastSnapshot = deriveForecastProgress({
      state: forecastState({ phase: "post-forecast" }),
      history: [],
      best: null,
      artifacts: artifacts(),
      liveEvents,
    })

    expect(step(critic(forecastSnapshot)!, "Evaluating raw test set")?.status).toBe("pending")
    expect(step(critic(forecastSnapshot)!, "Generating visualizations")?.status).toBe("pending")
    expect(step(critic(forecastSnapshot)!, "Distillation")?.status).toBe("pending")
    expect(step(critic(forecastSnapshot)!, "Writing final report")?.status).toBe("pending")
    expect(step(critic(postForecastSnapshot)!, "Evaluating raw test set")?.status).toBe("active")
    expect(step(critic(postForecastSnapshot)!, "Generating visualizations")?.status).toBe("active")
    expect(step(critic(postForecastSnapshot)!, "Distillation")?.status).toBe("active")
    expect(step(critic(postForecastSnapshot)!, "Writing final report")?.status).toBe("active")
  })
})
