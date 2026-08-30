import { afterEach, describe, expect, mock, spyOn, test } from "bun:test"
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "fs"
import { mkdir, readFile, writeFile } from "fs/promises"
import { createHash } from "node:crypto"
import path from "path"
import { PassThrough } from "stream"
import { Instance } from "@/project/instance"
import { MessageID, SessionID } from "@/session/schema"
import { Process } from "@/util/process"
import { tmpdir } from "../../../test/fixture/fixture"
import { computeSpecHash } from "./hash"
import { statisticalModelArtifact } from "./model-family"
import type { ForecastModelEnvironment } from "./model-environment"

const resolvedForecastEnv: ForecastModelEnvironment = {
  env: {
    HF_ENDPOINT: "https://hf-mirror.example",
    HF_HUB_CACHE: "/resolved/hub",
    HF_XET_CACHE: "/resolved/xet",
    HF_ASSETS_CACHE: "/resolved/assets",
  },
  cacheRoot: "/resolved",
  hubCache: "/resolved/hub",
  xetCache: "/resolved/xet",
  assetsCache: "/resolved/assets",
  transformersCache: null,
  endpoint: "https://hf-mirror.example",
  source: "operator",
  warnings: [],
}

const resolveForecastModelEnvironmentMock = mock(() => resolvedForecastEnv)
const MOCK_RUN_ID = "a".repeat(64)
const FOUNDATION_REVISION = "b".repeat(40)

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex")
}

const ctx = {
  sessionID: SessionID.make("ses_generate-model-env-test"),
  messageID: MessageID.make("msg_generate-model-env-test"),
  callID: "generate-model-env-test-call",
  agent: "generate-model-env-test-agent",
  abort: AbortSignal.any([]),
  messages: [],
  metadata: () => {},
  ask: async () => {},
}

function mockChildProcess(output: object): Process.Child {
  const stdout = new PassThrough()
  const stderr = new PassThrough()

  const exited = new Promise<number>((resolve) => {
    queueMicrotask(() => {
      stdout.end(JSON.stringify(output))
      stderr.end()
      resolve(0)
    })
  })

  return {
    stdout,
    stderr,
    kill: mock(() => true),
    exited,
  } as unknown as Process.Child
}

function mockSuccessfulChildProcess(
  command: string[],
  output: {
    run_id: string
    out_dir: string
    metrics: { mse: number; mae: number; wape: number; mase: null }
    status?: "success" | "cached"
  },
): Process.Child {
  const specPath = command[command.indexOf("--spec") + 1]
  const runDir = output.out_dir
  const spec = JSON.parse(readFileSync(specPath, "utf-8"))
  mkdirSync(runDir, { recursive: true })
  copyFileSync(specPath, path.join(runDir, "spec.json"))
  const predictionArtifacts = {
    predictions: "timestamp,actual,predicted,prediction_layout,window_index,horizon_index\nx,1,1,sliding_window,0,0\n",
    actual: "",
    pred: "",
    trainScale: JSON.stringify({ train_scale: 1 }),
  }
  writeFileSync(path.join(runDir, "predictions.csv"), predictionArtifacts.predictions, "utf-8")
  writeFileSync(path.join(runDir, "actual.npy"), predictionArtifacts.actual)
  writeFileSync(path.join(runDir, "pred.npy"), predictionArtifacts.pred)
  writeFileSync(path.join(runDir, "train_scale.json"), predictionArtifacts.trainScale, "utf-8")
  const reuseIndex = command.indexOf("--reuse-model-from")
  let sourceRunId = output.run_id
  let sourceArtifactKind = "torch_state_dict"
  let sourceArtifactSha256 = ""
  let foundationRevision: string | null = null
  if (reuseIndex >= 0) {
    const sourceRunDir = command[reuseIndex + 1]
    const sourceModel = String(spec.model)
    const statisticalArtifact = statisticalModelArtifact(sourceModel)
    sourceRunId = path.basename(sourceRunDir)
    sourceArtifactKind = statisticalArtifact
      ? statisticalArtifact.kind
      : sourceModel.toLocaleLowerCase("en-US") === "chronos"
        ? "foundation_model_reference"
        : "torch_state_dict"
    sourceArtifactSha256 =
      sourceArtifactKind === "foundation_model_reference"
        ? JSON.parse(readFileSync(path.join(sourceRunDir, "model", "foundation.json"), "utf-8")).sha256
        : JSON.parse(readFileSync(path.join(sourceRunDir, "model", "metadata.json"), "utf-8")).sha256
    foundationRevision = sourceArtifactKind === "foundation_model_reference" ? FOUNDATION_REVISION : null
    mkdirSync(path.join(runDir, "model"), { recursive: true })
    writeFileSync(
      path.join(runDir, "model", "reuse.json"),
      JSON.stringify({
        source_run_id: sourceRunId,
        source_artifact_kind: sourceArtifactKind,
        source_artifact_sha256: sourceArtifactSha256,
        foundation_revision: foundationRevision,
        training_performed: false,
        evaluated_split: spec.eval_split,
      }),
      "utf-8",
    )
  } else if (spec.eval_split === "val") {
    const model = String(spec.model)
    const statisticalArtifact = statisticalModelArtifact(model)
    const modelDir = path.join(runDir, "model")
    mkdirSync(modelDir, { recursive: true })
    if (model.toLocaleLowerCase("en-US") === "chronos") {
      sourceArtifactKind = "foundation_model_reference"
      sourceArtifactSha256 = computeSpecHash({
        repo_id: "amazon/chronos-bolt-base",
        revision: FOUNDATION_REVISION,
      })
      foundationRevision = FOUNDATION_REVISION
      writeFileSync(
        path.join(modelDir, "foundation.json"),
        JSON.stringify({
          kind: "foundation_model_reference",
          model: "Chronos",
          repo_id: "amazon/chronos-bolt-base",
          revision: FOUNDATION_REVISION,
          sha256: computeSpecHash({
            repo_id: "amazon/chronos-bolt-base",
            revision: FOUNDATION_REVISION,
          }),
          format: "huggingface_snapshot",
        }),
        "utf-8",
      )
    } else if (statisticalArtifact) {
      const results = "statistical-results"
      sourceArtifactKind = statisticalArtifact.kind
      sourceArtifactSha256 = sha256(results)
      writeFileSync(path.join(modelDir, statisticalArtifact.filename), results, "utf-8")
      writeFileSync(
        path.join(modelDir, "metadata.json"),
        JSON.stringify({
          kind: statisticalArtifact.metadataKind,
          path: `model/${statisticalArtifact.filename}`,
          sha256: sourceArtifactSha256,
          model,
          eval_split: "val",
          phase: "forecast",
          trained_on: "train",
          format: statisticalArtifact.format,
        }),
        "utf-8",
      )
    } else {
      const checkpoint = "checkpoint"
      sourceArtifactSha256 = sha256(checkpoint)
      writeFileSync(path.join(modelDir, "checkpoint.pth"), checkpoint, "utf-8")
      writeFileSync(
        path.join(modelDir, "metadata.json"),
        JSON.stringify({
          kind: "checkpoint",
          path: "model/checkpoint.pth",
          sha256: sha256(checkpoint),
          model,
          eval_split: "val",
          phase: "forecast",
          trained_on: "train",
          format: "torch_state_dict",
        }),
        "utf-8",
      )
    }
  }
  writeFileSync(
    path.join(runDir, "eval.json"),
    JSON.stringify({
      run_id: output.run_id,
      evaluator_hash: "eval-hash",
      spec_hash: output.run_id,
      metrics: output.metrics,
      eval_split: spec.eval_split,
      timestamp: "2026-06-07T00:00:00.000Z",
      artifact_binding: {
        model_artifact: {
          kind: sourceArtifactKind,
          sha256: sourceArtifactSha256,
          source_run_id: sourceRunId,
          foundation_revision: foundationRevision,
        },
        prediction_artifacts: {
          predictions_sha256: sha256(predictionArtifacts.predictions),
          actual_sha256: sha256(predictionArtifacts.actual),
          pred_sha256: sha256(predictionArtifacts.pred),
          train_scale_sha256: sha256(predictionArtifacts.trainScale),
        },
      },
    }),
    "utf-8",
  )
  return mockChildProcess({ ...output, status: output.status ?? "success" })
}

async function writeTask(directory: string, overrides: Record<string, unknown> = {}) {
  const forecastDir = path.join(directory, ".forecast")
  await mkdir(forecastDir, { recursive: true })
  await writeFile(
    path.join(forecastDir, "task.json"),
    JSON.stringify(
      {
        dataset_path: "/tmp/dataset.csv",
        target_col: "value",
        time_col: "timestamp",
        pred_len: 24,
        seq_len: 48,
        label_len: 24,
        window_stride: 1,
        train_ratio: 0.7,
        val_ratio: 0.2,
        test_ratio: 0.1,
        eval_metrics: ["mse", "mae"],
        model_families: ["DLinear"],
        frozen_at: new Date().toISOString(),
        ...overrides,
      },
      null,
      2,
    ),
    "utf-8",
  )
}

async function writeReusableValidationRun(directory: string, runId: string, model = "DLinear") {
  const runDir = path.join(directory, ".forecast", "runs", runId)
  const spec = {
    model,
    dataset_path: "/tmp/dataset.csv",
    target_col: "value",
    time_col: "timestamp",
    pred_len: 24,
    seq_len: 48,
    label_len: 24,
    train_ratio: 0.7,
    val_ratio: 0.2,
    test_ratio: 0.1,
    metrics: ["mse", "mae"],
    adjustment_fit_holdout_ratio: 1 / 2,
    phase: "forecast",
    eval_split: "val",
  }
  await mkdir(path.join(runDir, "model"), { recursive: true })
  await writeFile(path.join(runDir, "spec.json"), JSON.stringify(spec, null, 2), "utf-8")
  const predictionArtifacts = {
    predictions: "timestamp,actual,predicted\nx,1,1\n",
    actual: "",
    pred: "",
    trainScale: JSON.stringify({ train_scale: 1 }),
  }
  await writeFile(path.join(runDir, "predictions.csv"), predictionArtifacts.predictions, "utf-8")
  await writeFile(path.join(runDir, "actual.npy"), predictionArtifacts.actual)
  await writeFile(path.join(runDir, "pred.npy"), predictionArtifacts.pred)
  await writeFile(path.join(runDir, "train_scale.json"), predictionArtifacts.trainScale, "utf-8")
  let artifactKind = "torch_state_dict"
  let artifactSha256 = sha256("checkpoint")
  let foundationRevision: string | null = null
  if (model.toLocaleLowerCase("en-US") !== "chronos") {
    const checkpoint = "checkpoint"
    await writeFile(path.join(runDir, "model", "checkpoint.pth"), checkpoint, "utf-8")
    await writeFile(
      path.join(runDir, "model", "metadata.json"),
      JSON.stringify({
        kind: "checkpoint",
        path: "model/checkpoint.pth",
        sha256: sha256(checkpoint),
        model,
        eval_split: "val",
        phase: "forecast",
        trained_on: "train",
        format: "torch_state_dict",
      }),
      "utf-8",
    )
  } else {
    artifactKind = "foundation_model_reference"
    artifactSha256 = computeSpecHash({
      repo_id: "amazon/chronos-bolt-base",
      revision: FOUNDATION_REVISION,
    })
    foundationRevision = FOUNDATION_REVISION
    await writeFile(
      path.join(runDir, "model", "foundation.json"),
      JSON.stringify({
        kind: "foundation_model_reference",
        model: "Chronos",
        repo_id: "amazon/chronos-bolt-base",
        revision: FOUNDATION_REVISION,
        sha256: computeSpecHash({
          repo_id: "amazon/chronos-bolt-base",
          revision: FOUNDATION_REVISION,
        }),
        format: "huggingface_snapshot",
      }),
      "utf-8",
    )
  }
  await writeFile(
    path.join(runDir, "eval.json"),
    JSON.stringify({
      run_id: runId,
      evaluator_hash: "eval-hash",
      spec_hash: runId,
      metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
      eval_split: "val",
      timestamp: "2026-06-07T00:00:00.000Z",
      artifact_binding: {
        model_artifact: {
          kind: artifactKind,
          sha256: artifactSha256,
          source_run_id: runId,
          foundation_revision: foundationRevision,
        },
        prediction_artifacts: {
          predictions_sha256: sha256(predictionArtifacts.predictions),
          actual_sha256: sha256(predictionArtifacts.actual),
          pred_sha256: sha256(predictionArtifacts.pred),
          train_scale_sha256: sha256(predictionArtifacts.trainScale),
        },
      },
    }),
    "utf-8",
  )
  return spec
}

afterEach(async () => {
  resolveForecastModelEnvironmentMock.mockClear()
  await Instance.disposeAll()
})

describe("GenerateModelTool cache_config subprocess env", () => {
  test("passes resolved HF_ENDPOINT and HF_HUB_CACHE HF_XET_CACHE HF_ASSETS_CACHE to Process.spawn", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path, { window_stride: 24, forecast_gap: 57 })

    const spawnSpy = spyOn(Process, "spawn").mockImplementation((command) =>
      mockSuccessfulChildProcess(command, {
        run_id: MOCK_RUN_ID,
        out_dir: path.join(tmp.path, ".forecast", "runs", MOCK_RUN_ID),
        metrics: {
          mse: 1,
          mae: 0.5,
          wape: 0.25,
          mase: null,
        },
      }),
    )
    const modelEnvironment = await import("./model-environment")
    const resolveSpy = spyOn(modelEnvironment, "resolveForecastModelEnvironment").mockImplementation(
      resolveForecastModelEnvironmentMock,
    )

    try {
      const { GenerateModelTool } = await import("./generate-model")
      const tool = await GenerateModelTool.init()

      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          await tool.execute(
            {
              spec: {
                model: "DLinear",
                pred_len: 24,
                seq_len: 48,
                label_len: 24,
                train_ratio: 0.4,
                val_ratio: 0.3,
                test_ratio: 0.3,
                split_mode: "timestamp",
                train_end_timestamp: "2099-01-01",
                validation_end_timestamp: "2099-02-01",
                adjustment_fit_holdout_ratio: 0.2,
                window_stride: 999,
                forecast_gap: 999,
                stride: 8,
              },
            },
            ctx,
          )
        },
      })

      expect(resolveForecastModelEnvironmentMock).toHaveBeenCalledWith({
        directory: tmp.path,
        env: process.env,
      })
      expect(spawnSpy).toHaveBeenCalledTimes(1)
      const options = spawnSpy.mock.calls[0]?.[1]
      const command = spawnSpy.mock.calls[0]?.[0] as string[]
      expect(options?.env).toEqual(
        expect.objectContaining({
          HF_ENDPOINT: "https://hf-mirror.example",
          HF_HUB_CACHE: "/resolved/hub",
          HF_XET_CACHE: "/resolved/xet",
          HF_ASSETS_CACHE: "/resolved/assets",
        }),
      )
      expect(command[command.indexOf("--run-id") + 1]).toMatch(/^[a-f0-9]{64}$/)
      const generatedSpec = JSON.parse(await readFile(command[command.indexOf("--spec") + 1], "utf-8"))
      expect(generatedSpec.adjustment_fit_holdout_ratio).toBeCloseTo(1 / 2)
      expect(generatedSpec.train_ratio).toBeCloseTo(0.7)
      expect(generatedSpec.val_ratio).toBeCloseTo(0.2)
      expect(generatedSpec.test_ratio).toBeCloseTo(0.1)
      expect(generatedSpec.split_mode).toBeUndefined()
      expect(generatedSpec.train_end_timestamp).toBeUndefined()
      expect(generatedSpec.validation_end_timestamp).toBeUndefined()
      expect(generatedSpec.window_stride).toBe(24)
      expect(generatedSpec.forecast_gap).toBe(57)
      expect(generatedSpec.stride).toBe(8)
      expect(generatedSpec.phase).toBe("forecast")
      expect(generatedSpec.eval_split).toBe("val")
      expect(generatedSpec.input_contract_version).toBe(3)
      const history = JSON.parse((await readFile(path.join(tmp.path, ".forecast", "history.jsonl"), "utf-8")).trim())
      expect(history.run_id).toBe(MOCK_RUN_ID)
      expect(history.spec.phase).toBe("forecast")
      expect(history.spec.eval_split).toBe("val")
      expect(history.spec.input_contract_version).toBe(3)
      expect(history.spec.window_stride).toBe(24)
      expect(history.spec.forecast_gap).toBe(57)
    } finally {
      resolveSpy.mockRestore()
      spawnSpy.mockRestore()
    }
  })

  test("enables heavy model extra for foundation model names case-insensitively", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path)

    const spawnSpy = spyOn(Process, "spawn").mockImplementation((command) =>
      mockSuccessfulChildProcess(command, {
        run_id: MOCK_RUN_ID,
        out_dir: path.join(tmp.path, ".forecast", "runs", MOCK_RUN_ID),
        metrics: {
          mse: 1,
          mae: 0.5,
          wape: 0.25,
          mase: null,
        },
      }),
    )
    const modelEnvironment = await import("./model-environment")
    const resolveSpy = spyOn(modelEnvironment, "resolveForecastModelEnvironment").mockImplementation(
      resolveForecastModelEnvironmentMock,
    )

    try {
      const { GenerateModelTool } = await import("./generate-model")
      const tool = await GenerateModelTool.init()

      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          await tool.execute(
            {
              spec: {
                model: "chronos",
                pred_len: 24,
                seq_len: 48,
                label_len: 24,
                input_contract_version: 1,
              },
            },
            ctx,
          )
        },
      })

      const command = spawnSpy.mock.calls[0]?.[0] as string[] | undefined
      expect(command).toContain("--extra")
      expect(command).toContain("heavy-models")
      expect(command?.indexOf("--extra")).toBeLessThan(command?.indexOf("python") ?? 0)
      const generatedSpec = JSON.parse(
        await readFile((command as string[])[(command as string[]).indexOf("--spec") + 1], "utf-8"),
      )
      expect(generatedSpec.input_contract_version).toBe(3)
    } finally {
      resolveSpy.mockRestore()
      spawnSpy.mockRestore()
    }
  })

  test("uses the sliding statistical prediction contract without changing neural contracts", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path)
    const spawnSpy = spyOn(Process, "spawn").mockImplementation((command) => {
      const assignedRunId = command[command.indexOf("--run-id") + 1]
      return mockSuccessfulChildProcess(command, {
        run_id: assignedRunId,
        out_dir: path.join(tmp.path, ".forecast", "runs", assignedRunId),
        metrics: {
          mse: 1,
          mae: 0.5,
          wape: 0.25,
          mase: null,
        },
      })
    })
    const modelEnvironment = await import("./model-environment")
    const resolveSpy = spyOn(modelEnvironment, "resolveForecastModelEnvironment").mockImplementation(
      resolveForecastModelEnvironmentMock,
    )

    try {
      const { GenerateModelTool } = await import("./generate-model")
      const tool = await GenerateModelTool.init()
      const first = await Instance.provide({
        directory: tmp.path,
        fn: () =>
          tool.execute(
            {
              spec: {
                model: "ARIMA",
                pred_len: 24,
                seq_len: 48,
                label_len: 24,
                input_contract_version: 1,
              },
            },
            ctx,
          ),
      })
      const second = await Instance.provide({
        directory: tmp.path,
        fn: () =>
          tool.execute(
            {
              spec: {
                model: "ARIMA",
                pred_len: 24,
                seq_len: 48,
                label_len: 24,
                input_contract_version: 1,
              },
            },
            ctx,
          ),
      })

      const command = spawnSpy.mock.calls[0]?.[0] as string[]
      const generatedSpec = JSON.parse(await readFile(command[command.indexOf("--spec") + 1], "utf-8"))
      expect(generatedSpec.input_contract_version).toBe(4)
      expect(JSON.parse(first.output).status).toBe("success")
      expect(JSON.parse(second.output).status).toBe("cached")
      expect(spawnSpy).toHaveBeenCalledTimes(1)
      const history = (await readFile(path.join(tmp.path, ".forecast", "history.jsonl"), "utf-8")).trim().split("\n")
      expect(history).toHaveLength(1)
    } finally {
      resolveSpy.mockRestore()
      spawnSpy.mockRestore()
    }
  })

  test("preserves cached status returned by the runner without duplicating history", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path)
    const spawnSpy = spyOn(Process, "spawn").mockImplementation((command) =>
      mockSuccessfulChildProcess(command, {
        run_id: MOCK_RUN_ID,
        out_dir: path.join(tmp.path, ".forecast", "runs", MOCK_RUN_ID),
        metrics: {
          mse: 1,
          mae: 0.5,
          wape: 0.25,
          mase: null,
        },
        status: "cached",
      }),
    )
    const modelEnvironment = await import("./model-environment")
    const resolveSpy = spyOn(modelEnvironment, "resolveForecastModelEnvironment").mockImplementation(
      resolveForecastModelEnvironmentMock,
    )

    try {
      const { GenerateModelTool } = await import("./generate-model")
      const tool = await GenerateModelTool.init()
      const output = await Instance.provide({
        directory: tmp.path,
        fn: () =>
          tool.execute(
            {
              spec: {
                model: "DLinear",
                pred_len: 24,
                seq_len: 48,
                label_len: 24,
              },
            },
            ctx,
          ),
      })

      expect(JSON.parse(output.output).status).toBe("cached")
      expect(output.metadata.cached).toBe(true)
      expect(await readFile(path.join(tmp.path, ".forecast", "history.jsonl"), "utf-8").catch(() => null)).toBeNull()
    } finally {
      resolveSpy.mockRestore()
      spawnSpy.mockRestore()
    }
  })

  test("copies the frozen timestamp split contract into generated specs", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path, {
      split_mode: "timestamp",
      train_ratio: 0.6,
      val_ratio: 0.2,
      test_ratio: 0.2,
      train_end_timestamp: "2026-01-25 23:00:00",
      validation_end_timestamp: "2026-02-03 07:00:00",
    })
    const spawnSpy = spyOn(Process, "spawn").mockImplementation((command) =>
      mockSuccessfulChildProcess(command, {
        run_id: MOCK_RUN_ID,
        out_dir: path.join(tmp.path, ".forecast", "runs", MOCK_RUN_ID),
        metrics: {
          mse: 1,
          mae: 0.5,
          wape: 0.25,
          mase: null,
        },
      }),
    )
    const modelEnvironment = await import("./model-environment")
    const resolveSpy = spyOn(modelEnvironment, "resolveForecastModelEnvironment").mockImplementation(
      resolveForecastModelEnvironmentMock,
    )

    try {
      const { GenerateModelTool } = await import("./generate-model")
      const tool = await GenerateModelTool.init()
      await Instance.provide({
        directory: tmp.path,
        fn: () =>
          tool.execute(
            {
              spec: {
                model: "DLinear",
                pred_len: 24,
                seq_len: 48,
                label_len: 24,
              },
            },
            ctx,
          ),
      })

      const command = spawnSpy.mock.calls[0]?.[0] as string[]
      const generatedSpec = JSON.parse(await readFile(command[command.indexOf("--spec") + 1], "utf-8"))
      expect(generatedSpec.split_mode).toBe("timestamp")
      expect(generatedSpec.train_ratio).toBeCloseTo(0.6)
      expect(generatedSpec.val_ratio).toBeCloseTo(0.2)
      expect(generatedSpec.test_ratio).toBeCloseTo(0.2)
      expect(generatedSpec.train_end_timestamp).toBe("2026-01-25 23:00:00")
      expect(generatedSpec.validation_end_timestamp).toBe("2026-02-03 07:00:00")
    } finally {
      resolveSpy.mockRestore()
      spawnSpy.mockRestore()
    }
  })

  test("legacy tasks preserve omitted window_stride while enforcing runtime default one", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path, { window_stride: undefined })
    const spawnSpy = spyOn(Process, "spawn").mockImplementation((command) =>
      mockSuccessfulChildProcess(command, {
        run_id: MOCK_RUN_ID,
        out_dir: path.join(tmp.path, ".forecast", "runs", MOCK_RUN_ID),
        metrics: {
          mse: 1,
          mae: 0.5,
          wape: 0.25,
          mase: null,
        },
      }),
    )
    const modelEnvironment = await import("./model-environment")
    const resolveSpy = spyOn(modelEnvironment, "resolveForecastModelEnvironment").mockImplementation(
      resolveForecastModelEnvironmentMock,
    )

    try {
      const { GenerateModelTool } = await import("./generate-model")
      const tool = await GenerateModelTool.init()
      await Instance.provide({
        directory: tmp.path,
        fn: () =>
          tool.execute(
            {
              spec: {
                model: "DLinear",
                pred_len: 24,
                seq_len: 48,
                label_len: 24,
                window_stride: 999,
              },
            },
            ctx,
          ),
      })

      const command = spawnSpy.mock.calls[0]?.[0] as string[]
      const generatedSpec = JSON.parse(await readFile(command[command.indexOf("--spec") + 1], "utf-8"))
      expect(generatedSpec.window_stride).toBeUndefined()
      expect(generatedSpec.forecast_gap).toBeUndefined()
    } finally {
      resolveSpy.mockRestore()
      spawnSpy.mockRestore()
    }
  })

  test("passes the selected validation run to post-forecast test inference", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path)
    const sourceRunId = "b".repeat(64)
    const sourceSpec = await writeReusableValidationRun(tmp.path, sourceRunId)

    const spawnSpy = spyOn(Process, "spawn").mockImplementation((command) =>
      mockSuccessfulChildProcess(command, {
        run_id: MOCK_RUN_ID,
        out_dir: path.join(tmp.path, ".forecast", "runs", MOCK_RUN_ID),
        metrics: {
          mse: 1,
          mae: 0.5,
          wape: 0.25,
          mase: null,
        },
      }),
    )

    try {
      const { GenerateModelTool } = await import("./generate-model")
      const tool = await GenerateModelTool.init()

      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          await tool.execute(
            {
              reuse_model_from_run_id: sourceRunId,
              spec: {
                ...sourceSpec,
                phase: "post-forecast",
                eval_split: "test",
              },
            },
            ctx,
          )
        },
      })

      const command = spawnSpy.mock.calls[0]?.[0] as string[]
      expect(command[command.indexOf("--reuse-model-from") + 1]).toBe(
        path.join(tmp.path, ".forecast", "runs", sourceRunId),
      )
      const history = JSON.parse((await readFile(path.join(tmp.path, ".forecast", "history.jsonl"), "utf-8")).trim())
      expect(history.spec.phase).toBe("post-forecast")
      expect(history.spec.eval_split).toBe("test")
    } finally {
      spawnSpy.mockRestore()
    }
  })

  test("rejects post-forecast test inference without a selected validation run", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path)
    const { GenerateModelTool } = await import("./generate-model")
    const tool = await GenerateModelTool.init()

    await expect(
      Instance.provide({
        directory: tmp.path,
        fn: async () =>
          tool.execute(
            {
              spec: {
                model: "DLinear",
                pred_len: 24,
                seq_len: 48,
                label_len: 24,
                phase: "post-forecast",
                eval_split: "test",
              },
            },
            ctx,
          ),
      }),
    ).rejects.toThrow("requires reuse_model_from_run_id")
  })

  test.each([
    ["forecast/test", { phase: "forecast", eval_split: "test" }],
    ["post-forecast/val", { phase: "post-forecast", eval_split: "val" }],
    ["pre-forecast/val", { phase: "pre-forecast", eval_split: "val" }],
  ])("rejects the invalid execution contract %s before spawning Python", async (_label, execution) => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path)
    const spawnSpy = spyOn(Process, "spawn")

    try {
      const { GenerateModelTool } = await import("./generate-model")
      const tool = await GenerateModelTool.init()
      await expect(
        Instance.provide({
          directory: tmp.path,
          fn: async () =>
            tool.execute(
              {
                spec: {
                  model: "DLinear",
                  pred_len: 24,
                  seq_len: 48,
                  label_len: 24,
                  ...execution,
                },
              },
              ctx,
            ),
        }),
      ).rejects.toThrow("generate_model only supports")
      expect(spawnSpy).not.toHaveBeenCalled()
    } finally {
      spawnSpy.mockRestore()
    }
  })

  test("rejects a fourth unique validation attempt after the crash threshold", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path)
    await mkdir(path.join(tmp.path, ".forecast"), { recursive: true })
    const crashes = [1, 2, 3].map((index) => ({
      run_id: `crash-${index}`,
      model: "TimeMixer",
      status: "crash",
      metrics: null,
      spec: {
        model: "TimeMixer",
        phase: "forecast",
        eval_split: "val",
      },
      failure_reason: "configuration error",
      elapsed_seconds: 1,
      timestamp: `2026-06-11T00:0${index}:00.000Z`,
    }))
    await writeFile(
      path.join(tmp.path, ".forecast", "history.jsonl"),
      `${crashes.map((record) => JSON.stringify(record)).join("\n")}\n`,
      "utf-8",
    )
    const spawnSpy = spyOn(Process, "spawn")
    const { GenerateModelTool } = await import("./generate-model")
    const tool = await GenerateModelTool.init()

    try {
      await expect(
        Instance.provide({
          directory: tmp.path,
          fn: () =>
            tool.execute(
              {
                spec: {
                  model: "DLinear",
                  pred_len: 24,
                  seq_len: 48,
                  label_len: 24,
                },
              },
              ctx,
            ),
        }),
      ).rejects.toThrow("Forecast crash threshold reached")
      expect(spawnSpy).not.toHaveBeenCalled()
    } finally {
      spawnSpy.mockRestore()
    }
  })

  test("retries an invalid deterministic validation run with an audited attempt index", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path)
    const spec = {
      dataset_path: "/tmp/dataset.csv",
      target_col: "value",
      time_col: "timestamp",
      pred_len: 24,
      seq_len: 48,
      label_len: 24,
      train_ratio: 0.7,
      val_ratio: 0.2,
      test_ratio: 0.1,
      metrics: ["mse", "mae"],
      phase: "forecast",
      eval_split: "val",
      model: "DLinear",
      adjustment_fit_holdout_ratio: 1 / 2,
      input_contract_version: 3,
    }
    const runId = computeSpecHash(spec)
    await writeFile(
      path.join(tmp.path, ".forecast", "history.jsonl"),
      `${JSON.stringify({
        run_id: runId,
        attempt_index: 1,
        model: "DLinear",
        status: "crash",
        metrics: null,
        spec,
        failure_reason: "configuration error",
        elapsed_seconds: 1,
        timestamp: "2026-06-11T00:00:00.000Z",
      })}\n`,
      "utf-8",
    )
    const spawnSpy = spyOn(Process, "spawn").mockImplementation((command) =>
      mockSuccessfulChildProcess(command, {
        run_id: runId,
        out_dir: path.join(tmp.path, ".forecast", "runs", runId),
        metrics: {
          mse: 1,
          mae: 0.5,
          wape: 0.25,
          mase: null,
        },
      }),
    )
    const { GenerateModelTool } = await import("./generate-model")
    const tool = await GenerateModelTool.init()

    try {
      const output = await Instance.provide({
        directory: tmp.path,
        fn: () =>
          tool.execute(
            {
              spec: {
                model: "DLinear",
                pred_len: 24,
                seq_len: 48,
                label_len: 24,
              },
            },
            ctx,
          ),
      })
      expect(JSON.parse(output.output)).toMatchObject({ run_id: runId, status: "success" })
      expect(spawnSpy).toHaveBeenCalledTimes(1)
      const records = (await readFile(path.join(tmp.path, ".forecast", "history.jsonl"), "utf-8"))
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line))
      expect(records).toHaveLength(2)
      expect(records.map((record) => record.attempt_index)).toEqual([1, 2])
      expect(records.map((record) => record.status)).toEqual(["crash", "success"])
    } finally {
      spawnSpy.mockRestore()
    }
  })

  test("rejects a selected validation run whose checkpoint changed after evaluation", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path)
    const sourceRunId = "e".repeat(64)
    const sourceSpec = await writeReusableValidationRun(tmp.path, sourceRunId)
    await writeFile(
      path.join(tmp.path, ".forecast", "runs", sourceRunId, "model", "checkpoint.pth"),
      "tampered-checkpoint",
      "utf-8",
    )
    const spawnSpy = spyOn(Process, "spawn")

    try {
      const { GenerateModelTool } = await import("./generate-model")
      const tool = await GenerateModelTool.init()
      await expect(
        Instance.provide({
          directory: tmp.path,
          fn: async () =>
            tool.execute(
              {
                reuse_model_from_run_id: sourceRunId,
                spec: {
                  ...sourceSpec,
                  phase: "post-forecast",
                  eval_split: "test",
                },
              },
              ctx,
            ),
        }),
      ).rejects.toThrow("does not match its validation metadata")
      expect(spawnSpy).not.toHaveBeenCalled()
    } finally {
      spawnSpy.mockRestore()
    }
  })

  test("does not accept an old retrained test cache without matching reuse provenance", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path)
    const sourceRunId = "c".repeat(64)
    const sourceSpec = await writeReusableValidationRun(tmp.path, sourceRunId)
    const testSpec = {
      ...sourceSpec,
      phase: "post-forecast",
      eval_split: "test",
    }
    const testRunId = computeSpecHash(testSpec)
    const testRunDir = path.join(tmp.path, ".forecast", "runs", testRunId)
    await mkdir(testRunDir, { recursive: true })
    await writeFile(path.join(testRunDir, "spec.json"), JSON.stringify(testSpec), "utf-8")
    await writeFile(
      path.join(testRunDir, "eval.json"),
      JSON.stringify({
        run_id: testRunId,
        evaluator_hash: "eval-hash",
        spec_hash: testRunId,
        metrics: { mse: 9, mae: 3, wape: 1, mase: null },
        eval_split: "test",
        timestamp: "2026-06-07T00:00:00.000Z",
      }),
      "utf-8",
    )
    for (const [name, contents] of [
      ["predictions.csv", "timestamp,actual,predicted\nx,1,9\n"],
      ["actual.npy", ""],
      ["pred.npy", ""],
      ["train_scale.json", JSON.stringify({ train_scale: 1 })],
    ]) {
      await writeFile(path.join(testRunDir, name), contents, "utf-8")
    }

    const spawnSpy = spyOn(Process, "spawn").mockImplementation((command) =>
      mockSuccessfulChildProcess(command, {
        run_id: testRunId,
        out_dir: testRunDir,
        metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
      }),
    )

    try {
      const { GenerateModelTool } = await import("./generate-model")
      const tool = await GenerateModelTool.init()
      const result = await Instance.provide({
        directory: tmp.path,
        fn: async () =>
          tool.execute(
            {
              reuse_model_from_run_id: sourceRunId,
              spec: testSpec,
            },
            ctx,
          ),
      })

      expect(spawnSpy).toHaveBeenCalledTimes(1)
      expect(result.metadata.cached).not.toBe(true)
      const reuse = JSON.parse(await readFile(path.join(testRunDir, "model", "reuse.json"), "utf-8"))
      expect(reuse).toMatchObject({
        source_run_id: sourceRunId,
        training_performed: false,
        evaluated_split: "test",
      })
    } finally {
      spawnSpy.mockRestore()
    }
  })

  test("foundation-model test inference records the selected validation source without training artifacts", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path)
    const sourceRunId = "d".repeat(64)
    const sourceSpec = await writeReusableValidationRun(tmp.path, sourceRunId, "Chronos")
    const testSpec = { ...sourceSpec, phase: "post-forecast", eval_split: "test" }
    const testRunId = computeSpecHash(testSpec)

    const spawnSpy = spyOn(Process, "spawn").mockImplementation((command) =>
      mockSuccessfulChildProcess(command, {
        run_id: testRunId,
        out_dir: path.join(tmp.path, ".forecast", "runs", testRunId),
        metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
      }),
    )

    try {
      const { GenerateModelTool } = await import("./generate-model")
      const tool = await GenerateModelTool.init()
      await Instance.provide({
        directory: tmp.path,
        fn: async () =>
          tool.execute(
            {
              reuse_model_from_run_id: sourceRunId,
              spec: testSpec,
            },
            ctx,
          ),
      })

      const command = spawnSpy.mock.calls[0]?.[0] as string[]
      expect(command).toContain("heavy-models")
      expect(command[command.indexOf("--reuse-model-from") + 1]).toBe(
        path.join(tmp.path, ".forecast", "runs", sourceRunId),
      )
      const reuse = JSON.parse(
        await readFile(path.join(tmp.path, ".forecast", "runs", testRunId, "model", "reuse.json"), "utf-8"),
      )
      expect(reuse.source_artifact_kind).toBe("foundation_model_reference")
    } finally {
      spawnSpy.mockRestore()
    }
  })

  test("foundation-model test inference accepts a legacy validation spec without phase", async () => {
    await using tmp = await tmpdir()
    await writeTask(tmp.path)
    const sourceRunId = "f".repeat(64)
    const sourceSpec = await writeReusableValidationRun(tmp.path, sourceRunId, "Chronos")
    const { phase: _phase, ...legacySourceSpec } = sourceSpec
    await writeFile(
      path.join(tmp.path, ".forecast", "runs", sourceRunId, "spec.json"),
      JSON.stringify(legacySourceSpec, null, 2),
      "utf-8",
    )
    const testSpec = { ...legacySourceSpec, phase: "post-forecast", eval_split: "test" }
    const testRunId = computeSpecHash(testSpec)

    const spawnSpy = spyOn(Process, "spawn").mockImplementation((command) =>
      mockSuccessfulChildProcess(command, {
        run_id: testRunId,
        out_dir: path.join(tmp.path, ".forecast", "runs", testRunId),
        metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
      }),
    )

    try {
      const { GenerateModelTool } = await import("./generate-model")
      const tool = await GenerateModelTool.init()
      await Instance.provide({
        directory: tmp.path,
        fn: async () =>
          tool.execute(
            {
              reuse_model_from_run_id: sourceRunId,
              spec: testSpec,
            },
            ctx,
          ),
      })

      expect(spawnSpy).toHaveBeenCalledTimes(1)
      const command = spawnSpy.mock.calls[0]?.[0] as string[]
      expect(command[command.indexOf("--reuse-model-from") + 1]).toBe(
        path.join(tmp.path, ".forecast", "runs", sourceRunId),
      )
    } finally {
      spawnSpy.mockRestore()
    }
  })
})
