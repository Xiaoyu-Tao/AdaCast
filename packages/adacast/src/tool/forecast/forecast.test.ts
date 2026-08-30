import { afterEach, describe, expect, spyOn, test } from "bun:test"
import { mkdir, readFile, writeFile } from "fs/promises"
import matter from "gray-matter"
import path from "path"
import z from "zod"
import { Instance } from "@/project/instance"
import { ModelID, ProviderID } from "@/provider/schema"
import { Question } from "@/question"
import { Session } from "@/session"
import { SessionPrompt } from "@/session/prompt"
import { MessageID, PartID, SessionID } from "@/session/schema"
import { tmpdir } from "../../../test/fixture/fixture"
import { PhaseGateError, TaskFrozenError } from "./errors"
import { normalizeForecastStateFrontmatter } from "./forecast-progress"
import { ForecastStateTool } from "./forecast-state"
import { summarizeCacheDiagnostics } from "./generate-model"
import { computeFileSha256 } from "./model-artifact"
import { ForecastPhaseState } from "./types"
import { ForecastPredictionPriorTool } from "./forecast-prediction-prior"
import { StderrEvent } from "./types"
import { ForecastTaskTool } from "./forecast-task"

type ForecastTaskInput = z.input<Awaited<ReturnType<typeof ForecastTaskTool.init>>["parameters"]>

const ctx = {
  sessionID: SessionID.make("ses_forecast-test"),
  messageID: MessageID.make("msg_forecast-test"),
  callID: "forecast-test-call",
  agent: "forecast-test-agent",
  abort: AbortSignal.any([]),
  messages: [],
  metadata: () => {},
  ask: async () => {},
}

function ctxWithUserPrompt(text: string): typeof ctx {
  const messageID = MessageID.make("msg_forecast-user-prompt")
  return {
    ...ctx,
    messages: [
      {
        info: {
          id: messageID,
          sessionID: ctx.sessionID,
          role: "user",
          time: { created: Date.now() },
          agent: "planner",
          model: testModel,
        },
        parts: [
          {
            id: PartID.make("part_forecast-user-prompt"),
            sessionID: ctx.sessionID,
            messageID,
            type: "text",
            text,
          },
        ],
      },
    ] as never,
  }
}

const validTaskInput: ForecastTaskInput = {
  dataset_path: "/tmp/dataset.csv",
  target_col: "value",
  time_col: "timestamp",
  pred_len: 24,
  seq_len: 48,
  label_len: 24,
  train_ratio: 0.7,
  val_ratio: 0.2,
  test_ratio: 0.1,
  eval_metrics: ["mse", "mae"],
  model_families: ["DLinear"],
  reflection_enabled: false,
}

const testModel = {
  providerID: ProviderID.make("test"),
  modelID: ModelID.make("test-model"),
}

function toolCtx(input: { sessionID?: SessionID; agent?: string } = {}) {
  return {
    ...ctx,
    sessionID: input.sessionID ?? ctx.sessionID,
    agent: input.agent ?? ctx.agent,
  }
}

async function answerPendingQuestion(label: string) {
  for (let attempt = 0; attempt < 30; attempt += 1) {
    const pending = await Question.list()
    if (pending[0]) {
      await Question.reply({ requestID: pending[0].id, answers: [[label]] })
      return pending[0]
    }
    await new Promise((resolve) => setTimeout(resolve, 5))
  }
  throw new Error("No pending question was created")
}

function record(run_id: string, mse: number) {
  return {
    run_id,
    model: "DLinear",
    status: "success" as const,
    metrics: {
      mse,
      mae: mse / 2,
      wape: mse / 3,
      mase: mse / 4,
    },
    spec: {
      model: "DLinear",
      pred_len: 24,
    },
    decision: "keep" as const,
    elapsed_seconds: 12,
    timestamp: new Date().toISOString(),
  }
}

async function defineTask(directory: string) {
  await seedForecastPythonProject(directory)

  await Instance.provide({
    directory,
    fn: async () => {
      const tool = await ForecastTaskTool.init()
      await tool.execute(validTaskInput, ctx)
    },
  })
}

async function seedForecastPythonProject(directory: string) {
  const pythonDir = path.join(directory, "python")
  await mkdir(pythonDir, { recursive: true })
  await writeFile(path.join(pythonDir, "pyproject.toml"), '[project]\nname = "adacast-test"\n', "utf-8")
}

async function writePreForecastReport(directory: string) {
  const { mkdir, writeFile } = await import("fs/promises")
  const reportsDir = path.join(directory, ".forecast", "reports")
  await mkdir(reportsDir, { recursive: true })
  await writeFile(
    path.join(reportsDir, "pre-forecast.md"),
    [
      "# Pre-Forecast Report",
      "",
      "## Task Summary",
      "Synthetic forecast task.",
      "",
      "## Data Characteristics",
      "Test data characteristics.",
      "",
      "## Domain Context",
      "Test domain context.",
      "",
      "## Risk Flags",
      "Test risk flags.",
      "",
      "## Modeling Direction Recommendations",
      "Test modeling directions.",
      "",
    ].join("\n"),
    "utf-8",
  )
}

async function enterForecastPhase(directory: string) {
  await Instance.provide({
    directory,
    fn: async () => {
      const { execute } = await ForecastStateTool.init()

      await execute({ subcommand: "init" }, ctx)
      await recordPredictionPriorDeclined(directory)
      await execute({ subcommand: "transition", target_phase: "pre-forecast" }, ctx)
      await writePreForecastReport(directory)

      await execute({ subcommand: "transition", target_phase: "skills-review" }, ctx)
      await execute({ subcommand: "confirm_skills" }, ctx)
      await execute({ subcommand: "transition", target_phase: "forecast" }, ctx)
    },
  })
}

async function createAutoModeSession(agent: "planner" | "forecaster" = "planner"): Promise<SessionID> {
  const session = await Session.create({ title: "Auto Mode forecast test" })
  await SessionPrompt.prompt({
    sessionID: session.id,
    agent,
    model: testModel,
    noReply: true,
    parts: [{ type: "text", text: `Seed ${agent} Auto Mode message.` }],
  })
  return session.id
}

async function lastUserMessage(sessionID: SessionID) {
  const messages = await Session.messages({ sessionID })
  return messages.findLast((message) => message.info.role === "user")
}

function textParts(message: Awaited<ReturnType<typeof lastUserMessage>>) {
  return message?.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("\n")
}

async function readForecastPhaseState(directory: string) {
  const raw = await readFile(path.join(directory, ".forecast", "STATE.md"), "utf-8")
  return ForecastPhaseState.parse(normalizeForecastStateFrontmatter(matter(raw).data as Record<string, unknown>))
}

async function writeForecastPhaseState(directory: string, state: ForecastPhaseState) {
  await writeFile(
    path.join(directory, ".forecast", "STATE.md"),
    matter.stringify("\n# Forecast State Log\n", state),
    "utf-8",
  )
}

async function recordPredictionPriorDeclined(directory: string) {
  const state = await readForecastPhaseState(directory)
  await writeForecastPhaseState(directory, {
    ...state,
    prediction_prior: {
      ...state.prediction_prior,
      enabled: false,
      user_decision: "declined",
      user_decision_at: "2026-06-04T00:00:00.000Z",
      user_decision_actor: "user",
      user_decision_reason: "Test fixture keeps the ordinary workflow.",
    },
  })
}

async function recordAdjustmentDisabled(directory: string) {
  const state = await readForecastPhaseState(directory)
  await writeForecastPhaseState(directory, {
    ...state,
    adjustment: {
      ...state.adjustment,
      enabled: false,
      mode: "off",
      strategy: null,
      user_decision: "disabled",
      user_decision_at: "2026-06-04T00:00:00.000Z",
      test_application_decision: "skip",
      test_application_decision_at: "2026-06-04T00:00:00.000Z",
    },
  })
}

afterEach(async () => {
  await Instance.disposeAll()
})

describe("forecast tools", () => {
  test("forecast_task rejects missing mandatory fields", async () => {
    await using tmp = await tmpdir()

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        await expect(
          tool.execute(
            {
              ...validTaskInput,
              target_col: undefined as never,
            },
            ctx,
          ),
        ).rejects.toThrow("target_col")
      },
    })
  })

  test("forecast_task rejects non-positive or inconsistent split ratios", async () => {
    await using tmp = await tmpdir()

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        await expect(
          tool.execute(
            {
              ...validTaskInput,
              train_ratio: 0.6,
            },
            ctx,
          ),
        ).rejects.toThrow("must equal 1")
        await expect(
          tool.execute(
            {
              ...validTaskInput,
              test_ratio: 0,
            },
            ctx,
          ),
        ).rejects.toThrow("expected number to be >0")
        await expect(
          tool.execute(
            {
              ...validTaskInput,
              window_stride: 0,
            },
            ctx,
          ),
        ).rejects.toThrow("expected number to be >0")
      },
    })
  })

  test("forecast_task writes task.json with frozen_at", async () => {
    await using tmp = await tmpdir()

    await defineTask(tmp.path)

    const taskPath = path.join(tmp.path, ".forecast", "task.json")
    const raw = await readFile(taskPath, "utf-8")
    const task = JSON.parse(raw)

    expect(task.frozen_at).toEqual(expect.any(String))
    expect(task.dataset_path).toBe(validTaskInput.dataset_path)
    expect(task.target_col).toBe(validTaskInput.target_col)
    expect(task.adjustment_fit_holdout_ratio).toBeCloseTo(1 / 2)
    expect(task.window_stride).toBe(1)
    expect(task.forecast_gap).toBe(0)
    expect(task.reflection_enabled).toBe(false)
  })

  test("forecast_task enables independent window reflection by default", async () => {
    await using tmp = await tmpdir()
    await seedForecastPythonProject(tmp.path)
    const { reflection_enabled: _reflectionEnabled, ...taskWithoutReflectionSetting } = validTaskInput

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        await tool.execute(taskWithoutReflectionSetting, ctx)
      },
    })

    const task = JSON.parse(await readFile(path.join(tmp.path, ".forecast", "task.json"), "utf-8"))
    expect(task.reflection_enabled).toBe(true)
  })

  test("explicit reflection opt-out stays synchronized whether task or state is initialized first", async () => {
    await using taskFirst = await tmpdir()
    await defineTask(taskFirst.path)
    await Instance.provide({
      directory: taskFirst.path,
      fn: async () => {
        const stateTool = await ForecastStateTool.init()
        await stateTool.execute({ subcommand: "init" }, ctx)
      },
    })
    expect((await readForecastPhaseState(taskFirst.path)).reflection.enabled).toBe(false)

    await using stateFirst = await tmpdir()
    await seedForecastPythonProject(stateFirst.path)
    await Instance.provide({
      directory: stateFirst.path,
      fn: async () => {
        const stateTool = await ForecastStateTool.init()
        await stateTool.execute({ subcommand: "init" }, ctx)
        const taskTool = await ForecastTaskTool.init()
        await taskTool.execute(validTaskInput, ctx)
      },
    })
    expect((await readForecastPhaseState(stateFirst.path)).reflection.enabled).toBe(false)
  })

  test("forecast_task requires the prediction-prior window as its next action", async () => {
    await using tmp = await tmpdir()

    await seedForecastPythonProject(tmp.path)
    const result = await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        return tool.execute(validTaskInput, ctx)
      },
    })

    expect(result.output).toContain("Required next action: call forecast_prediction_prior")
    expect(result.output).toContain("ask_start")
  })

  test("forecast_state rejects phase progress after the frozen task is edited", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)
    const taskPath = path.join(tmp.path, ".forecast", "task.json")
    const task = JSON.parse(await readFile(taskPath, "utf-8"))
    await writeFile(taskPath, JSON.stringify({ ...task, reflection_enabled: !task.reflection_enabled }, null, 2))

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()
        await execute({ subcommand: "init" }, ctx)
        const gateError = await execute({ subcommand: "transition", target_phase: "pre-forecast" }, ctx).catch(
          (error) => error,
        )
        expect(PhaseGateError.isInstance(gateError)).toBe(true)
        if (!PhaseGateError.isInstance(gateError)) throw gateError
        expect(gateError.data.missing_preconditions).toContain(
          "frozen task.json changed after forecast_task; restore the exact frozen task or start a new task instead of editing managed artifacts",
        )
      },
    })
  })

  test("forecast_task accepts expert priors that are not continuous Prompt excerpts", async () => {
    await using tmp = await tmpdir()
    const expertPrior = "When snow is forecast, increase the load forecast; the amount is unspecified."
    const prompt = `Build this task. Expert prior: ${expertPrior}`
    const synthesizedPrior = "During cold weather, raise the forecast by a suitable amount."

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        await tool.execute(
          {
            ...validTaskInput,
            adjustment_skill_priors: synthesizedPrior,
          },
          ctxWithUserPrompt(prompt),
        )
      },
    })

    const task = JSON.parse(await readFile(path.join(tmp.path, ".forecast", "task.json"), "utf-8"))
    expect(task.adjustment_skill_priors).toBe(synthesizedPrior)
    const taskLock = JSON.parse(await readFile(path.join(tmp.path, ".forecast", "task-lock.json"), "utf-8"))
    expect(taskLock).toMatchObject({ version: 1, frozen_at: task.frozen_at })
    expect(taskLock.task_sha256).toHaveLength(64)
  })

  test("forecast_task refuses to drop an explicit generative prediction expert prior from the user Prompt", async () => {
    await using tmp = await tmpdir()
    const prompt = "生成式预测推理先验 skill：当出现极端天气时上调预测值，具体幅度由验证集前半部分确定。"

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        await expect(tool.execute(validTaskInput, ctxWithUserPrompt(prompt))).rejects.toThrow(
          "adjustment_skill_priors is missing",
        )
      },
    })

    await expect(readFile(path.join(tmp.path, ".forecast", "task.json"), "utf-8")).rejects.toThrow()
  })

  test("forecast_task freezes an explicit unified window stride", async () => {
    await using tmp = await tmpdir()

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        await tool.execute({ ...validTaskInput, window_stride: 24 }, ctx)
      },
    })

    const task = JSON.parse(await readFile(path.join(tmp.path, ".forecast", "task.json"), "utf-8"))
    expect(task.window_stride).toBe(24)
  })

  test("forecast_task freezes an explicit forecast gap", async () => {
    await using tmp = await tmpdir()

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        await tool.execute({ ...validTaskInput, forecast_gap: 57 }, ctx)
      },
    })

    const task = JSON.parse(await readFile(path.join(tmp.path, ".forecast", "task.json"), "utf-8"))
    expect(task.forecast_gap).toBe(57)
  })

  test("forecast_task accepts stride as an alias without persisting the alias", async () => {
    await using tmp = await tmpdir()

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        await tool.execute({ ...validTaskInput, stride: 24 }, ctx)
      },
    })

    const task = JSON.parse(await readFile(path.join(tmp.path, ".forecast", "task.json"), "utf-8"))
    expect(task.window_stride).toBe(24)
    expect(task.stride).toBeUndefined()
  })

  test("forecast_task rejects conflicting stride aliases", async () => {
    await using tmp = await tmpdir()

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        await expect(tool.execute({ ...validTaskInput, window_stride: 24, stride: 12 }, ctx)).rejects.toThrow(
          "stride and window_stride must match",
        )
      },
    })
  })

  test("forecast_task freezes two inclusive origin timestamp boundaries", async () => {
    await using tmp = await tmpdir()
    const datasetPath = path.join(tmp.path, "timestamp-split.csv")
    const timestamps = Array.from({ length: 10 }, (_, index) => `2026-01-01 ${String(index).padStart(2, "0")}:00:00`)
    await writeFile(
      datasetPath,
      ["timestamp,value", ...timestamps.map((timestamp, index) => `${timestamp},${index}`)].join("\n"),
      "utf-8",
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        await tool.execute(
          {
            ...validTaskInput,
            dataset_path: datasetPath,
            train_ratio: undefined,
            val_ratio: undefined,
            test_ratio: undefined,
            train_end_timestamp: timestamps[3],
            validation_end_timestamp: timestamps[6],
          },
          ctx,
        )
      },
    })

    const task = JSON.parse(await readFile(path.join(tmp.path, ".forecast", "task.json"), "utf-8"))
    expect(task.split_mode).toBe("timestamp")
    expect(task.train_end_timestamp).toBe(timestamps[3])
    expect(task.validation_end_timestamp).toBe(timestamps[6])
    expect(task.train_ratio).toBeCloseTo(0.4)
    expect(task.val_ratio).toBeCloseTo(0.3)
    expect(task.test_ratio).toBeCloseTo(0.3)
  })

  test("forecast_task streams timestamp splits without retaining unrelated CSV fields", async () => {
    await using tmp = await tmpdir()
    const datasetPath = path.join(tmp.path, "large-timestamp-split.csv")
    const rowCount = 50_000
    const payload = "x".repeat(512)
    const rows = ["payload,timestamp,value"]
    for (let index = 0; index < rowCount; index += 1) {
      const timestamp = new Date(Date.UTC(2026, 0, 1) + index * 1000)
        .toISOString()
        .slice(0, 19)
        .replace("T", " ")
      rows.push(`${payload},${timestamp},${index}`)
    }
    await writeFile(datasetPath, rows.join("\n"), "utf-8")

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        await tool.execute(
          {
            ...validTaskInput,
            dataset_path: datasetPath,
            train_ratio: undefined,
            val_ratio: undefined,
            test_ratio: undefined,
            train_end_timestamp: "2026-01-01 00:00:19",
            validation_end_timestamp: "2026-01-01 00:00:39",
          },
          ctx,
        )
      },
    })

    const task = JSON.parse(await readFile(path.join(tmp.path, ".forecast", "task.json"), "utf-8"))
    expect(task.train_ratio).toBeCloseTo(20 / rowCount)
    expect(task.val_ratio).toBeCloseTo(20 / rowCount)
  })

  test("forecast_task recovers a pending freeze left by a terminated process", async () => {
    await using tmp = await tmpdir()
    const forecastRoot = path.join(tmp.path, ".forecast")
    await mkdir(forecastRoot, { recursive: true })
    await writeFile(path.join(forecastRoot, "task.json"), "partial task", "utf-8")
    await writeFile(path.join(forecastRoot, "task-lock.json"), "partial lock", "utf-8")
    await writeFile(
      path.join(forecastRoot, "task-freeze.pending.json"),
      JSON.stringify({ version: 1, state_content: null }),
      "utf-8",
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        await tool.execute(validTaskInput, ctx)
      },
    })

    expect(await Bun.file(path.join(forecastRoot, "task-freeze.pending.json")).exists()).toBe(false)
    expect(JSON.parse(await readFile(path.join(forecastRoot, "task.json"), "utf-8"))).toMatchObject({
      target_col: "value",
    })
  })

  test("forecast_task recovers explicit timestamp split, forecast gap, and stride from the user prompt", async () => {
    await using tmp = await tmpdir()
    const datasetPath = path.join(tmp.path, "timestamp-split.csv")
    await writeFile(
      datasetPath,
      ["timestamp,value", "2025/6/30 9:15,0", "2025/6/30 9:30,1", "2025/9/30 9:30,2", "2025/10/1 0:00,3"].join("\n"),
      "utf-8",
    )

    const prompt = [
      "我有一个区域电力负荷预测数据集。",
      "请使用时间戳切分，使用 2025-06-30 09:30:00 与 2025-09-30 09:30:00 这两个时间戳划分训练集、验证集和测试集。",
      "forecast_gap = 57",
      "stride = 96",
    ].join("\n")

    const result = await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        return tool.execute(
          {
            ...validTaskInput,
            dataset_path: datasetPath,
            train_ratio: 0.87,
            val_ratio: 0.076,
            test_ratio: 0.054,
            forecast_gap: 0,
          },
          ctxWithUserPrompt(prompt),
        )
      },
    })

    const task = JSON.parse(await readFile(path.join(tmp.path, ".forecast", "task.json"), "utf-8"))
    expect(task.split_mode).toBe("timestamp")
    expect(task.train_end_timestamp).toBe("2025-06-30 09:30:00")
    expect(task.validation_end_timestamp).toBe("2025-09-30 09:30:00")
    expect(task.train_ratio).toBeCloseTo(0.5)
    expect(task.val_ratio).toBeCloseTo(0.25)
    expect(task.test_ratio).toBeCloseTo(0.25)
    expect(task.forecast_gap).toBe(57)
    expect(task.window_stride).toBe(96)
    expect(result.output).toContain("Recovered task contract fields")
  })

  test("forecast_task recovers timestamp split, gap, and stride from task text fields", async () => {
    await using tmp = await tmpdir()
    const datasetPath = path.join(tmp.path, "huabei-style-timestamp-split.csv")
    await writeFile(
      datasetPath,
      [
        "timestamp,value",
        "2025/6/30 9:15,0",
        "2025/6/30 9:30,1",
        "2025/6/30 9:45,2",
        "2025/9/30 9:30,3",
        "2025/9/30 9:45,4",
        "2025/10/1 0:00,5",
      ].join("\n"),
      "utf-8",
    )

    const businessContext = [
      "时间戳切分：训练集截至 2025/6/30 9:30，验证集 2025/6/30 9:30~2025/9/30 9:30，测试集 2025/9/30 9:30 至末尾。",
      "预测窗口调度：gap=57，stride=96。",
    ].join("\n")

    const result = await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        return tool.execute(
          {
            ...validTaskInput,
            dataset_path: datasetPath,
            train_ratio: 0.87,
            val_ratio: 0.048,
            test_ratio: 0.082,
            forecast_gap: 0,
            business_context: businessContext,
          },
          ctx,
        )
      },
    })

    const task = JSON.parse(await readFile(path.join(tmp.path, ".forecast", "task.json"), "utf-8"))
    expect(task.split_mode).toBe("timestamp")
    expect(task.train_end_timestamp).toBe("2025/6/30 9:30")
    expect(task.validation_end_timestamp).toBe("2025/9/30 9:30")
    expect(task.train_ratio).toBeCloseTo(2 / 6)
    expect(task.val_ratio).toBeCloseTo(2 / 6)
    expect(task.test_ratio).toBeCloseTo(2 / 6)
    expect(task.forecast_gap).toBe(57)
    expect(task.window_stride).toBe(96)
    expect(result.output).toContain("Recovered task contract fields")
  })

  test("forecast_task rejects mixed and ambiguous timestamp split inputs", async () => {
    await using mixed = await tmpdir()
    await Instance.provide({
      directory: mixed.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        await expect(
          tool.execute(
            {
              ...validTaskInput,
              split_mode: "timestamp",
              train_end_timestamp: "2026-01-01 03:00:00",
              validation_end_timestamp: "2026-01-01 06:00:00",
            },
            ctx,
          ),
        ).rejects.toThrow("accepts only two timestamps")
      },
    })

    await using duplicate = await tmpdir()
    const datasetPath = path.join(duplicate.path, "duplicate.csv")
    await writeFile(
      datasetPath,
      [
        "timestamp,value",
        "2026-01-01 00:00:00,0",
        "2026-01-01 01:00:00,1",
        "2026-01-01 01:00:00,2",
        "2026-01-01 02:00:00,3",
      ].join("\n"),
      "utf-8",
    )
    await Instance.provide({
      directory: duplicate.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        await expect(
          tool.execute(
            {
              ...validTaskInput,
              dataset_path: datasetPath,
              train_ratio: undefined,
              val_ratio: undefined,
              test_ratio: undefined,
              train_end_timestamp: "2026-01-01 01:00:00",
              validation_end_timestamp: "2026-01-01 02:00:00",
            },
            ctx,
          ),
        ).rejects.toThrow("ambiguous")
      },
    })
  })

  test("forecast_task throws TaskFrozenError on duplicate", async () => {
    await using tmp = await tmpdir()

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastTaskTool.init()
        await tool.execute(validTaskInput, ctx)
        await expect(tool.execute(validTaskInput, ctx)).rejects.toBeInstanceOf(TaskFrozenError)
      },
    })
  })

  test("forecast_prediction_prior import cannot override a declined prior decision", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()
        await execute({ subcommand: "init" }, ctx)
      },
    })
    const state = await readForecastPhaseState(tmp.path)
    await writeForecastPhaseState(tmp.path, {
      ...state,
      prediction_prior: {
        ...state.prediction_prior,
        enabled: false,
        user_decision: "declined",
        user_decision_at: "2026-06-04T00:00:00.000Z",
        user_decision_actor: "user",
        user_decision_reason: "User chose the ordinary workflow.",
      },
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastPredictionPriorTool.init()
        await expect(
          tool.execute({ subcommand: "import", prior_dataset_path: path.join(tmp.path, "prior.csv") }, ctx),
        ).rejects.toThrow("user declined prediction-prior mode")
      },
    })
  })

  test("forecast_prediction_prior ask_start shows the early user choice window", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const stateTool = await ForecastStateTool.init()
        await stateTool.execute({ subcommand: "init" }, ctx)

        const priorTool = await ForecastPredictionPriorTool.init()
        const resultPromise = priorTool.execute({ subcommand: "ask_start" }, ctx)
        const question = await answerPendingQuestion("不提交预测先验")
        const result = JSON.parse((await resultPromise).output)

        expect(question.questions[0].header).toBe("预测先验")
        expect(question.questions[0].question).toContain("是否提交用户预测先验数据集")
        expect(question.questions[0].question).toContain("验证集和测试集目标列替换为你的预测先验")
        expect(question.questions[0].options.map((option) => option.label)).toEqual(["不提交预测先验", "提交预测先验"])
        expect(result.decision).toBe("declined")
      },
    })
  })

  test("forecast_prediction_prior materialize_test rematerializes stale cached UserPrior test runs", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)
    const fakeResult = {
      status: "success",
      action: "materialize-test",
      prior_dataset_path: path.join(tmp.path, "prior.csv"),
      prior_dataset_sha256: "a".repeat(64),
      test_run: {
        run_id: "fresh-test-run",
        spec_hash: "fresh-test-run",
        run_path: ".forecast/runs/fresh-test-run",
        spec_path: ".forecast/runs/fresh-test-run/spec.json",
        predictions_path: ".forecast/runs/fresh-test-run/predictions.csv",
        eval_path: ".forecast/runs/fresh-test-run/eval.json",
        metrics: { mse: 1, mae: 1, wape: 1, mase: null },
        model_artifact_sha256: "b".repeat(64),
      },
      user_prior_manifest: {},
    }
    const spawnSpy = spyOn(Bun, "spawn").mockImplementation(
      () =>
        ({
          stdout: new Response(JSON.stringify(fakeResult)).body,
          stderr: new Response("").body,
          exited: Promise.resolve(0),
        }) as never,
    )

    try {
      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const { execute: stateExecute } = await ForecastStateTool.init()
          await stateExecute({ subcommand: "init" }, ctx)
        },
      })
      const state = await readForecastPhaseState(tmp.path)
      await writeForecastPhaseState(tmp.path, {
        ...state,
        phase: "post-forecast",
        prediction_prior: {
          ...state.prediction_prior,
          enabled: true,
          user_decision: "enabled",
          prior_dataset_path: path.join(tmp.path, "prior.csv"),
          prior_dataset_sha256: "a".repeat(64),
          validation_run_id: "validation-run",
          test_materialized: true,
          test_run_id: "stale-test-run",
        },
      })

      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const tool = await ForecastPredictionPriorTool.init()
          const result = JSON.parse((await tool.execute({ subcommand: "materialize_test" }, ctx)).output)
          expect(result.materialized).toBe(true)
          expect(result.test_run_id).toBe("fresh-test-run")
        },
      })

      expect(spawnSpy).toHaveBeenCalledTimes(1)
      const updated = await readForecastPhaseState(tmp.path)
      expect(updated.prediction_prior.test_run_id).toBe("fresh-test-run")
    } finally {
      spawnSpy.mockRestore()
    }
  })

  test("forecast_state append writes to history.jsonl", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastStateTool.init()
        await tool.execute(
          {
            subcommand: "append",
            record: record("run-append", 1.25),
          },
          ctx,
        )
      },
    })

    const historyPath = path.join(tmp.path, ".forecast", "history.jsonl")
    const history = await readFile(historyPath, "utf-8")
    expect(history).toContain("run-append")
  })

  test("forecast_state resume reconstructs state", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastStateTool.init()
        await tool.execute({ subcommand: "append", record: record("run-a", 3) }, ctx)
        await tool.execute({ subcommand: "append", record: record("run-b", 1) }, ctx)
        await tool.execute({ subcommand: "append", record: record("run-c", 2) }, ctx)

        const result = await tool.execute({ subcommand: "resume" }, ctx)
        const state = JSON.parse(result.output)

        expect(state.experimentCount).toBe(3)
        expect(state.completedRunIds).toEqual(["run-a", "run-b", "run-c"])
        expect(state.bestResult.run_id).toBe("run-b")
      },
    })
  })

  test("forecast_state budget increments", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const tool = await ForecastStateTool.init()
        await tool.execute({ subcommand: "budget", budget_increment: { elapsed_seconds: 10 } }, ctx)
        await tool.execute({ subcommand: "budget", budget_increment: { elapsed_seconds: 5 } }, ctx)
      },
    })

    const budgetPath = path.join(tmp.path, ".forecast", "budget.json")
    const budget = JSON.parse(await readFile(budgetPath, "utf-8"))
    expect(budget.experiment_count).toBe(2)
    expect(budget.total_elapsed_seconds).toBe(15)
  })

  test("post-forecast transition requires a generative prediction reasoning start decision", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)
    await enterForecastPhase(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()

        const gateError = await execute({ subcommand: "transition", target_phase: "post-forecast" }, ctx).catch(
          (error) => error,
        )
        expect(PhaseGateError.isInstance(gateError)).toBe(true)
        if (!PhaseGateError.isInstance(gateError)) {
          throw gateError
        }
        expect(gateError.data.missing_preconditions).toContain(
          "ask the user to choose whether to disable generative prediction reasoning, use lightweight prediction reasoning, or use deep prediction reasoning with forecast_adjustment_decision ask_start before entering post-forecast",
        )

        await recordAdjustmentDisabled(tmp.path)
        const result = await execute({ subcommand: "transition", target_phase: "post-forecast" }, ctx)
        const parsed = JSON.parse(result.output)
        const state = await readForecastPhaseState(tmp.path)

        expect(parsed.ok).toBe(true)
        expect(state.phase).toBe("post-forecast")
        expect(state.adjustment.user_decision).toBe("disabled")
      },
    })
  })

  test("post-forecast transition requires an audited resolution for a frozen user expert prior", async () => {
    await using tmp = await tmpdir()
    await seedForecastPythonProject(tmp.path)
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const taskTool = await ForecastTaskTool.init()
        await taskTool.execute(
          {
            ...validTaskInput,
            adjustment_skill_priors: "Always increase the forecast by an unspecified amount.",
          },
          ctx,
        )
      },
    })
    await enterForecastPhase(tmp.path)

    const forecastState = await readForecastPhaseState(tmp.path)
    await writeForecastPhaseState(tmp.path, {
      ...forecastState,
      adjustment: {
        ...forecastState.adjustment,
        enabled: true,
        mode: "rule_skill",
        strategy: "b_comprehensive",
        user_decision: "enabled_b",
        user_decision_at: "2026-07-16T00:00:00.000Z",
        source_model_set_locked: true,
        source_model_set_path: ".forecast/adjustments/source-model-set.json",
        fit_runs_ready: true,
        fit_run_manifest_path: ".forecast/adjustments/fit-runs/manifest.json",
        forecaster_locked: true,
        no_effective_skill: true,
        no_effective_skill_reason: "No candidate retained.",
      },
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()
        const unresolved = await execute({ subcommand: "transition", target_phase: "post-forecast" }, ctx).catch(
          (error) => error,
        )
        expect(PhaseGateError.isInstance(unresolved)).toBe(true)
        if (!PhaseGateError.isInstance(unresolved)) throw unresolved
        expect(unresolved.data.missing_preconditions).toContain(
          "task contains adjustment_skill_priors; independently inspect, write, validate, and accept or reject at least one user_prior_mirror candidate before entering post-forecast",
        )
      },
    })

    const writeArtifact = async (file: string, value: unknown) => {
      await mkdir(path.dirname(file), { recursive: true })
      await writeFile(file, `${JSON.stringify(value, null, 2)}\n`, "utf-8")
    }
    await writeArtifact(path.join(tmp.path, ".forecast", "adjustments", "source-model-set.json"), {
      source_model_set_id: "source-set",
      created_at: "2026-07-16T00:00:00.000Z",
      created_from: "post_raw_forecast_search",
      members: [
        {
          family: "deep_learning",
          validation_run_id: "run-val",
          model: "DLinear",
          spec_hash: "validation-spec-hash",
          validation_metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
        },
      ],
    })
    await writeArtifact(
      path.join(tmp.path, ".forecast", "adjustments", "candidates", "user-prior", "candidate.json"),
      {
        candidate_id: "user-prior",
        source_model_set_id: "source-set",
        parent_skill_version: null,
        status: "rejected",
        origin: "user_prior_mirror",
        user_prior_contract: {
          evidence_path: `.forecast/adjustments/user-prior/evidence-${"a".repeat(64)}.json`,
          source_sha256: "b".repeat(64),
          validation_front_match: "none",
          rule_mappings: [
            {
              rule_id: "rare-up",
              source_excerpt: "Always increase the forecast by an unspecified amount.",
            },
          ],
          unresolved_parameters_before_llm: [{ rule_id: "rare-up", field: "action.value" }],
          parameter_completions: [
            {
              rule_id: "rare-up",
              field: "condition.kind",
              value: "always",
              source: "explicit_prompt",
              rationale: "Always is explicit in the frozen expert prior.",
            },
            {
              rule_id: "rare-up",
              field: "action.type",
              value: "add",
              source: "explicit_prompt",
              rationale: "Increase is represented as an additive correction.",
            },
            {
              rule_id: "rare-up",
              field: "action.value",
              value: 1,
              source: "llm_inferred",
              rationale: "Conservative missing-value completion.",
            },
          ],
        },
        rules: [
          {
            id: "rare-up",
            condition: { kind: "always" },
            action: { type: "add", value: 1 },
            rationale: "Preserve the user's rare-event upward rule.",
            evidence_refs: [".forecast/adjustments/diagnostics/fit/fit-run/diagnostics.json"],
            bounds: {},
          },
        ],
        skill_markdown: "# User prior\n",
        created_at: "2026-07-16T00:00:00.000Z",
        llm_provenance: { user_prior_mirror: true },
      },
    )
    await writeArtifact(
      path.join(
        tmp.path,
        ".forecast",
        "adjustments",
        "validation",
        "run-val",
        "user-prior",
        "adjusted_eval.json",
      ),
      {
        run_id: "run-val",
        candidate_id: "user-prior",
        split: "val",
        raw_eval_path: ".forecast/runs/run-val/eval.json",
        adjusted_predictions_path: ".forecast/adjustments/validation/run-val/user-prior/adjusted_predictions.csv",
        raw_metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
        adjusted_metrics: { mse: 1.1, mae: 0.55, wape: 0.275, mase: null },
        delta_vs_raw: { mse: 0.1, mae: 0.05, wape: 0.025, mase: null },
        acceptance_decision: {
          accepted: false,
          reason: "rejected: held-out validation MSE regressed",
          checks: { mse_non_regression: false },
        },
        timestamp: "2026-07-16T00:00:00.000Z",
      },
    )
    const auditPath = path.join(
      tmp.path,
      ".forecast",
      "adjustments",
      "validation",
      "run-val",
      "user-prior",
      "leakage-audit.jsonl",
    )
    await mkdir(path.dirname(auditPath), { recursive: true })
    await writeFile(auditPath, `${JSON.stringify({ check: "blind_input_forbidden_keys", passed: true })}\n`, "utf-8")

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()
        const result = await execute({ subcommand: "transition", target_phase: "post-forecast" }, ctx)
        expect(JSON.parse(result.output).ok).toBe(true)
      },
    })
    expect((await readForecastPhaseState(tmp.path)).phase).toBe("post-forecast")
  })

  test("post-forecast transition enforces the default reflection policy and matching validation record", async () => {
    await using tmp = await tmpdir()
    await seedForecastPythonProject(tmp.path)
    const { reflection_enabled: _reflectionEnabled, ...taskWithoutReflectionSetting } = validTaskInput
    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const taskTool = await ForecastTaskTool.init()
        await taskTool.execute(taskWithoutReflectionSetting, ctx)
      },
    })
    await enterForecastPhase(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()
        await execute({ subcommand: "append", record: record("reflection-validation", 0.5) }, ctx)
        await recordAdjustmentDisabled(tmp.path)

        const missingPolicy = await execute({ subcommand: "transition", target_phase: "post-forecast" }, ctx).catch(
          (error) => error,
        )
        expect(PhaseGateError.isInstance(missingPolicy)).toBe(true)
        if (!PhaseGateError.isInstance(missingPolicy)) throw missingPolicy
        expect(missingPolicy.data.missing_preconditions).toContain(
          "reflection_enabled is true; freeze the independent window-reflection policy before entering post-forecast",
        )

        const configured = await readForecastPhaseState(tmp.path)
        await mkdir(path.join(tmp.path, ".forecast", "reflection"), { recursive: true })
        const reflectionPolicyPath = path.join(tmp.path, ".forecast", "reflection", "policy.json")
        await writeFile(
          reflectionPolicyPath,
          `${JSON.stringify({
            controller_version: "1",
            enabled: true,
            max_reflection_rounds: 3,
            max_format_retries: 2,
            trigger_policy: { hard_violation_count: 1, independent_soft_violation_count: 2 },
            constraints: {
              nonnegative: true,
              absolute_min: 0,
              absolute_max: 100,
              soft_abs_step: 10,
              max_abs_step: 20,
              window_mean_abs_delta_limit: 10,
              point_emergency_delta_limit: 40,
            },
            validation_gate: {
              metric: "mse",
              accept_if: "final_mse <= baseline_mse",
              zero_trigger: "enabled_no_trigger",
            },
          })}\n`,
          "utf-8",
        )
        await writeForecastPhaseState(tmp.path, {
          ...configured,
          reflection: {
            ...configured.reflection,
            configured: true,
            policy_path: ".forecast/reflection/policy.json",
            policy_sha256: await computeFileSha256(reflectionPolicyPath),
          },
        })
        const missingValidation = await execute({ subcommand: "transition", target_phase: "post-forecast" }, ctx).catch(
          (error) => error,
        )
        expect(PhaseGateError.isInstance(missingValidation)).toBe(true)
        if (!PhaseGateError.isInstance(missingValidation)) throw missingValidation
        expect(missingValidation.data.missing_preconditions[0]).toContain("reflection-validation")

        const withPolicy = await readForecastPhaseState(tmp.path)
        await writeForecastPhaseState(tmp.path, {
          ...withPolicy,
          reflection: {
            ...withPolicy.reflection,
            validation_records: [
              {
                validation_run_id: "reflection-validation",
                baseline_kind: "raw_anchor",
                adjustment_candidate_id: null,
                status: "enabled_no_trigger",
                trigger_count: 0,
                corrected_count: 0,
                fallback_count: 0,
                baseline_metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
                final_metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
                delta_mse: 0,
                result_path: ".forecast/reflection/validation/reflection-validation/raw/reflection_eval.json",
                completed_at: "2026-07-15T00:00:00.000Z",
              },
            ],
          },
        })

        const missingArtifact = await execute({ subcommand: "transition", target_phase: "post-forecast" }, ctx).catch(
          (error) => error,
        )
        expect(PhaseGateError.isInstance(missingArtifact)).toBe(true)
        if (!PhaseGateError.isInstance(missingArtifact)) throw missingArtifact
        expect(missingArtifact.data.missing_preconditions[0]).toContain("reflection-validation")

        const resultDir = path.join(tmp.path, ".forecast", "reflection", "validation", "reflection-validation", "raw")
        await mkdir(resultDir, { recursive: true })
        const reflectionPredictionsPath = path.join(resultDir, "final_predictions.csv")
        const reflectionAuditPath = path.join(resultDir, "leakage-audit.json")
        await writeFile(
          reflectionPredictionsPath,
          "window_index,horizon_index,timestamp,actual,predicted,baseline_predicted\n0,0,2026-01-01,1,1,1\n",
          "utf-8",
        )
        await writeFile(reflectionAuditPath, '{"passed":true}\n', "utf-8")
        const reflectionArtifact = {
            controller_version: "1",
            source_run_id: "reflection-validation",
            split: "val",
            baseline_kind: "raw_anchor",
            adjustment_candidate_id: null,
            status: "enabled_no_trigger",
            trigger_count: 0,
            corrected_count: 0,
            fallback_count: 0,
            baseline_metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
            final_metrics: { mse: 1, mae: 0.5, wape: 0.25, mase: null },
            delta_mse: 0,
            final_predictions_path:
              ".forecast/reflection/validation/reflection-validation/raw/final_predictions.csv",
            final_predictions_sha256: await computeFileSha256(reflectionPredictionsPath),
            leakage_audit_sha256: await computeFileSha256(reflectionAuditPath),
            completed_at: "2026-07-15T00:00:00.000Z",
          }
        await writeFile(
          path.join(resultDir, "reflection_eval.json"),
          `${JSON.stringify(reflectionArtifact)}\n`,
          "utf-8",
        )

        const result = await execute({ subcommand: "transition", target_phase: "post-forecast" }, ctx)
        expect(JSON.parse(result.output).ok).toBe(true)
      },
    })
  })

  test("post-forecast transition publishes final skills to global Forecast memory", async () => {
    await using tmp = await tmpdir()
    const previousMemoryDir = process.env.ADACAST_FORECAST_MEMORY_DIR
    process.env.ADACAST_FORECAST_MEMORY_DIR = path.join(tmp.path, "global-forecast-memory")

    try {
      await defineTask(tmp.path)
      await enterForecastPhase(tmp.path)
      const skillDir = path.join(tmp.path, ".forecast", "skills", "linear-models")
      await mkdir(skillDir, { recursive: true })
      await writeFile(
        path.join(skillDir, "SKILL.md"),
        [
          "---",
          "name: linear-models",
          "description: Final linear model guidance",
          "---",
          "",
          "# Linear Models",
          "",
          "## Parameter Space",
          "- train_epochs: 3",
          "",
        ].join("\n"),
        "utf-8",
      )

      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const { execute } = await ForecastStateTool.init()
          await recordAdjustmentDisabled(tmp.path)

          const result = await execute({ subcommand: "transition", target_phase: "post-forecast" }, ctx)
          const parsed = JSON.parse(result.output)
          const state = await readForecastPhaseState(tmp.path)

          expect(parsed.ok).toBe(true)
          expect(state.skill_memory_published_count).toBe(1)
          expect(state.skill_memory_published_at).toEqual(expect.any(String))
          expect(state.skill_memory_manifest_path).toEqual(expect.any(String))

          const manifest = JSON.parse(await readFile(state.skill_memory_manifest_path!, "utf-8"))
          expect(manifest.count).toBe(1)
          expect(manifest.skills[0].original_skill_name).toBe("linear-models")
          expect(await readFile(manifest.skills[0].path, "utf-8")).toContain("Final linear model guidance")
        },
      })
    } finally {
      if (previousMemoryDir === undefined) {
        delete process.env.ADACAST_FORECAST_MEMORY_DIR
      } else {
        process.env.ADACAST_FORECAST_MEMORY_DIR = previousMemoryDir
      }
    }
  })

  test("post-forecast transition continues when global Forecast memory publish fails", async () => {
    await using tmp = await tmpdir()
    const previousMemoryDir = process.env.ADACAST_FORECAST_MEMORY_DIR
    const blockedMemoryRoot = path.join(tmp.path, "blocked-memory-root")
    process.env.ADACAST_FORECAST_MEMORY_DIR = blockedMemoryRoot

    try {
      await defineTask(tmp.path)
      await enterForecastPhase(tmp.path)
      await writeFile(blockedMemoryRoot, "not a directory", "utf-8")

      const skillDir = path.join(tmp.path, ".forecast", "skills", "linear-models")
      await mkdir(skillDir, { recursive: true })
      await writeFile(
        path.join(skillDir, "SKILL.md"),
        [
          "---",
          "name: linear-models",
          "description: Final linear model guidance",
          "---",
          "",
          "# Linear Models",
          "",
          "## Parameter Space",
          "- train_epochs: 3",
          "",
        ].join("\n"),
        "utf-8",
      )

      await Instance.provide({
        directory: tmp.path,
        fn: async () => {
          const { execute } = await ForecastStateTool.init()
          await recordAdjustmentDisabled(tmp.path)

          const result = await execute({ subcommand: "transition", target_phase: "post-forecast" }, ctx)
          const parsed = JSON.parse(result.output)
          const state = await readForecastPhaseState(tmp.path)

          expect(parsed.ok).toBe(true)
          expect(state.phase).toBe("post-forecast")
          expect(state.skill_memory_published_count).toBe(0)
          expect(state.skill_memory_published_at).toBeNull()
          expect(state.skill_memory_manifest_path).toBeNull()
          expect(state.skill_memory_publish_error).toEqual(expect.any(String))
        },
      })
    } finally {
      if (previousMemoryDir === undefined) {
        delete process.env.ADACAST_FORECAST_MEMORY_DIR
      } else {
        process.env.ADACAST_FORECAST_MEMORY_DIR = previousMemoryDir
      }
    }
  })

  test("post-forecast transition points Auto Mode to the LLM auto_start decision", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)
    await enterForecastPhase(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()
        await execute(
          {
            subcommand: "configure_auto_mode",
            auto_mode_update: {
              enabled: true,
              active_agent: "forecaster",
              state: "running",
            },
          },
          ctx,
        )

        const gateError = await execute({ subcommand: "transition", target_phase: "post-forecast" }, ctx).catch(
          (error) => error,
        )
        expect(PhaseGateError.isInstance(gateError)).toBe(true)
        if (!PhaseGateError.isInstance(gateError)) {
          throw gateError
        }
        expect(gateError.data.missing_preconditions).toContain(
          "record the Auto Mode LLM generative prediction reasoning decision (disabled, lightweight prediction reasoning, or deep prediction reasoning) with forecast_adjustment_decision auto_start before entering post-forecast",
        )
      },
    })
  })
})

describe("forecast_state update_stop_counters", () => {
  test("append ignores an already recorded deterministic run id", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)
    await enterForecastPhase(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()
        const experiment = record("run-duplicate", 1.0)

        expect(JSON.parse((await execute({ subcommand: "append", record: experiment }, ctx)).output)).toMatchObject({
          appended: true,
        })
        expect(JSON.parse((await execute({ subcommand: "append", record: experiment }, ctx)).output)).toMatchObject({
          appended: false,
          reason: "run_id_already_recorded",
        })

        const lines = (await readFile(path.join(tmp.path, ".forecast", "history.jsonl"), "utf-8")).trim().split("\n")
        expect(lines).toHaveLength(1)
      },
    })
  })

  test("increments no-improvement counter on non-improving experiment", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)
    await enterForecastPhase(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()

        await execute({ subcommand: "append", record: record("run-1", 1.0) }, ctx)
        const result = await execute(
          { subcommand: "update_stop_counters", stop_counter_input: { improvement: false, crashed: false } },
          ctx,
        )
        const parsed = JSON.parse(result.output)

        expect(parsed.counters.consecutive_no_improvement).toBe(1)
        expect(parsed.counters.consecutive_crashes).toBe(0)
        expect(parsed.stop).toBe(false)
        expect(parsed.hitl_triggered).toBe(false)
      },
    })
  })

  test("counts retry attempts for crash stopping while preserving unique-run budget", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)
    await enterForecastPhase(tmp.path)
    const historyPath = path.join(tmp.path, ".forecast", "history.jsonl")
    const crashAttempt = {
      ...record("retry-run", 0),
      status: "crash" as const,
      metrics: null,
      decision: "crash" as const,
      failure_reason: "corrupt artifact",
    }
    await writeFile(
      historyPath,
      [
        JSON.stringify({ ...crashAttempt, attempt_index: 1 }),
        JSON.stringify({ ...crashAttempt, attempt_index: 2 }),
        "",
      ].join("\n"),
      "utf-8",
    )

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()
        const first = JSON.parse(
          (
            await execute(
              {
                subcommand: "update_stop_counters",
                stop_counter_input: { run_id: "retry-run", improvement: false, crashed: true },
              },
              ctx,
            )
          ).output,
        )
        expect(first.counters).toMatchObject({
          consecutive_crashes: 2,
          experiments_remaining: 19,
        })

        const second = JSON.parse(
          (
            await execute(
              {
                subcommand: "update_stop_counters",
                stop_counter_input: { run_id: "retry-run", improvement: false, crashed: true },
              },
              ctx,
            )
          ).output,
        )
        expect(second.counters).toEqual(first.counters)

        const state = ForecastPhaseState.parse(
          normalizeForecastStateFrontmatter(
            matter(await readFile(path.join(tmp.path, ".forecast", "STATE.md"), "utf-8")).data as Record<
              string,
              unknown
            >,
          ),
        )
        expect(state.stop_counter_last_run_id).toBe("retry-run")
        expect(state.stop_counter_last_attempt_index).toBe(2)
      },
    })
  })

  test("triggers HITL when no-improvement reaches threshold", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)
    await enterForecastPhase(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()

        for (let index = 1; index <= 3; index += 1) {
          await execute({ subcommand: "append", record: record(`run-${index}`, 1.0) }, ctx)
          const result = await execute(
            { subcommand: "update_stop_counters", stop_counter_input: { improvement: false, crashed: false } },
            ctx,
          )
          const parsed = JSON.parse(result.output)

          if (index < 3) {
            expect(parsed.hitl_triggered).toBe(false)
            expect(parsed.stop).toBe(false)
          } else {
            expect(parsed.hitl_triggered).toBe(true)
            expect(parsed.stop).toBe(false)
          }
        }
      },
    })
  })

  test("returns stop=true with crash_threshold when consecutive crashes reach M", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)
    await enterForecastPhase(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()

        for (let index = 1; index <= 3; index += 1) {
          const crashRecord = {
            ...record(`crash-${index}`, 0),
            status: "crash" as const,
            metrics: null,
            decision: "crash" as const,
            failure_reason: "OOM during training",
          }

          await execute({ subcommand: "append", record: crashRecord }, ctx)
          const result = await execute(
            { subcommand: "update_stop_counters", stop_counter_input: { improvement: false, crashed: true } },
            ctx,
          )
          const parsed = JSON.parse(result.output)

          if (index < 3) {
            expect(parsed.stop).toBe(false)
          } else {
            expect(parsed.stop).toBe(true)
            expect(parsed.reason).toBe("crash_threshold")
          }
        }
      },
    })
  })

  test("returns stop=true with budget_exhausted when experiment count reaches K", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)
    await enterForecastPhase(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()
        let finalResult: Record<string, unknown> | null = null

        for (let index = 1; index <= 20; index += 1) {
          await execute({ subcommand: "append", record: record(`budget-${index}`, 1.0 / index) }, ctx)
          const result = await execute(
            { subcommand: "update_stop_counters", stop_counter_input: { improvement: true, crashed: false } },
            ctx,
          )
          finalResult = JSON.parse(result.output)
        }

        expect(finalResult).not.toBeNull()
        expect(finalResult?.stop).toBe(true)
        expect(finalResult?.reason).toBe("budget_exhausted")
      },
    })
  })
})

describe("forecast_state record_expert_feedback", () => {
  test("appends expert feedback to history and resets no-improvement counter", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)
    await enterForecastPhase(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()

        await execute({ subcommand: "append", record: record("run-1", 1.0) }, ctx)
        await execute(
          { subcommand: "update_stop_counters", stop_counter_input: { improvement: false, crashed: false } },
          ctx,
        )
        await execute({ subcommand: "append", record: record("run-2", 1.0) }, ctx)
        await execute(
          { subcommand: "update_stop_counters", stop_counter_input: { improvement: false, crashed: false } },
          ctx,
        )

        const feedbackResult = await execute(
          {
            subcommand: "record_expert_feedback",
            expert_feedback_text: "Try using longer sequence lengths for this seasonal pattern.",
          },
          ctx,
        )
        const parsed = JSON.parse(feedbackResult.output)

        expect(parsed.recorded).toBe(true)
        expect(parsed.counter_reset).toBe(true)
        expect(parsed.consecutive_no_improvement).toBe(0)

        const historyRaw = await readFile(path.join(tmp.path, ".forecast", "history.jsonl"), "utf-8")
        expect(historyRaw).toContain('"type":"expert_feedback"')
        expect(historyRaw).toContain("Try using longer sequence lengths")
      },
    })
  })
})

describe("forecast_state read enriched D-12 output", () => {
  test("returns structured context with current_best, recent_failures, budget, and expert_feedback", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)
    await enterForecastPhase(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()

        await execute({ subcommand: "append", record: record("run-best", 0.5) }, ctx)
        await execute({ subcommand: "best", record: record("run-best", 0.5), update_best: true }, ctx)

        const crashRecord = {
          ...record("run-crash", 0),
          status: "crash" as const,
          metrics: null,
          decision: "crash" as const,
          failure_reason: "OOM during training",
        }
        await execute({ subcommand: "append", record: crashRecord }, ctx)
        await execute({ subcommand: "record_expert_feedback", expert_feedback_text: "Check memory usage" }, ctx)

        const readResult = await execute({ subcommand: "read" }, ctx)
        const data = JSON.parse(readResult.output)

        expect(data.current_best).not.toBeNull()
        expect(data.current_best.run_id).toBe("run-best")
        expect(data.current_best.model).toBe("DLinear")
        expect(data.current_best.metrics.mse).toBe(0.5)

        expect(data.recent_failures).toBeInstanceOf(Array)
        expect(data.recent_failures.length).toBeGreaterThanOrEqual(1)
        expect(data.recent_failures[0].failure_reason).toBe("OOM during training")

        expect(data.budget).toBeDefined()
        expect(data.budget.total).toBe(20)
        expect(data.budget.remaining).toBeLessThanOrEqual(20)
        expect(typeof data.budget.consecutive_no_improvement).toBe("number")
        expect(typeof data.budget.consecutive_crashes).toBe("number")

        expect(data.expert_feedback).not.toBeNull()
        expect(data.expert_feedback.text).toBe("Check memory usage")
      },
    })
  })
})

describe("forecast_state CAST.md threshold override", () => {
  test("reads thresholds from CAST.md when present", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)

    const { writeFile } = await import("fs/promises")
    await writeFile(
      path.join(tmp.path, "CAST.md"),
      "# Constraints\n\nmax_experiments: 50\nno_improvement_rounds: 5\ncrash_threshold: 10\n",
      "utf-8",
    )

    await enterForecastPhase(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()
        const readResult = await execute({ subcommand: "read" }, ctx)
        const data = JSON.parse(readResult.output)

        expect(data.budget.total).toBe(50)
      },
    })
  })
})

describe("forecast_state auto mode", () => {
  test("persists auto_mode.enabled, active_agent, and state in STATE.md", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()

        await execute({ subcommand: "init" }, ctx)
        await execute(
          {
            subcommand: "configure_auto_mode",
            auto_mode_update: {
              enabled: true,
              active_agent: "planner",
              state: "clarifying",
              auto_approve_permissions: true,
              auto_approve_skills: true,
            },
          },
          ctx,
        )
      },
    })

    const statePath = path.join(tmp.path, ".forecast", "STATE.md")
    const parsed = matter(await readFile(statePath, "utf-8"))
    const data = parsed.data as {
      auto_mode: {
        enabled: boolean
        active_agent: string | null
        state: string
        auto_approve_permissions: boolean
        auto_approve_skills: boolean
      }
    }

    expect(data.auto_mode.enabled).toBe(true)
    expect(data.auto_mode.active_agent).toBe("planner")
    expect(data.auto_mode.state).toBe("clarifying")
    expect(data.auto_mode.auto_approve_permissions).toBe(true)
    expect(data.auto_mode.auto_approve_skills).toBe(true)
  })

  test("forecast_state read returns auto mode fields without changing budget or current_best structure", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()

        await execute({ subcommand: "init" }, ctx)
        await execute({ subcommand: "best", record: record("run-auto-mode", 0.25), update_best: true }, ctx)
        await execute(
          {
            subcommand: "configure_auto_mode",
            auto_mode_update: {
              enabled: true,
              active_agent: "planner",
              state: "clarifying",
            },
          },
          ctx,
        )

        const readResult = await execute({ subcommand: "read" }, ctx)
        const data = JSON.parse(readResult.output)

        expect(data.auto_mode.enabled).toBe(true)
        expect(data.auto_mode.active_agent).toBe("planner")
        expect(data.auto_mode.state).toBe("clarifying")
        expect(data.budget).toEqual(
          expect.objectContaining({
            total: expect.any(Number),
            remaining: expect.any(Number),
            consecutive_no_improvement: expect.any(Number),
            consecutive_crashes: expect.any(Number),
          }),
        )
        expect(data.current_best).toEqual(
          expect.objectContaining({
            run_id: "run-auto-mode",
            model: "DLinear",
            metrics: expect.objectContaining({
              mse: 0.25,
              mae: expect.any(Number),
              wape: expect.any(Number),
              mase: expect.any(Number),
            }),
          }),
        )
      },
    })
  })

  test("queues a Forecaster continuation when Auto Mode enters forecast phase", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const sessionID = await createAutoModeSession("planner")
        const { execute } = await ForecastStateTool.init()
        const plannerCtx = toolCtx({ sessionID, agent: "planner" })

        await execute({ subcommand: "init" }, plannerCtx)
        await execute(
          {
            subcommand: "configure_auto_mode",
            auto_mode_update: {
              enabled: true,
              active_agent: "planner",
              state: "skills-review",
              auto_approve_permissions: true,
              auto_approve_skills: true,
            },
          },
          plannerCtx,
        )
        await recordPredictionPriorDeclined(tmp.path)
        await execute({ subcommand: "transition", target_phase: "pre-forecast" }, plannerCtx)
        await writePreForecastReport(tmp.path)
        await execute({ subcommand: "transition", target_phase: "skills-review" }, plannerCtx)
        await execute({ subcommand: "confirm_skills" }, plannerCtx)

        const result = await execute({ subcommand: "transition", target_phase: "forecast" }, plannerCtx)
        const parsed = JSON.parse(result.output)
        const state = await readForecastPhaseState(tmp.path)
        const lastUser = await lastUserMessage(sessionID)

        expect(parsed.continuation_queued).toBe(true)
        expect(parsed.next_agent).toBe("forecaster")
        expect(state.auto_mode.active_agent).toBe("forecaster")
        expect(state.auto_mode.state).toBe("running")
        expect(state.auto_mode.pause_reason).toBeNull()
        expect(state.auto_mode.handoffs.at(-1)).toEqual(
          expect.objectContaining({
            from: "planner",
            to: "forecaster",
          }),
        )
        expect(lastUser?.info.role).toBe("user")
        expect(lastUser?.info.agent).toBe("forecaster")
        expect(textParts(lastUser)).toContain("Auto Mode continuation for the Forecaster agent")
        expect(textParts(lastUser)).toContain("Do not ask the user to confirm routine Forecaster steps")
        expect(textParts(lastUser)).toContain("Do not call the question tool or wait for user input")
        expect(textParts(lastUser)).toContain("forecast_adjustment_decision auto_start")
        expect(textParts(lastUser)).toContain("forecast_adjustment_decision auto_escalation")
        expect(textParts(lastUser)).not.toContain("Create the compacted handoff prompt")
        expect(textParts(lastUser)).toContain(".forecast/CONTEXT.md")
        expect(textParts(lastUser)).toContain(".forecast/task.json")
      },
    })
  })

  test("queues a Critic continuation when Auto Mode enters post-forecast phase", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const sessionID = await createAutoModeSession("planner")
        const { execute } = await ForecastStateTool.init()
        const plannerCtx = toolCtx({ sessionID, agent: "planner" })
        const forecasterCtx = toolCtx({ sessionID, agent: "forecaster" })

        await execute({ subcommand: "init" }, plannerCtx)
        await execute(
          {
            subcommand: "configure_auto_mode",
            auto_mode_update: {
              enabled: true,
              active_agent: "planner",
              state: "skills-review",
            },
          },
          plannerCtx,
        )
        await recordPredictionPriorDeclined(tmp.path)
        await execute({ subcommand: "transition", target_phase: "pre-forecast" }, plannerCtx)
        await writePreForecastReport(tmp.path)
        await execute({ subcommand: "transition", target_phase: "skills-review" }, plannerCtx)
        await execute({ subcommand: "confirm_skills" }, plannerCtx)
        await execute({ subcommand: "transition", target_phase: "forecast" }, plannerCtx)
        await execute({ subcommand: "append", record: record("critic-best", 0.42) }, forecasterCtx)
        await execute({ subcommand: "best", record: record("critic-best", 0.42), update_best: true }, forecasterCtx)
        await recordAdjustmentDisabled(tmp.path)

        const result = await execute({ subcommand: "transition", target_phase: "post-forecast" }, forecasterCtx)
        const parsed = JSON.parse(result.output)
        const state = await readForecastPhaseState(tmp.path)
        const lastUser = await lastUserMessage(sessionID)

        expect(parsed.continuation_queued).toBe(true)
        expect(parsed.next_agent).toBe("critic")
        expect(parsed.compaction_queued).toBe(false)
        expect(state.auto_mode.active_agent).toBe("critic")
        expect(state.auto_mode.state).toBe("running")
        expect(state.auto_mode.pause_reason).toBeNull()
        expect(state.auto_mode.handoffs.at(-1)).toEqual(
          expect.objectContaining({
            from: "forecaster",
            to: "critic",
          }),
        )
        expect(lastUser?.info.role).toBe("user")
        expect(lastUser?.info.agent).toBe("critic")
        expect(textParts(lastUser)).toContain("Auto Mode continuation for the Critic agent")
        expect(textParts(lastUser)).toContain("Do not ask the user to confirm routine Critic steps")
        expect(textParts(lastUser)).toContain("Do not call the question tool or wait for user input")
        expect(textParts(lastUser)).toContain("forecast_adjustment_decision auto_test_application")
        expect(textParts(lastUser)).not.toContain("Create the compacted handoff prompt")
        expect(textParts(lastUser)).toContain(".forecast/best.json")
        expect(textParts(lastUser)).toContain("critic-best")
      },
    })
  })

  test("keeps Auto Mode running when no-improvement triggers its automatic stop path", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)
    await enterForecastPhase(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()

        await execute(
          {
            subcommand: "configure_auto_mode",
            auto_mode_update: {
              enabled: true,
              active_agent: "forecaster",
              state: "running",
            },
          },
          ctx,
        )

        for (let index = 1; index <= 3; index += 1) {
          await execute({ subcommand: "append", record: record(`auto-hitl-${index}`, 1.0) }, ctx)
          await execute(
            { subcommand: "update_stop_counters", stop_counter_input: { improvement: false, crashed: false } },
            ctx,
          )
        }

        const state = await readForecastPhaseState(tmp.path)
        expect(state.auto_mode.state).toBe("running")
        expect(state.auto_mode.pause_reason).toBeNull()
      },
    })
  })

  test("pauses Auto Mode with a concrete reason when a hard stop is reached", async () => {
    await using tmp = await tmpdir()
    await defineTask(tmp.path)
    await enterForecastPhase(tmp.path)

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const { execute } = await ForecastStateTool.init()

        await execute(
          {
            subcommand: "configure_auto_mode",
            auto_mode_update: {
              enabled: true,
              active_agent: "forecaster",
              state: "running",
            },
          },
          ctx,
        )

        for (let index = 1; index <= 3; index += 1) {
          await execute(
            {
              subcommand: "append",
              record: {
                ...record(`auto-crash-${index}`, 0),
                status: "crash" as const,
                metrics: null,
                decision: "crash" as const,
                failure_reason: "OOM during training",
              },
            },
            ctx,
          )
          await execute(
            { subcommand: "update_stop_counters", stop_counter_input: { improvement: false, crashed: true } },
            ctx,
          )
        }

        const state = await readForecastPhaseState(tmp.path)
        expect(state.auto_mode.state).toBe("paused")
        expect(state.auto_mode.pause_reason).toContain("crash_threshold")
        expect(state.auto_mode.pause_reason).toContain("manual recovery")
      },
    })
  })
})

describe("forecast cache diagnostics", () => {
  test("parses structured cache stderr events", () => {
    const events = [
      {
        type: "cache_config",
        cache_root: ".forecast/model-cache",
        hub_cache: ".forecast/model-cache/hub",
        xet_cache: ".forecast/model-cache/xet",
        assets_cache: ".forecast/model-cache/assets",
        transformers_cache: null,
        endpoint: "https://hf-mirror.example",
        source: "project_default",
        warnings: [],
        elapsed_s: 0.01,
      },
      {
        type: "cache_hit",
        repo_id: "amazon/chronos-bolt-base",
        elapsed_s: 0.02,
      },
      {
        type: "cache_miss",
        repo_id: "amazon/chronos-bolt-base",
        elapsed_s: 0.03,
      },
      {
        type: "cache_warning",
        code: "project_cache_unwritable",
        message: "Project model cache .forecast/model-cache is not writable; using Hugging Face defaults.",
        elapsed_s: 0.04,
      },
      {
        type: "dependency_missing",
        module: "uni2ts",
        package: "uni2ts",
        guidance:
          "Install heavy model dependencies with: uv run --project python --extra heavy-models python -m adacast_ml.runner ...",
        elapsed_s: 0.05,
      },
    ].map((event) => StderrEvent.parse(event))

    expect(events.map((event) => event.type)).toEqual([
      "cache_config",
      "cache_hit",
      "cache_miss",
      "cache_warning",
      "dependency_missing",
    ])
    expect(["HF_ENDPOINT", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE", "TRANSFORMERS_CACHE"]).toHaveLength(5)
  })

  test("summarizes cache diagnostics for compact generate_model output", () => {
    const events = [
      StderrEvent.parse({
        type: "cache_config",
        cache_root: ".forecast/model-cache",
        hub_cache: ".forecast/model-cache/hub",
        xet_cache: ".forecast/model-cache/xet",
        assets_cache: ".forecast/model-cache/assets",
        transformers_cache: null,
        endpoint: "https://hf-mirror.example",
        source: "project_default",
        warnings: [],
        elapsed_s: 0.01,
      }),
      StderrEvent.parse({ type: "cache_hit", repo_id: "amazon/chronos-bolt-base", elapsed_s: 0.02 }),
      StderrEvent.parse({ type: "cache_miss", repo_id: "amazon/chronos-bolt-base", elapsed_s: 0.03 }),
      StderrEvent.parse({
        type: "cache_warning",
        code: "project_cache_unwritable",
        message: "Project model cache .forecast/model-cache is not writable; using Hugging Face defaults.",
        elapsed_s: 0.04,
      }),
      StderrEvent.parse({
        type: "dependency_missing",
        module: "uni2ts",
        package: "uni2ts",
        guidance:
          "Install heavy model dependencies with: uv run --project python --extra heavy-models python -m adacast_ml.runner ...",
        elapsed_s: 0.05,
      }),
    ]

    const cache_summary = summarizeCacheDiagnostics(events, ".forecast/runs/run-1/train.log")

    expect(cache_summary?.source).toBe("project_default")
    expect(cache_summary?.endpoint).toBe("https://hf-mirror.example")
    expect(cache_summary?.cache_root).toBe(".forecast/model-cache")
    expect(cache_summary?.hits).toBe(1)
    expect(cache_summary?.misses).toBe(1)
    expect(cache_summary?.warning_count).toBe(1)
    expect(cache_summary?.dependency_missing_count).toBe(1)
    expect(cache_summary?.log_path).toBe(".forecast/runs/run-1/train.log")
    expect(cache_summary?.warnings).toEqual([
      {
        code: "project_cache_unwritable",
        message: "Project model cache .forecast/model-cache is not writable; using Hugging Face defaults.",
      },
    ])
    expect(cache_summary?.dependencies_missing).toEqual([
      {
        module: "uni2ts",
        package: "uni2ts",
        guidance:
          "Install heavy model dependencies with: uv run --project python --extra heavy-models python -m adacast_ml.runner ...",
      },
    ])
  })
})
