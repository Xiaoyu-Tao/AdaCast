import { readFile } from "fs/promises"
import path from "path"
import z from "zod"
import { Instance } from "@/project/instance"
import { Question } from "@/question"
import { Filesystem } from "@/util/filesystem"
import { Tool } from "../tool"
import DESCRIPTION from "./forecast-prediction-prior.txt"
import { resolveForecastPythonProject } from "./python-project"
import {
  AdjustmentFitRunManifest,
  AdjustmentPolicy,
  AdjustmentSourceModelSet,
  EvalResult,
  ForecastPredictionPriorState,
  parseStoredAdjustmentPolicy,
} from "./types"
import {
  adjustmentRoot,
  ensureAdjustmentDirs,
  forecastPath,
  forecastRoot,
  readJsonFile,
  readPhaseState,
  writeJsonFile,
  writePhaseState,
} from "./forecast-adjustment-common"
import { assertFrozenTaskIntegrity } from "./forecast-task-integrity"

const parameters = z.object({
  subcommand: z.enum(["read", "ask_start", "ask_path", "import", "materialize_test"]),
  prior_dataset_path: z.string().optional().describe("Absolute path to the user-submitted prediction-prior CSV"),
})

const PRIOR_OPTION_SKIP = "不提交预测先验"
const PRIOR_OPTION_SUBMIT = "提交预测先验"

const PriorRun = z.object({
  run_id: z.string().min(1),
  spec_hash: z.string().min(1),
  run_path: z.string().min(1),
  spec_path: z.string().min(1),
  predictions_path: z.string().min(1),
  eval_path: z.string().min(1),
  metrics: EvalResult.shape.metrics,
  model_artifact_sha256: z.string().length(64),
})

const PriorImportResult = z.object({
  status: z.literal("success"),
  action: z.literal("import"),
  prior_dataset_path: z.string().min(1),
  prior_dataset_sha256: z.string().length(64),
  source_model_set: AdjustmentSourceModelSet,
  fit_run_manifest: AdjustmentFitRunManifest,
  user_prior_manifest: z.record(z.string(), z.unknown()),
  validation_run: PriorRun,
  fit_run: PriorRun,
})

const PriorTestResult = z.object({
  status: z.literal("success"),
  action: z.literal("materialize-test"),
  prior_dataset_path: z.string().min(1),
  prior_dataset_sha256: z.string().length(64),
  test_run: PriorRun,
  user_prior_manifest: z.record(z.string(), z.unknown()),
})

function taskPath(): string {
  return path.join(forecastRoot(), "task.json")
}

function policyPath(): string {
  return path.join(adjustmentRoot(), "policy.json")
}

function userPriorManifestPath(): string {
  return path.join(forecastRoot(), "user-prior", "manifest.json")
}

async function assertTaskFrozen(): Promise<void> {
  if (!(await Filesystem.exists(taskPath()))) {
    throw new Error("forecast_prediction_prior requires .forecast/task.json. Freeze the task first with forecast_task.")
  }
  await assertFrozenTaskIntegrity(forecastRoot())
}

async function readPolicy(): Promise<z.infer<typeof AdjustmentPolicy>> {
  try {
    return parseStoredAdjustmentPolicy(await readJsonFile(policyPath()))
  } catch {
    return AdjustmentPolicy.parse({})
  }
}

async function runPredictionPriorPython(action: "import" | "materialize-test", args: string[]) {
  const pythonDir = await resolveForecastPythonProject(Instance.directory)
  const proc = Bun.spawn(
    ["uv", "run", "--project", pythonDir, "python", "-m", "adacast_ml.prediction_prior", "--action", action, ...args],
    {
      cwd: Instance.directory,
      stdout: "pipe",
      stderr: "pipe",
    },
  )
  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ])
  const exitCode = await proc.exited
  let parsed: unknown = null
  try {
    parsed = JSON.parse(stdout)
  } catch {
    parsed = null
  }
  if (exitCode !== 0) {
    const parsedError =
      parsed && typeof parsed === "object" && "error" in parsed ? String((parsed as { error: unknown }).error) : null
    const stderrTail = stderr
      .split("\n")
      .filter(Boolean)
      .slice(-10)
      .join("\n")
    throw new Error(parsedError ?? (stderrTail || `adacast_ml.prediction_prior failed with exit code ${exitCode}`))
  }
  if (!parsed) throw new Error("adacast_ml.prediction_prior returned invalid JSON")
  return parsed
}

function evaluatorHashStore(): string {
  return path.join(forecastRoot(), "evaluator.sha256")
}

async function askSingle(ctx: Parameters<typeof Question.ask>[0]): Promise<string> {
  const answers = await Question.ask(ctx)
  const answer = answers[0]?.[0]?.trim()
  if (!answer) throw new Error("Question returned no answer")
  return answer
}

async function askPriorPath(ctx: Parameters<typeof Question.ask>[0], previousError?: string): Promise<string> {
  const answer = await askSingle({
    ...ctx,
    questions: [
      {
        header: "预测先验路径",
        question: [
          previousError ? `上一次校验未通过：${previousError}` : "",
          "请输入预测先验 CSV 的绝对路径。",
          "要求：这是一个完整数据集；时间列、目标列、协变量、行数、顺序和频率与真实数据集一致；训练集目标值保持真实值不变；验证集与测试集目标列替换为你的预测先验。",
          "该文件不会覆盖原真实数据集。系统会先校验文件，校验通过后直接进入 UserPrior 的深度生成式预测推理流程。",
        ]
          .filter(Boolean)
          .join("\n"),
        custom: true,
        options: [],
      },
    ],
  })
  return answer
}

async function recordDeclined(now: string, actor: "user", reason: string) {
  const state = await readPhaseState()
  await writePhaseState(
    {
      ...state,
      prediction_prior: {
        ...state.prediction_prior,
        enabled: false,
        user_decision: "declined",
        user_decision_at: now,
        user_decision_actor: actor,
        user_decision_reason: reason,
      },
    },
    `## Prediction Prior Declined\n\n${reason}`,
  )
  return { decision: "declined", prediction_prior: (await readPhaseState()).prediction_prior }
}

async function writeImportedState(result: z.infer<typeof PriorImportResult>, now: string, reason: string) {
  const currentPolicy = await readPolicy()
  const policy = AdjustmentPolicy.parse({
    ...currentPolicy,
    enabled: true,
    mode: "rule_skill",
    strategy: "b_comprehensive",
    max_candidate_iterations: Math.max(currentPolicy.max_candidate_iterations, 5),
    auto_confirm_if_validation_passes: false,
    configured_at: currentPolicy.configured_at ?? now,
  })
  await writeJsonFile(policyPath(), policy)

  const state = await readPhaseState()
  const predictionPrior = ForecastPredictionPriorState.parse({
    ...state.prediction_prior,
    enabled: true,
    user_decision: "enabled",
    user_decision_at: now,
    user_decision_actor: "user",
    user_decision_reason: reason,
    prior_dataset_path: result.prior_dataset_path,
    prior_dataset_sha256: result.prior_dataset_sha256,
    manifest_path: forecastPath("user-prior", "manifest.json"),
    source_model_set_id: result.source_model_set.source_model_set_id,
    validation_run_id: result.validation_run.run_id,
    validation_metrics: result.validation_run.metrics,
    fit_run_id: result.fit_run.run_id,
    fit_metrics: result.fit_run.metrics,
    test_run_id: null,
    test_metrics: null,
    test_materialized: false,
  })
  await writePhaseState(
    {
      ...state,
      prediction_prior: predictionPrior,
      adjustment: {
        ...state.adjustment,
        enabled: true,
        mode: "rule_skill",
        strategy: "b_comprehensive",
        user_decision: "enabled_b",
        user_decision_at: now,
        user_decision_actor: "user",
        user_decision_reason: reason,
        policy_path: forecastPath("adjustments", "policy.json"),
        source_model_set_locked: true,
        source_model_set_path: forecastPath("adjustments", "source-model-set.json"),
        fit_run_manifest_path: forecastPath("adjustments", "fit-runs", "manifest.json"),
        fit_runs_ready: true,
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
    },
    [
      "## Prediction Prior Imported",
      "",
      `UserPrior validation run: ${result.validation_run.run_id}.`,
      `UserPrior validation-front fit run: ${result.fit_run.run_id}.`,
      "Deep generative prediction reasoning is enabled automatically for this prediction-prior workflow.",
    ].join("\n"),
  )
  return {
    imported: true,
    prediction_prior: (await readPhaseState()).prediction_prior,
    adjustment: (await readPhaseState()).adjustment,
    source_model_set: result.source_model_set,
    fit_run_manifest: result.fit_run_manifest,
  }
}

async function importPriorDataset(priorDatasetPath: string) {
  await assertTaskFrozen()
  await ensureAdjustmentDirs()
  const state = await readPhaseState()
  if (state.phase === "post-forecast") {
    throw new Error("forecast_prediction_prior import is not allowed after entering post-forecast")
  }
  if (state.prediction_prior.enabled) {
    throw new Error("forecast_prediction_prior import found an already imported prediction-prior dataset")
  }
  if (state.prediction_prior.user_decision === "declined") {
    throw new Error("forecast_prediction_prior import is not allowed after the user declined prediction-prior mode")
  }
  if (state.adjustment.source_model_set_locked && !state.prediction_prior.enabled) {
    throw new Error("forecast_prediction_prior import is not allowed after ordinary adjustment source models are frozen")
  }
  const raw = await runPredictionPriorPython("import", [
    "--task-json",
    taskPath(),
    "--prior-dataset-path",
    priorDatasetPath,
    "--forecast-root",
    forecastRoot(),
    "--evaluator-hash-store",
    evaluatorHashStore(),
  ])
  const result = PriorImportResult.parse(raw)
  return writeImportedState(
    result,
    new Date().toISOString(),
    "User submitted a prediction-prior dataset; AdaCast will skip model search and use deep generative prediction reasoning on UserPrior.",
  )
}

async function askAndImportPriorDataset(ctx: Parameters<typeof Question.ask>[0]) {
  let previousError: string | undefined
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    const priorPath = await askPriorPath(ctx, previousError)
    try {
      return await importPriorDataset(priorPath)
    } catch (error) {
      previousError = error instanceof Error ? error.message : String(error)
    }
  }
  throw new Error(
    `Prediction prior dataset validation failed after 3 attempts. Last error: ${previousError ?? "unknown error"}`,
  )
}

async function materializeTestPrior() {
  await assertTaskFrozen()
  await ensureAdjustmentDirs()
  const state = await readPhaseState()
  if (state.phase !== "post-forecast") {
    throw new Error("forecast_prediction_prior materialize_test is only allowed during post-forecast")
  }
  if (!state.prediction_prior.enabled || !state.prediction_prior.prior_dataset_path) {
    throw new Error("forecast_prediction_prior materialize_test requires an imported prediction-prior dataset")
  }
  if (!state.prediction_prior.validation_run_id || !state.prediction_prior.prior_dataset_sha256) {
    throw new Error("forecast_prediction_prior materialize_test requires a validation prior source")
  }
  if (state.adjustment.test_application_records.some((record) => record.decision !== "pending")) {
    throw new Error("forecast_prediction_prior materialize_test must run before test adjustment decisions start")
  }
  // UserPrior test materialization is deterministic and cheap. Always rerun it
  // before test adjustment starts so stale artifacts from older CLIs cannot be
  // silently reused with an outdated prediction layout or split contract.
  const raw = await runPredictionPriorPython("materialize-test", [
    "--task-json",
    taskPath(),
    "--prior-dataset-path",
    state.prediction_prior.prior_dataset_path,
    "--forecast-root",
    forecastRoot(),
    "--evaluator-hash-store",
    evaluatorHashStore(),
    "--source-validation-run-id",
    state.prediction_prior.validation_run_id,
    "--expected-prior-sha256",
    state.prediction_prior.prior_dataset_sha256,
  ])
  const result = PriorTestResult.parse(raw)
  const current = await readPhaseState()
  const predictionPrior = ForecastPredictionPriorState.parse({
    ...current.prediction_prior,
    test_run_id: result.test_run.run_id,
    test_metrics: result.test_run.metrics,
    test_materialized: true,
  })
  await writePhaseState(
    {
      ...current,
      prediction_prior: predictionPrior,
    },
    `## Prediction Prior Test Materialized\n\nUserPrior test run ${result.test_run.run_id} was created during post-forecast.`,
  )
  return {
    materialized: true,
    test_run_id: result.test_run.run_id,
    test_metrics: result.test_run.metrics,
    prediction_prior: (await readPhaseState()).prediction_prior,
  }
}

async function readPriorState() {
  const state = await readPhaseState()
  const manifest = await readJsonFile(userPriorManifestPath()).catch(() => null)
  const sourceSet = await readJsonFile(path.join(adjustmentRoot(), "source-model-set.json")).catch(() => null)
  const fitManifest = await readJsonFile(path.join(adjustmentRoot(), "fit-runs", "manifest.json")).catch(() => null)
  const priorSource =
    state.prediction_prior.validation_run_id &&
    (await Filesystem.exists(
      path.join(forecastRoot(), "runs", state.prediction_prior.validation_run_id, "model", "prior-source.json"),
    ))
      ? await readFile(
          path.join(forecastRoot(), "runs", state.prediction_prior.validation_run_id, "model", "prior-source.json"),
          "utf-8",
        ).then((raw) => JSON.parse(raw))
      : null
  return {
    prediction_prior: state.prediction_prior,
    adjustment: state.adjustment,
    manifest,
    source_model_set: sourceSet,
    fit_run_manifest: fitManifest,
    prior_source: priorSource,
  }
}

export const ForecastPredictionPriorTool = Tool.define("forecast_prediction_prior", {
  description: DESCRIPTION,
  parameters,
  async execute(params, ctx) {
    let result: unknown

    switch (params.subcommand) {
      case "read": {
        result = await readPriorState()
        break
      }

      case "ask_start": {
        await assertTaskFrozen()
        const state = await readPhaseState()
        if (state.prediction_prior.user_decision !== "pending") {
          result = { already_decided: true, prediction_prior: state.prediction_prior }
          break
        }
        const answer = await askSingle({
          sessionID: ctx.sessionID,
          tool: ctx.callID ? { messageID: ctx.messageID, callID: ctx.callID } : undefined,
          questions: [
            {
              header: "预测先验",
              question: [
                "是否提交用户预测先验数据集？",
                "如果提交，请提供一份完整 CSV：时间列、目标列、协变量、行数、顺序和频率与真实数据集一致；训练集目标值保持真实值不变；验证集和测试集目标列替换为你的预测先验。",
                "该文件不会覆盖原真实数据集。提交后系统将跳过模型 Zoo 搜索和模型调参，直接对 UserPrior 启用深度生成式预测推理；若验证集后半段证明修正无效，测试阶段会保留原始用户先验。",
              ].join("\n"),
              custom: false,
              options: [
                {
                  label: PRIOR_OPTION_SKIP,
                  description: "继续原有流程，后续仍可按原设计选择是否启用生成式预测推理。",
                },
                {
                  label: PRIOR_OPTION_SUBMIT,
                  description: "提交完整先验数据集，跳过模型搜索，直接进入 UserPrior 深度预测推理。",
                },
              ],
            },
          ],
        })
        if (answer === PRIOR_OPTION_SKIP) {
          result = await recordDeclined(
            new Date().toISOString(),
            "user",
            "User chose not to submit a prediction-prior dataset; the ordinary workflow remains unchanged.",
          )
          break
        }
        if (answer !== PRIOR_OPTION_SUBMIT) {
          throw new Error(`forecast_prediction_prior ask_start returned unexpected answer "${answer}"`)
        }
        result = await askAndImportPriorDataset({
          sessionID: ctx.sessionID,
          tool: ctx.callID ? { messageID: ctx.messageID, callID: ctx.callID } : undefined,
          questions: [],
        })
        break
      }

      case "ask_path": {
        result = await askAndImportPriorDataset({
          sessionID: ctx.sessionID,
          tool: ctx.callID ? { messageID: ctx.messageID, callID: ctx.callID } : undefined,
          questions: [],
        })
        break
      }

      case "import": {
        if (!params.prior_dataset_path) {
          throw new Error("forecast_prediction_prior import requires prior_dataset_path")
        }
        result = await importPriorDataset(params.prior_dataset_path)
        break
      }

      case "materialize_test": {
        result = await materializeTestPrior()
        break
      }
    }

    return {
      title: `Forecast prediction prior: ${params.subcommand}`,
      metadata: {
        subcommand: params.subcommand,
        ...(typeof result === "object" && result && "test_run_id" in result
          ? { run_id: String((result as { test_run_id: unknown }).test_run_id) }
          : {}),
      },
      output: JSON.stringify(result),
    }
  },
})
