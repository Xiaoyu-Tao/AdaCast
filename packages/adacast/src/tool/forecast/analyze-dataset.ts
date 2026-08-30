import { mkdir, readFile, writeFile } from "fs/promises"
import path from "path"
import z from "zod"
import { Instance } from "@/project/instance"
import { Filesystem } from "@/util/filesystem"
import { Tool } from "../tool"
import DESCRIPTION from "./analyze-dataset.txt"
import { TaskNotFoundError } from "./errors"
import { TaskConfig } from "./types"
import { resolveForecastPythonProject } from "./python-project"

const parameters = z.object({})
const AnalyzeDatasetTask = z.object({
  dataset_path: TaskConfig.shape.dataset_path,
  target_col: TaskConfig.shape.target_col,
  time_col: TaskConfig.shape.time_col,
  split_mode: TaskConfig.shape.split_mode,
  train_ratio: TaskConfig.shape.train_ratio.default(0.7),
  val_ratio: TaskConfig.shape.val_ratio.default(0.1),
  test_ratio: TaskConfig.shape.test_ratio.optional(),
  train_end_timestamp: TaskConfig.shape.train_end_timestamp,
  validation_end_timestamp: TaskConfig.shape.validation_end_timestamp,
})

type AnalyzeDatasetMetadata = {
  status: "success" | "error"
  output_path?: string
  exit_code?: number
}

export const AnalyzeDatasetTool = Tool.define<typeof parameters, AnalyzeDatasetMetadata>("analyze_dataset", {
  description: DESCRIPTION,
  parameters,
  async execute() {
    const directory = Instance.directory
    const forecastRoot = path.join(directory, ".forecast")
    const taskPath = path.join(forecastRoot, "task.json")

    if (!(await Filesystem.exists(taskPath))) {
      throw new TaskNotFoundError({ expected_path: taskPath })
    }

    const task = AnalyzeDatasetTask.parse(JSON.parse(await readFile(taskPath, "utf-8")))
    const reportsDir = path.join(forecastRoot, "reports")
    const outPath = path.join(reportsDir, "quantitative.json")
    await mkdir(reportsDir, { recursive: true })

    const pythonDir = await resolveForecastPythonProject(directory)
    const analyzerScript = path.join(pythonDir, "src", "adacast_ml", "analyzer.py")
    const command = [
      "uv",
      "run",
      "--project",
      pythonDir,
      "python",
      analyzerScript,
      "--dataset-path",
      task.dataset_path,
      "--target-col",
      task.target_col,
      "--time-col",
      task.time_col,
      "--split-mode",
      task.split_mode ?? "ratio",
      "--train-ratio",
      String(task.train_ratio),
      "--val-ratio",
      String(task.val_ratio),
      "--out",
      outPath,
    ]
    if (task.test_ratio !== undefined) command.push("--test-ratio", String(task.test_ratio))
    if (task.split_mode === "timestamp") {
      command.push(
        "--train-end-timestamp",
        task.train_end_timestamp!,
        "--validation-end-timestamp",
        task.validation_end_timestamp!,
      )
    }
    const proc = Bun.spawn(command, {
      cwd: directory,
      stdout: "pipe",
      stderr: "pipe",
    })

    const [stdout, stderr] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text()])
    const exitCode = await proc.exited

    if (exitCode !== 0) {
      const stderrTail = stderr.split("\n").filter(Boolean).slice(-10).join("\n")

      return {
        title: "Analyze dataset: error",
        metadata: { status: "error", exit_code: exitCode },
        output: JSON.stringify({
          status: "error",
          exit_code: exitCode,
          stderr_tail: stderrTail,
          message: "Python analyzer failed. Verify the Python environment with: cd python && uv sync",
        }),
      }
    }

    let analysis: unknown
    try {
      const outputContent = await readFile(outPath, "utf-8")
      analysis = JSON.parse(outputContent)
    } catch {
      try {
        analysis = JSON.parse(stdout)
        await writeFile(outPath, JSON.stringify(analysis, null, 2), "utf-8")
      } catch {
        analysis = { raw_stdout: stdout.slice(0, 2000) }
      }
    }

    return {
      title: "Analyze dataset: complete",
      metadata: { status: "success", output_path: outPath },
      output: JSON.stringify({
        status: "success",
        output_path: outPath,
        analysis,
      }),
    }
  },
})
