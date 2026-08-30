import { describe, expect, test } from "bun:test"
import path from "path"
import { Process } from "@/util/process"
import { computeSpecHash } from "./hash"

const ROOT = path.resolve(import.meta.dir, "..", "..", "..", "..", "..")
const PYTHON_PROJECT = path.join(ROOT, "python")
const PYTHON_EXECUTABLE =
  process.platform === "win32"
    ? path.join(PYTHON_PROJECT, ".venv", "Scripts", "python.exe")
    : path.join(PYTHON_PROJECT, ".venv", "bin", "python")
const PYTHON_HASH_SCRIPT = [
  "import json, os",
  "from adacast_ml.utils.hash import compute_spec_hash",
  'print(compute_spec_hash(json.loads(os.environ["ADACAST_SPEC"])))',
].join("; ")

async function computePythonSpecHash(spec: Record<string, unknown>) {
  const out = await Process.run([PYTHON_EXECUTABLE, "-c", PYTHON_HASH_SCRIPT], {
    cwd: ROOT,
    env: {
      ADACAST_SPEC: JSON.stringify(spec),
    },
  })

  return out.stdout.toString().trim()
}

describe("forecast.hash", () => {
  test("matches Python for a flat spec", async () => {
    const spec = {
      model: "DLinear",
      pred_len: 96,
    }

    expect(computeSpecHash(spec)).toBe(await computePythonSpecHash(spec))
  })

  test("matches Python for a nested spec", async () => {
    const spec = {
      model: "DLinear",
      hyperparams: {
        batch_size: 32,
        learning_rate: 0.001,
      },
      pred_len: 96,
    }

    expect(computeSpecHash(spec)).toBe(await computePythonSpecHash(spec))
  })

  test("matches Python for Unicode paths and nested text", async () => {
    const spec = {
      model: "SegRNN",
      dataset_path: "D:\\桌面大文件\\学习资料\\光伏数据.csv",
      context: {
        description: "光伏功率预测",
      },
      pred_len: 96,
    }

    expect(computeSpecHash(spec)).toBe(await computePythonSpecHash(spec))
  })

  test("normalizes key ordering before hashing", () => {
    expect(computeSpecHash({ z: 1, a: 2 })).toBe(computeSpecHash({ a: 2, z: 1 }))
  })
})
