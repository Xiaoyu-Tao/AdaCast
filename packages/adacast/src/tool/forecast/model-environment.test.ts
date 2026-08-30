import { describe, expect, test } from "bun:test"
import { writeFile } from "fs/promises"
import path from "path"
import { tmpdir } from "../../../test/fixture/fixture"
import { resolveForecastModelEnvironment, STANDARD_HF_CACHE_ENV_KEYS } from "./model-environment"

const STANDARD_ENV_NAMES = [
  "HF_HOME",
  "HF_HUB_CACHE",
  "HF_XET_CACHE",
  "HF_ASSETS_CACHE",
  "TRANSFORMERS_CACHE",
  "HUGGINGFACE_HUB_CACHE",
  "PYTORCH_TRANSFORMERS_CACHE",
  "PYTORCH_PRETRAINED_BERT_CACHE",
]

describe("resolveForecastModelEnvironment cache_config", () => {
  test("preserves operator HF_ENDPOINT and HF_HUB_CACHE without rewriting them", async () => {
    await using tmp = await tmpdir()

    const resolved = resolveForecastModelEnvironment({
      directory: tmp.path,
      env: {
        HF_ENDPOINT: "https://hf-mirror.example",
        HF_HUB_CACHE: "/operator/hub",
      },
    })

    expect(STANDARD_HF_CACHE_ENV_KEYS).toEqual(STANDARD_ENV_NAMES)
    expect(resolved.endpoint).toBe("https://hf-mirror.example")
    expect(resolved.source).toBe("operator")
    expect(resolved.hubCache).toBe("/operator/hub")
    expect(resolved.env.HF_ENDPOINT).toBe("https://hf-mirror.example")
    expect(resolved.env.HF_HUB_CACHE).toBe("/operator/hub")
    expect(resolved.warnings).toEqual([])
  })

  test("uses .forecast/model-cache hub xet assets defaults when no standard cache env exists", async () => {
    await using tmp = await tmpdir()

    const resolved = resolveForecastModelEnvironment({ directory: tmp.path, env: {} })

    expect(resolved.source).toBe("project_default")
    expect(resolved.cacheRoot).toContain(path.join(".forecast", "model-cache"))
    expect(resolved.hubCache).toContain(path.join(".forecast", "model-cache", "hub"))
    expect(resolved.xetCache).toContain(path.join(".forecast", "model-cache", "xet"))
    expect(resolved.assetsCache).toContain(path.join(".forecast", "model-cache", "assets"))
    expect(resolved.transformersCache).toBeNull()
    expect(resolved.env).toEqual(
      expect.objectContaining({
        HF_HUB_CACHE: path.join(tmp.path, ".forecast", "model-cache", "hub"),
        HF_XET_CACHE: path.join(tmp.path, ".forecast", "model-cache", "xet"),
        HF_ASSETS_CACHE: path.join(tmp.path, ".forecast", "model-cache", "assets"),
      }),
    )
    expect(resolved.env.HF_HOME).toBeUndefined()
  })

  test("derives missing HF_HUB_CACHE from legacy TRANSFORMERS_CACHE", async () => {
    await using tmp = await tmpdir()

    const resolved = resolveForecastModelEnvironment({
      directory: tmp.path,
      env: {
        TRANSFORMERS_CACHE: "/legacy/transformers",
      },
    })

    expect(resolved.source).toBe("operator")
    expect(resolved.transformersCache).toBe("/legacy/transformers")
    expect(resolved.hubCache).toBe("/legacy/transformers")
    expect(resolved.env.TRANSFORMERS_CACHE).toBe("/legacy/transformers")
    expect(resolved.env.HF_HUB_CACHE).toBe("/legacy/transformers")
  })

  test("returns cache_warning semantics when .forecast/model-cache cannot be used", async () => {
    await using tmp = await tmpdir()
    await writeFile(path.join(tmp.path, ".forecast"), "not a directory", "utf-8")

    const resolved = resolveForecastModelEnvironment({ directory: tmp.path, env: {} })

    expect(resolved.source).toBe("hf_default")
    expect(resolved.cacheRoot).toBeNull()
    expect(resolved.env.HF_HUB_CACHE).toBeUndefined()
    expect(resolved.warnings).toEqual([
      {
        code: "project_cache_unwritable",
        message: "Project model cache .forecast/model-cache is not writable; using Hugging Face defaults.",
      },
    ])
  })
})
