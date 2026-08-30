import { afterEach, beforeEach, describe, expect, test } from "bun:test"
import { mkdir, mkdtemp, rm, writeFile } from "fs/promises"
import os from "os"
import path from "path"
import { resolveForecastPythonProject } from "../../src/tool/forecast/python-project"

const originalExecPath = process.execPath
const originalPythonProjectEnv = process.env.ADACAST_PYTHON_PROJECT

async function writePyproject(dir: string) {
  await mkdir(dir, { recursive: true })
  await writeFile(path.join(dir, "pyproject.toml"), "[project]\nname = 'adacast-ml'\nversion = '0.0.0'\n", "utf-8")
}

function setExecPath(value: string) {
  Object.defineProperty(process, "execPath", {
    value,
    configurable: true,
    writable: true,
  })
}

describe("resolveForecastPythonProject", () => {
  let tmpdir: string

  beforeEach(async () => {
    tmpdir = await mkdtemp(path.join(os.tmpdir(), "adacast-python-project-"))
    delete process.env.ADACAST_PYTHON_PROJECT
    setExecPath(originalExecPath)
  })

  afterEach(async () => {
    setExecPath(originalExecPath)
    if (originalPythonProjectEnv === undefined) {
      delete process.env.ADACAST_PYTHON_PROJECT
    } else {
      process.env.ADACAST_PYTHON_PROJECT = originalPythonProjectEnv
    }
    await rm(tmpdir, { recursive: true, force: true })
  })

  test("prefers ADACAST_PYTHON_PROJECT when configured", async () => {
    const configured = path.join(tmpdir, "configured-runtime")
    const workspace = path.join(tmpdir, "workspace")
    await writePyproject(configured)
    await writePyproject(path.join(workspace, "python"))
    process.env.ADACAST_PYTHON_PROJECT = configured

    const resolved = await resolveForecastPythonProject(workspace, { includeBundledFallback: false })

    expect(resolved).toBe(configured)
  })

  test("uses workspace python project when present", async () => {
    const workspace = path.join(tmpdir, "workspace")
    const pythonDir = path.join(workspace, "python")
    await writePyproject(pythonDir)

    const resolved = await resolveForecastPythonProject(workspace, { includeBundledFallback: false })

    expect(resolved).toBe(pythonDir)
  })

  test("uses executable-adjacent runtime in bin/python", async () => {
    const workspace = path.join(tmpdir, "workspace")
    const execDir = path.join(tmpdir, "package", "bin")
    const pythonDir = path.join(execDir, "python")
    await mkdir(workspace, { recursive: true })
    await writePyproject(pythonDir)

    const resolved = await resolveForecastPythonProject(workspace, {
      execPath: path.join(execDir, "adacast"),
      includeBundledFallback: false,
    })

    expect(resolved).toBe(pythonDir)
  })

  test("uses shared install runtime under share/adacast/python", async () => {
    const workspace = path.join(tmpdir, "workspace")
    const execDir = path.join(tmpdir, "usr", "bin")
    const pythonDir = path.join(tmpdir, "usr", "share", "adacast", "python")
    await mkdir(workspace, { recursive: true })
    await writePyproject(pythonDir)

    const resolved = await resolveForecastPythonProject(workspace, {
      execPath: path.join(execDir, "adacast"),
      includeBundledFallback: false,
    })

    expect(resolved).toBe(pythonDir)
  })

  test("uses packaged runtime from a local node_modules install", async () => {
    const workspace = path.join(tmpdir, "workspace")
    const pythonDir = path.join(workspace, "node_modules", "adacast-linux-x64", "bin", "python")
    await mkdir(workspace, { recursive: true })
    await writePyproject(pythonDir)

    const resolved = await resolveForecastPythonProject(workspace, {
      execPath: path.join(tmpdir, "missing", "adacast"),
      includeBundledFallback: false,
    })

    expect(resolved).toBe(pythonDir)
  })

  test("throws when no runtime is available", async () => {
    const workspace = path.join(tmpdir, "workspace")
    await mkdir(workspace, { recursive: true })

    await expect(
      resolveForecastPythonProject(workspace, {
        execPath: path.join(tmpdir, "missing", "adacast"),
        includeBundledFallback: false,
      }),
    ).rejects.toThrow("Forecast Python project not found")
  })
})
