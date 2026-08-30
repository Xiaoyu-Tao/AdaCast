import { describe, expect, test } from "bun:test"
import { mkdir, writeFile } from "fs/promises"
import path from "path"
import { tmpdir } from "../../../test/fixture/fixture"
import {
  assertFrozenTaskIntegrity,
  inspectFrozenTaskIntegrity,
  writeFrozenTaskIntegrityLock,
} from "./forecast-task-integrity"

describe("frozen forecast task integrity", () => {
  test("keeps legacy unlocked tasks readable for backward compatibility", async () => {
    await using tmp = await tmpdir()
    const forecastRoot = path.join(tmp.path, ".forecast")
    await mkdir(forecastRoot, { recursive: true })
    await writeFile(path.join(forecastRoot, "task.json"), JSON.stringify({ model: "UserPrior" }))

    expect(await inspectFrozenTaskIntegrity(forecastRoot)).toEqual({ status: "legacy_unlocked" })
    await expect(assertFrozenTaskIntegrity(forecastRoot)).resolves.toBeUndefined()
  })

  test("detects any task change after forecast_task writes the integrity lock", async () => {
    await using tmp = await tmpdir()
    const forecastRoot = path.join(tmp.path, ".forecast")
    const taskPath = path.join(forecastRoot, "task.json")
    await mkdir(forecastRoot, { recursive: true })
    await writeFile(taskPath, JSON.stringify({ model: "UserPrior", reflection_enabled: true }))
    await writeFrozenTaskIntegrityLock(taskPath, "2026-07-16T00:00:00.000Z")

    expect(await inspectFrozenTaskIntegrity(forecastRoot)).toMatchObject({ status: "locked" })
    await writeFile(taskPath, JSON.stringify({ model: "UserPrior", reflection_enabled: false }))
    expect(await inspectFrozenTaskIntegrity(forecastRoot)).toMatchObject({ status: "modified" })
    await expect(assertFrozenTaskIntegrity(forecastRoot)).rejects.toThrow("task.json changed after forecast_task")
  })
})
