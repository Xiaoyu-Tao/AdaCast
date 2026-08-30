import path from "path"
import z from "zod"
import { Filesystem } from "@/util/filesystem"
import { computeFileSha256 } from "./model-artifact"
import { readJsonFile, writeJsonFile } from "./forecast-adjustment-common"

const FrozenTaskIntegrityLock = z.object({
  version: z.literal(1),
  task_sha256: z.string().length(64),
  frozen_at: z.string().min(1),
})

export type FrozenTaskIntegrityStatus =
  | { status: "locked"; task_sha256: string }
  | { status: "legacy_unlocked" }
  | { status: "invalid_lock"; reason: string }
  | { status: "modified"; expected_sha256: string; actual_sha256: string }

function lockPath(forecastRoot: string): string {
  return path.join(forecastRoot, "task-lock.json")
}

export async function writeFrozenTaskIntegrityLock(taskPath: string, frozenAt: string): Promise<void> {
  const taskSha256 = await computeFileSha256(taskPath)
  await writeJsonFile(lockPath(path.dirname(taskPath)), {
    version: 1,
    task_sha256: taskSha256,
    frozen_at: frozenAt,
  })
}

export async function inspectFrozenTaskIntegrity(forecastRoot: string): Promise<FrozenTaskIntegrityStatus> {
  const taskPath = path.join(forecastRoot, "task.json")
  const integrityPath = lockPath(forecastRoot)
  if (!(await Filesystem.exists(integrityPath))) return { status: "legacy_unlocked" }

  const parsed = FrozenTaskIntegrityLock.safeParse(await readJsonFile(integrityPath).catch(() => null))
  if (!parsed.success) {
    return { status: "invalid_lock", reason: "task-lock.json is missing required integrity metadata" }
  }
  if (!(await Filesystem.exists(taskPath))) {
    return { status: "invalid_lock", reason: "task-lock.json exists but task.json is missing" }
  }

  const actualSha256 = await computeFileSha256(taskPath)
  if (actualSha256 !== parsed.data.task_sha256) {
    return {
      status: "modified",
      expected_sha256: parsed.data.task_sha256,
      actual_sha256: actualSha256,
    }
  }
  return { status: "locked", task_sha256: actualSha256 }
}

export async function assertFrozenTaskIntegrity(forecastRoot: string): Promise<void> {
  const integrity = await inspectFrozenTaskIntegrity(forecastRoot)
  if (integrity.status === "legacy_unlocked" || integrity.status === "locked") return
  if (integrity.status === "invalid_lock") {
    throw new Error(`Frozen task integrity check failed: ${integrity.reason}. Do not edit managed .forecast artifacts.`)
  }
  throw new Error(
    "Frozen task integrity check failed: task.json changed after forecast_task froze it. " +
      `Expected SHA-256 ${integrity.expected_sha256}, got ${integrity.actual_sha256}. ` +
      "Do not edit task.json directly; start a new task when frozen parameters must change.",
  )
}
