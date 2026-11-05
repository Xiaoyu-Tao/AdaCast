import path from "node:path"
import { createHash, randomUUID } from "node:crypto"
import { mkdir, readFile, rename, realpath, stat, writeFile } from "node:fs/promises"
import { AppError } from "./contracts"

export const now = () => new Date().toISOString()
export function canonical(value: any): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value)
  if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]"
  return "{" + Object.keys(value).sort().filter(k => value[k] !== undefined)
    .map(k => JSON.stringify(k) + ":" + canonical(value[k])).join(",") + "}"
}
export const digest = (value: string | Uint8Array) => createHash("sha256").update(value).digest("hex")
export const id = (prefix: string) => prefix + "_" + randomUUID().replaceAll("-", "")
export async function atomicWrite(file: string, value: string | Uint8Array) {
  await mkdir(path.dirname(file), { recursive: true })
  const tmp = file + ".tmp-" + randomUUID()
  await writeFile(tmp, value, { mode: 0o600 })
  await rename(tmp, file)
}
export const writeJSON = (file: string, value: unknown) => atomicWrite(file, JSON.stringify(value, null, 2) + "\n")
export async function readJSON<T = any>(file: string): Promise<T> { return JSON.parse(await readFile(file, "utf8")) }
export async function exists(file: string) { return stat(file).then(() => true, () => false) }
export function inside(root: string, file: string) {
  const rel = path.relative(root, file)
  return rel === "" || (!rel.startsWith(".." + path.sep) && rel !== ".." && !path.isAbsolute(rel))
}
export async function safePath(root: string, relative: string) {
  const file = path.resolve(root, relative)
  if (!inside(root, file)) throw new AppError("PATH_DENIED", "Path escapes the permitted directory", 3)
  let cursor = file
  while (!await exists(cursor)) cursor = path.dirname(cursor)
  if (!inside(await realpath(root), await realpath(cursor))) throw new AppError("PATH_DENIED", "Symlink escapes permitted directory", 3)
  return file
}
export function checkAbort(signal: AbortSignal) {
  if (signal.aborted) throw signal.reason instanceof AppError ? signal.reason : new AppError("INTERRUPTED", "Execution interrupted", 130, true)
}
export function errorText(error: unknown) { return error instanceof Error ? error.message : String(error) }
export function sanitize(text: string) {
  for (const [key, value] of Object.entries(process.env)) {
    if (value && value.length >= 8 && /(?:KEY|TOKEN|SECRET|PASSWORD)$/i.test(key)) text = text.replaceAll(value, "[REDACTED]")
  }
  return text
}
