import { createHash } from "node:crypto"

export function deepSortKeys(obj: unknown): unknown {
  if (obj === null || obj === undefined) return obj
  if (Array.isArray(obj)) return obj.map((value) => deepSortKeys(value))
  if (typeof obj === "object") {
    const sorted: Record<string, unknown> = {}
    for (const key of Object.keys(obj as Record<string, unknown>).sort()) {
      sorted[key] = deepSortKeys((obj as Record<string, unknown>)[key])
    }
    return sorted
  }
  return obj
}

export function computeSpecHash(spec: Record<string, unknown>): string {
  const canonical = JSON.stringify(deepSortKeys(spec))
  return createHash("sha256").update(canonical, "utf-8").digest("hex")
}
