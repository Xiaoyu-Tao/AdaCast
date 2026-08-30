import { createHash } from "node:crypto"
import { mkdir, readFile, readdir, writeFile } from "fs/promises"
import matter from "gray-matter"
import path from "path"
import { ConfigMarkdown } from "@/config/markdown"
import { Global } from "@/global"
import { Filesystem } from "@/util/filesystem"
import { ExperimentRecord, TaskConfig } from "./types"

export type ForecastSkillMemoryItem = {
  name: string
  description: string
  path: string
  manifest_path: string | null
  original_skill_name?: string
  published_at?: string
}

type LocalForecastSkill = {
  name: string
  description: string
  path: string
  content: string
}

type ForecastTaskMemorySummary = {
  dataset_name: string | null
  target_col: string | null
  time_col: string | null
  pred_len: number | null
  forecast_gap: number | null
  seq_len: number | null
  label_len: number | null
  window_stride: number | null
  train_ratio: number | null
  val_ratio: number | null
  test_ratio: number | null
  split_mode: "ratio" | "timestamp" | null
  train_end_timestamp: string | null
  validation_end_timestamp: string | null
  model_families: string[]
}

type ForecastBestMemorySummary = {
  run_id: string
  model: string
  metrics: Record<string, number | null> | null
} | null

type ForecastSkillMemoryManifest = {
  name: string
  original_skill_name: string
  original_description: string
  source_project: string
  source_skill_sha256: string
  memory_sha256: string
  published_at: string
  reason: string
  task: ForecastTaskMemorySummary | null
  best: ForecastBestMemorySummary
}

type ForecastSkillMemoryPublishResult = {
  published: boolean
  published_at: string
  reason: string
  count: number
  memory_root: string
  skills_root: string
  manifest_path: string | null
  skills: Array<{
    name: string
    original_skill_name: string
    path: string
    manifest_path: string
    source_skill_sha256: string
  }>
}

export function forecastSkillMemoryRoot(): string {
  const override = process.env.ADACAST_FORECAST_MEMORY_DIR
  if (override) return path.resolve(override)

  const testHome = process.env.ADACAST_TEST_HOME
  if (testHome) return path.join(testHome, ".adacast", "forecast-memory")

  return path.join(Global.Path.data, "forecast-memory")
}

export function forecastSkillMemorySkillsDir(): string {
  return path.join(forecastSkillMemoryRoot(), "skills")
}

function localSkillsDir(directory: string): string {
  return path.join(directory, ".forecast", "skills")
}

function forecastRoot(directory: string): string {
  return path.join(directory, ".forecast")
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex")
}

function slugify(value: string): string {
  const slug = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
  return slug || "skill"
}

function assertSingleSegment(value: string, label: string): void {
  if (
    value.length === 0 ||
    value === "." ||
    value === ".." ||
    value.includes(path.sep) ||
    value.includes(path.posix.sep) ||
    value.includes(path.win32.sep)
  ) {
    throw new Error(`Invalid ${label} "${value}". Use a single skill name segment.`)
  }
}

function resolveSkillDir(root: string, skillName: string): string {
  assertSingleSegment(skillName, "skill name")
  const target = path.resolve(root, skillName)
  const relative = path.relative(root, target)
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`Invalid skill name "${skillName}". Use a single skill name segment.`)
  }
  return target
}

function projectLabel(directory: string): string {
  return path.basename(path.resolve(directory)) || "forecast-project"
}

async function readLocalForecastSkills(directory: string): Promise<LocalForecastSkill[]> {
  const root = localSkillsDir(directory)
  if (!(await Filesystem.isDir(root))) return []

  const entries = await readdir(root, { withFileTypes: true })
  const skills: LocalForecastSkill[] = []
  for (const entry of entries.toSorted((a, b) => a.name.localeCompare(b.name))) {
    if (!entry.isDirectory()) continue

    const skillPath = path.join(root, entry.name, "SKILL.md")
    if (!(await Filesystem.exists(skillPath))) continue

    try {
      const parsed = await ConfigMarkdown.parse(skillPath)
      const metadata = {
        name: typeof parsed.data.name === "string" ? parsed.data.name : entry.name,
        description: typeof parsed.data.description === "string" ? parsed.data.description : "",
      }
      skills.push({
        name: metadata.name,
        description: metadata.description,
        path: skillPath,
        content: parsed.content.trim(),
      })
    } catch {
      // A malformed local skill should not prevent other finalized skills from being promoted.
    }
  }
  return skills
}

async function readTaskSummary(directory: string): Promise<ForecastTaskMemorySummary | null> {
  try {
    const task = TaskConfig.parse(JSON.parse(await readFile(path.join(forecastRoot(directory), "task.json"), "utf-8")))
    return {
      dataset_name: path.basename(task.dataset_path),
      target_col: task.target_col,
      time_col: task.time_col,
      pred_len: task.pred_len,
      forecast_gap: task.forecast_gap,
      seq_len: task.seq_len,
      label_len: task.label_len,
      window_stride: task.window_stride,
      train_ratio: task.train_ratio,
      val_ratio: task.val_ratio,
      test_ratio: task.test_ratio,
      split_mode: task.split_mode ?? "ratio",
      train_end_timestamp: task.train_end_timestamp ?? null,
      validation_end_timestamp: task.validation_end_timestamp ?? null,
      model_families: task.model_families,
    }
  } catch {
    return null
  }
}

async function readBestSummary(directory: string): Promise<ForecastBestMemorySummary> {
  try {
    const best = ExperimentRecord.parse(
      JSON.parse(await readFile(path.join(forecastRoot(directory), "best.json"), "utf-8")),
    )
    return {
      run_id: best.run_id,
      model: best.model,
      metrics: best.metrics,
    }
  } catch {
    return null
  }
}

function renderTaskSummary(task: ForecastTaskMemorySummary | null): string[] {
  if (!task) return ["- Task metadata was unavailable when this memory was published."]
  return [
    `- Dataset: ${task.dataset_name ?? "unknown"}`,
    `- Target/time columns: ${task.target_col ?? "unknown"} / ${task.time_col ?? "unknown"}`,
    `- Windowing: seq_len=${task.seq_len ?? "unknown"}, label_len=${task.label_len ?? "unknown"}, forecast_gap=${task.forecast_gap ?? 0}, pred_len=${task.pred_len ?? "unknown"}, window_stride=${task.window_stride ?? 1}`,
    task.split_mode === "timestamp"
      ? `- Split origin timestamps (inclusive): train origin through ${task.train_end_timestamp ?? "unknown"}, validation origin through ${task.validation_end_timestamp ?? "unknown"}, evaluated targets start after forecast_gap`
      : `- Split ratios: train=${task.train_ratio ?? "unknown"}, val=${task.val_ratio ?? "unknown"}, test=${task.test_ratio ?? "unknown"}`,
    `- Model families: ${task.model_families.length > 0 ? task.model_families.join(", ") : "unknown"}`,
  ]
}

function renderBestSummary(best: ForecastBestMemorySummary): string[] {
  if (!best) return ["- No best validation record was available when this memory was published."]
  const metrics = best.metrics
    ? Object.entries(best.metrics)
        .map(([key, value]) => `${key}=${value ?? "null"}`)
        .join(", ")
    : "none"
  return [`- Best validation run: ${best.model} (${best.run_id})`, `- Metrics: ${metrics}`]
}

function buildMemoryMarkdown(input: { source: LocalForecastSkill; manifest: ForecastSkillMemoryManifest }): string {
  const lines = [
    `# Forecast Memory: ${input.source.name}`,
    "",
    "## Reuse Guidance",
    "- This is a long-lived cross-project memory promoted from a completed Forecast phase.",
    "- Treat it as prior experience only; validate it against the current dataset before reuse.",
    "- Prefer reusing it only when task granularity, horizon, model family, and data conditions are similar.",
    "- Do not assume the stored parameters are optimal for a new project or domain.",
    "",
    "## Source Task Summary",
    ...renderTaskSummary(input.manifest.task),
    "",
    "## Final Forecast Evidence",
    ...renderBestSummary(input.manifest.best),
    "",
    "## Original Forecast Skill",
    input.source.content,
    "",
  ]
  return lines.join("\n").trim()
}

export async function publishForecastSkillMemory(input: {
  directory: string
  reason?: string
}): Promise<ForecastSkillMemoryPublishResult> {
  const reason = input.reason ?? "manual"
  const sourceSkills = await readLocalForecastSkills(input.directory)
  const publishedAt = new Date().toISOString()
  const root = forecastSkillMemoryRoot()
  const skillsRoot = forecastSkillMemorySkillsDir()

  if (sourceSkills.length === 0) {
    return {
      published: false,
      published_at: publishedAt,
      reason,
      count: 0,
      memory_root: root,
      skills_root: skillsRoot,
      manifest_path: null,
      skills: [],
    }
  }

  await mkdir(skillsRoot, { recursive: true })
  const task = await readTaskSummary(input.directory)
  const best = await readBestSummary(input.directory)
  const project = projectLabel(input.directory)
  const published: ForecastSkillMemoryPublishResult["skills"] = []

  for (const source of sourceSkills) {
    const sourceSkillSha = sha256(
      JSON.stringify({ name: source.name, description: source.description, content: source.content }),
    )
    const memoryHash = sha256(
      JSON.stringify({
        sourceSkillSha,
        sourceName: source.name,
        task,
        best,
      }),
    )
    const memoryName = `forecast-memory-${slugify(source.name)}-${memoryHash.slice(0, 12)}`
    const memoryDir = path.join(skillsRoot, memoryName)
    await mkdir(memoryDir, { recursive: true })

    const manifest: ForecastSkillMemoryManifest = {
      name: memoryName,
      original_skill_name: source.name,
      original_description: source.description,
      source_project: project,
      source_skill_sha256: sourceSkillSha,
      memory_sha256: memoryHash,
      published_at: publishedAt,
      reason,
      task,
      best,
    }
    const skillMd = matter.stringify(`\n${buildMemoryMarkdown({ source, manifest })}\n`, {
      name: memoryName,
      description: `Forecast memory from ${project}: ${source.description || source.name}`,
      adacast_forecast_memory: {
        original_skill_name: source.name,
        source_project: project,
        published_at: publishedAt,
        memory_sha256: memoryHash,
      },
    })

    const skillPath = path.join(memoryDir, "SKILL.md")
    const manifestPath = path.join(memoryDir, "manifest.json")
    await writeFile(skillPath, skillMd, "utf-8")
    await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf-8")
    published.push({
      name: memoryName,
      original_skill_name: source.name,
      path: skillPath,
      manifest_path: manifestPath,
      source_skill_sha256: sourceSkillSha,
    })
  }

  const publishManifest = {
    published_at: publishedAt,
    reason,
    source_project: project,
    count: published.length,
    skills: published,
  }
  const manifestName = `${publishedAt.replace(/[:.]/g, "-")}-${sha256(JSON.stringify(published)).slice(0, 12)}.json`
  const manifestsRoot = path.join(root, "manifests")
  await mkdir(manifestsRoot, { recursive: true })
  const manifestPath = path.join(manifestsRoot, manifestName)
  await writeFile(manifestPath, `${JSON.stringify(publishManifest, null, 2)}\n`, "utf-8")
  await writeFile(path.join(root, "latest-manifest.json"), `${JSON.stringify(publishManifest, null, 2)}\n`, "utf-8")

  return {
    published: true,
    published_at: publishedAt,
    reason,
    count: published.length,
    memory_root: root,
    skills_root: skillsRoot,
    manifest_path: manifestPath,
    skills: published,
  }
}

export async function listForecastSkillMemory(): Promise<ForecastSkillMemoryItem[]> {
  const root = forecastSkillMemorySkillsDir()
  if (!(await Filesystem.isDir(root))) return []

  const entries = await readdir(root, { withFileTypes: true })
  const items: ForecastSkillMemoryItem[] = []
  for (const entry of entries.toSorted((a, b) => a.name.localeCompare(b.name))) {
    if (!entry.isDirectory()) continue

    const skillPath = path.join(root, entry.name, "SKILL.md")
    if (!(await Filesystem.exists(skillPath))) continue

    try {
      const parsed = await ConfigMarkdown.parse(skillPath)
      const manifestPath = path.join(root, entry.name, "manifest.json")
      let manifest: Partial<ForecastSkillMemoryManifest> = {}
      if (await Filesystem.exists(manifestPath)) {
        manifest = JSON.parse(await readFile(manifestPath, "utf-8")) as Partial<ForecastSkillMemoryManifest>
      }
      items.push({
        name: typeof parsed.data.name === "string" ? parsed.data.name : entry.name,
        description: typeof parsed.data.description === "string" ? parsed.data.description : "",
        path: skillPath,
        manifest_path: (await Filesystem.exists(manifestPath)) ? manifestPath : null,
        original_skill_name: manifest.original_skill_name,
        published_at: manifest.published_at,
      })
    } catch {
      // Skip malformed memory entries so one bad memory does not hide the rest.
    }
  }
  return items
}

export async function readForecastSkillMemory(memoryName: string): Promise<{
  name: string
  path: string
  content: string
  manifest: ForecastSkillMemoryManifest | null
} | null> {
  const memoryDir = resolveSkillDir(forecastSkillMemorySkillsDir(), memoryName)
  const skillPath = path.join(memoryDir, "SKILL.md")
  if (!(await Filesystem.exists(skillPath))) return null

  const manifestPath = path.join(memoryDir, "manifest.json")
  let manifest: ForecastSkillMemoryManifest | null = null
  if (await Filesystem.exists(manifestPath)) {
    try {
      manifest = JSON.parse(await readFile(manifestPath, "utf-8")) as ForecastSkillMemoryManifest
    } catch {
      manifest = null
    }
  }

  return {
    name: memoryName,
    path: skillPath,
    content: await readFile(skillPath, "utf-8"),
    manifest,
  }
}

export async function importForecastSkillMemory(input: {
  directory: string
  memoryName: string
  targetSkillName?: string
  overwrite?: boolean
}): Promise<{ imported: true; name: string; path: string; source_memory: string }> {
  const memory = await readForecastSkillMemory(input.memoryName)
  if (!memory) throw new Error(`Forecast memory "${input.memoryName}" not found`)

  const parsed = matter(memory.content)
  const targetName = input.targetSkillName ?? memory.manifest?.original_skill_name ?? input.memoryName
  assertSingleSegment(targetName, "target_skill_name")

  const targetDir = resolveSkillDir(localSkillsDir(input.directory), targetName)
  const targetPath = path.join(targetDir, "SKILL.md")
  if ((await Filesystem.exists(targetPath)) && !input.overwrite) {
    throw new Error(`Local skill "${targetName}" already exists; pass overwrite=true to replace it`)
  }

  await mkdir(targetDir, { recursive: true })
  const description =
    memory.manifest?.original_description ||
    (typeof parsed.data.description === "string"
      ? parsed.data.description
      : `Imported forecast memory ${input.memoryName}`)
  const skillMd = matter.stringify(`\n${parsed.content.trim()}\n`, {
    name: targetName,
    description: `Imported forecast memory: ${description}`,
  })
  await writeFile(targetPath, skillMd, "utf-8")

  return {
    imported: true,
    name: targetName,
    path: targetPath,
    source_memory: input.memoryName,
  }
}
