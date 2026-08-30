import { mkdir, readFile, readdir, rm, writeFile } from "fs/promises"
import matter from "gray-matter"
import path from "path"
import z from "zod"
import { ConfigMarkdown } from "@/config/markdown"
import { Instance } from "@/project/instance"
import { Filesystem } from "@/util/filesystem"
import { Tool } from "../tool"
import DESCRIPTION from "./forecast-skill.txt"
import {
  importForecastSkillMemory,
  listForecastSkillMemory,
  publishForecastSkillMemory,
  readForecastSkillMemory,
} from "./forecast-skill-memory"

const parameters = z.object({
  subcommand: z
    .enum(["list", "read", "write", "delete", "publish_memory", "list_memory", "read_memory", "import_memory"])
    .describe(
      "Skill operation: list/read/write/delete for local .forecast/skills, publish_memory to promote finalized local skills to global Forecast memory, list_memory/read_memory to inspect global Forecast memory, import_memory to copy a memory skill into the current task.",
    ),
  skill_name: z.string().optional().describe("Skill name for read/write/delete operations. Used as directory name under .forecast/skills/"),
  memory_name: z.string().optional().describe("Forecast memory skill name for read_memory/import_memory operations"),
  target_skill_name: z.string().optional().describe("Optional local skill name when importing a memory skill"),
  overwrite: z.boolean().optional().describe("Allow import_memory to replace an existing local skill"),
  skill_description: z.string().optional().describe("Short skill description for YAML frontmatter (write only)"),
  skill_content: z
    .string()
    .optional()
    .describe(
      "Full Markdown body content for the skill (write only). Should include Applicable Conditions, Parameter Space, Feature Template, Runner Entry Point, and Risk Warnings sections.",
    ),
})

type SkillListItem = {
  name: string
  description: string
}

function skillsDir(directory: string): string {
  return path.join(directory, ".forecast", "skills")
}

function resolveSkillDir(root: string, skillName: string): string {
  if (
    skillName.length === 0 ||
    skillName === "." ||
    skillName === ".." ||
    skillName.includes(path.sep) ||
    skillName.includes(path.posix.sep) ||
    skillName.includes(path.win32.sep)
  ) {
    throw new Error(`Invalid skill_name "${skillName}". Use a single directory name under .forecast/skills/.`)
  }

  const target = path.resolve(root, skillName)
  const relative = path.relative(root, target)
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`Invalid skill_name "${skillName}". Use a single directory name under .forecast/skills/.`)
  }

  return target
}

async function listSkills(root: string): Promise<SkillListItem[]> {
  if (!(await Filesystem.isDir(root))) {
    return []
  }

  const entries = await readdir(root, { withFileTypes: true })
  const skills: SkillListItem[] = []

  for (const entry of entries.toSorted((a, b) => a.name.localeCompare(b.name))) {
    if (!entry.isDirectory()) continue

    const skillPath = path.join(root, entry.name, "SKILL.md")
    if (!(await Filesystem.exists(skillPath))) continue

    try {
      const parsed = await ConfigMarkdown.parse(skillPath)
      const metadata = z
        .object({
          name: z.string(),
          description: z.string(),
        })
        .parse(parsed.data)

      skills.push({
        name: metadata.name,
        description: metadata.description,
      })
    } catch {
      // Skip invalid skills so one malformed file does not block listing the rest.
    }
  }

  return skills
}

export const ForecastSkillTool = Tool.define("forecast_skill", {
  description: DESCRIPTION,
  parameters,
  async execute(params) {
    const root = skillsDir(Instance.directory)
    let result: unknown

    switch (params.subcommand) {
      case "list": {
        const skills = await listSkills(root)
        result = { skills, count: skills.length }
        break
      }

      case "read": {
        if (!params.skill_name) {
          throw new Error("forecast_skill read requires skill_name")
        }

        const skillDir = resolveSkillDir(root, params.skill_name)
        const skillPath = path.join(skillDir, "SKILL.md")
        if (!(await Filesystem.exists(skillPath))) {
          result = { error: `Skill "${params.skill_name}" not found at ${skillPath}` }
          break
        }

        const content = await readFile(skillPath, "utf-8")
        result = { name: params.skill_name, path: skillPath, content }
        break
      }

      case "write": {
        if (!params.skill_name) {
          throw new Error("forecast_skill write requires skill_name")
        }
        if (!params.skill_description) {
          throw new Error("forecast_skill write requires skill_description")
        }
        if (!params.skill_content) {
          throw new Error("forecast_skill write requires skill_content")
        }

        await mkdir(root, { recursive: true })

        const skillDir = resolveSkillDir(root, params.skill_name)
        await mkdir(skillDir, { recursive: true })

        const skillPath = path.join(skillDir, "SKILL.md")
        const skillMd = matter.stringify(`\n${params.skill_content.trim()}\n`, {
          name: params.skill_name,
          description: params.skill_description,
        })

        await writeFile(skillPath, skillMd, "utf-8")
        result = { written: true, path: skillPath, name: params.skill_name }
        break
      }

      case "delete": {
        if (!params.skill_name) {
          throw new Error("forecast_skill delete requires skill_name")
        }

        const skillDir = resolveSkillDir(root, params.skill_name)
        if (!(await Filesystem.isDir(skillDir))) {
          result = { error: `Skill directory "${params.skill_name}" not found` }
          break
        }

        await rm(skillDir, { recursive: true, force: true })
        result = { deleted: true, name: params.skill_name }
        break
      }

      case "publish_memory": {
        result = await publishForecastSkillMemory({
          directory: Instance.directory,
          reason: "forecast_skill publish_memory",
        })
        break
      }

      case "list_memory": {
        const memories = await listForecastSkillMemory()
        result = { memories, count: memories.length }
        break
      }

      case "read_memory": {
        if (!params.memory_name) {
          throw new Error("forecast_skill read_memory requires memory_name")
        }

        const memory = await readForecastSkillMemory(params.memory_name)
        result = memory ?? { error: `Forecast memory "${params.memory_name}" not found` }
        break
      }

      case "import_memory": {
        if (!params.memory_name) {
          throw new Error("forecast_skill import_memory requires memory_name")
        }

        result = await importForecastSkillMemory({
          directory: Instance.directory,
          memoryName: params.memory_name,
          targetSkillName: params.target_skill_name,
          overwrite: params.overwrite,
        })
        break
      }
    }

    return {
      title: `Forecast skill: ${params.subcommand}`,
      metadata: { subcommand: params.subcommand, skill_name: params.skill_name, memory_name: params.memory_name },
      output: JSON.stringify(result),
    }
  },
})
