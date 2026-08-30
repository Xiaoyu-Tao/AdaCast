import { afterEach, beforeEach, describe, expect, test } from "bun:test"
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "fs/promises"
import os from "os"
import path from "path"
import { Instance } from "../../src/project/instance"
import { MessageID, SessionID } from "../../src/session/schema"
import { Skill } from "../../src/skill"
import { ForecastSkillTool } from "../../src/tool/forecast/forecast-skill"

const ctx = {
  sessionID: SessionID.make("ses_forecast-skill-test"),
  messageID: MessageID.make("msg_forecast-skill-test"),
  callID: "forecast-skill-test-call",
  agent: "forecast-skill-test-agent",
  abort: AbortSignal.any([]),
  messages: [],
  metadata: () => {},
  ask: async () => {},
}

describe("forecast_skill CRUD", () => {
  let tmpdir: string
  let previousMemoryDir: string | undefined

  beforeEach(async () => {
    tmpdir = await mkdtemp(path.join(os.tmpdir(), "forecast-skill-test-"))
    previousMemoryDir = process.env.ADACAST_FORECAST_MEMORY_DIR
    process.env.ADACAST_FORECAST_MEMORY_DIR = path.join(tmpdir, "global-forecast-memory")
    await mkdir(path.join(tmpdir, ".forecast", "skills"), { recursive: true })
  })

  afterEach(async () => {
    await Instance.disposeAll()
    if (previousMemoryDir === undefined) {
      delete process.env.ADACAST_FORECAST_MEMORY_DIR
    } else {
      process.env.ADACAST_FORECAST_MEMORY_DIR = previousMemoryDir
    }
    await rm(tmpdir, { recursive: true, force: true })
  })

  async function run(params: Record<string, unknown>) {
    return Instance.provide({
      directory: tmpdir,
      fn: async () => {
        const tool = await ForecastSkillTool.init()
        return tool.execute(params as never, ctx)
      },
    })
  }

  function skillPath(skillName: string) {
    return path.join(tmpdir, ".forecast", "skills", skillName, "SKILL.md")
  }

  async function writeSkill(
    skillName: string,
    skillDescription = "Linear models for trend dominated series",
    skillContent = "# Linear Models\n\n## Applicable Conditions\n- Strong trend",
  ) {
    return run({
      subcommand: "write",
      skill_name: skillName,
      skill_description: skillDescription,
      skill_content: skillContent,
    })
  }

  test("list returns an empty array when no skills exist", async () => {
    const result = await run({ subcommand: "list" })
    const output = JSON.parse(result.output)

    expect(output.skills).toEqual([])
    expect(output.count).toBe(0)
  })

  test("write creates SKILL.md with the expected YAML frontmatter", async () => {
    await writeSkill("linear-models")

    const content = await readFile(skillPath("linear-models"), "utf-8")
    const lines = content.split("\n")

    expect(lines[0]).toBe("---")
    expect(lines[1]).toBe("name: linear-models")
    expect(lines[2]).toBe("description: Linear models for trend dominated series")
    expect(lines[3]).toBe("---")
    expect(lines[4]).toBe("")
  })

  test("write places the markdown body after the frontmatter block", async () => {
    await writeSkill("linear-models")

    const content = await readFile(skillPath("linear-models"), "utf-8")
    const lines = content.split("\n")

    expect(lines[5]).toBe("# Linear Models")
    expect(content).toContain("## Applicable Conditions")
    expect(content).toContain("- Strong trend")
  })

  test("read returns the full content of a previously written skill", async () => {
    await writeSkill("linear-models")

    const result = await run({ subcommand: "read", skill_name: "linear-models" })
    const output = JSON.parse(result.output)
    const content = await readFile(skillPath("linear-models"), "utf-8")

    expect(output.name).toBe("linear-models")
    expect(output.path).toBe(skillPath("linear-models"))
    expect(output.content).toBe(content)
  })

  test("list returns names and descriptions after writing multiple skills", async () => {
    await writeSkill("linear-models")
    await writeSkill("transformers", "Transformer models for complex seasonality", "# Transformers\n\n## Applicable Conditions\n- Multi scale seasonality")

    const result = await run({ subcommand: "list" })
    const output = JSON.parse(result.output)

    expect(output.count).toBe(2)
    expect(output.skills).toEqual([
      { name: "linear-models", description: "Linear models for trend dominated series" },
      { name: "transformers", description: "Transformer models for complex seasonality" },
    ])
  })

  test("delete removes the skill directory and SKILL.md file", async () => {
    await writeSkill("linear-models")

    const result = await run({ subcommand: "delete", skill_name: "linear-models" })
    const output = JSON.parse(result.output)

    expect(output).toEqual({ deleted: true, name: "linear-models" })
    await expect(stat(path.join(tmpdir, ".forecast", "skills", "linear-models"))).rejects.toThrow()
  })

  test("list after delete no longer includes the removed skill", async () => {
    await writeSkill("linear-models")
    await writeSkill("transformers", "Transformer models for complex seasonality", "# Transformers\n\n## Applicable Conditions\n- Multi scale seasonality")
    await run({ subcommand: "delete", skill_name: "linear-models" })

    const result = await run({ subcommand: "list" })
    const output = JSON.parse(result.output)

    expect(output.count).toBe(1)
    expect(output.skills).toEqual([{ name: "transformers", description: "Transformer models for complex seasonality" }])
  })

  test("read of a missing skill returns an error object", async () => {
    const result = await run({ subcommand: "read", skill_name: "missing-skill" })
    const output = JSON.parse(result.output)

    expect(output.error).toContain('Skill "missing-skill" not found')
  })

  test("delete of a missing skill returns an error object", async () => {
    const result = await run({ subcommand: "delete", skill_name: "missing-skill" })
    const output = JSON.parse(result.output)

    expect(output.error).toBe('Skill directory "missing-skill" not found')
  })

  test("publish_memory stores finalized local skills in global Forecast memory", async () => {
    await writeSkill(
      "linear-models",
      "Linear models for trend dominated series",
      "# Linear Models\n\n## Parameter Space\n- train_epochs: 3",
    )
    await writeFile(
      path.join(tmpdir, ".forecast", "task.json"),
      JSON.stringify(
        {
          dataset_path: "data/project-demand.csv",
          target_col: "demand",
          time_col: "timestamp",
          pred_len: 24,
          seq_len: 48,
          label_len: 24,
          train_ratio: 0.7,
          val_ratio: 0.2,
          test_ratio: 0.1,
          eval_metrics: ["mse", "mae"],
          model_families: ["DLinear"],
          frozen_at: "2026-06-18T00:00:00.000Z",
        },
        null,
        2,
      ),
      "utf-8",
    )

    const result = await run({ subcommand: "publish_memory" })
    const output = JSON.parse(result.output)

    expect(output.published).toBe(true)
    expect(output.count).toBe(1)
    expect(output.skills[0].original_skill_name).toBe("linear-models")

    const memoryContent = await readFile(output.skills[0].path, "utf-8")
    expect(memoryContent).toContain("Forecast Memory: linear-models")
    expect(memoryContent).toContain("Reuse Guidance")
    expect(memoryContent).toContain("project-demand.csv")
    expect(memoryContent).toContain("train_epochs: 3")
    expect(memoryContent).not.toContain("data/project-demand.csv")

    const listResult = JSON.parse((await run({ subcommand: "list_memory" })).output)
    expect(listResult.count).toBe(1)
    expect(listResult.memories[0].name).toBe(output.skills[0].name)
  })

  test("read_memory and import_memory copy a global memory skill into local task skills", async () => {
    await writeSkill("transformers", "Transformer models", "# Transformers\n\n## Parameter Space\n- d_model: 64")

    const published = JSON.parse((await run({ subcommand: "publish_memory" })).output)
    const memoryName = published.skills[0].name
    const readResult = JSON.parse((await run({ subcommand: "read_memory", memory_name: memoryName })).output)
    expect(readResult.content).toContain("Forecast Memory: transformers")

    await run({ subcommand: "delete", skill_name: "transformers" })
    const imported = JSON.parse(
      (await run({
        subcommand: "import_memory",
        memory_name: memoryName,
        target_skill_name: "imported-transformers",
      })).output,
    )

    expect(imported.imported).toBe(true)
    const importedContent = await readFile(skillPath("imported-transformers"), "utf-8")
    expect(importedContent).toContain("name: imported-transformers")
    expect(importedContent).toContain("Forecast Memory: transformers")
  })

  test("imported Forecast memory is available through the normal Skill loader in another workspace", async () => {
    await writeSkill("deep-learning", "Deep learning defaults", "# Deep Learning\n\n## Parameter Space\n- batch_size: 32")

    const published = JSON.parse((await run({ subcommand: "publish_memory" })).output)
    const memoryName = published.skills[0].name
    await Instance.disposeAll()

    const otherDir = await mkdtemp(path.join(os.tmpdir(), "forecast-skill-other-"))
    try {
      await writeFile(
        path.join(otherDir, "adacast.json"),
        `${JSON.stringify({ skills: { paths: [".forecast/skills"] } }, null, 2)}\n`,
        "utf-8",
      )

      await Instance.provide({
        directory: otherDir,
        fn: async () => {
          const tool = await ForecastSkillTool.init()
          await tool.execute(
            {
              subcommand: "import_memory",
              memory_name: memoryName,
              target_skill_name: "imported-memory",
            },
            ctx,
          )
        },
      })
      await Instance.disposeAll()

      const skill = await Instance.provide({
        directory: otherDir,
        fn: async () => Skill.get("imported-memory"),
      })
      expect(skill?.description).toContain("Imported forecast memory")
      expect(skill?.content).toContain("Forecast Memory: deep-learning")
      expect(skill?.content).toContain("batch_size: 32")
    } finally {
      await Instance.disposeAll()
      await rm(otherDir, { recursive: true, force: true })
    }
  })
})
