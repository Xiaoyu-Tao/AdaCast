import { describe, expect, test } from "bun:test"
import { readFileSync } from "fs"
import path from "path"

const promptPath = path.join(import.meta.dir, "index.tsx")

function promptSource() {
  return readFileSync(promptPath, "utf-8")
}

describe("TUI prompt Auto Mode entry", () => {
  test("exposes Auto Mode as a new-session prompt control", () => {
    const source = promptSource()

    expect(source).toContain("AUTO_MODE_SYSTEM_PROMPT")
    expect(source).toContain("Planner -> Forecaster -> Critic")
    expect(source).toContain("autoMode: false")
    expect(source).toContain("prompt.auto_mode.toggle")
    expect(source).toContain('enabled: !props.sessionID && store.mode === "normal"')
    expect(source).toContain('<Show when={!props.sessionID && store.mode === "normal"}>')
    expect(source).toContain("Auto Mode")
    expect(source).toContain("Planner -> Forecaster -> Critic")
    expect(source).toContain("strict forecast_state phase sequence")
    expect(source).toContain("transition to skills-review")
    expect(source).toContain("Never edit .forecast/STATE.md directly")
    expect(source).toContain("Forecaster HITL")
    expect(source).toContain("explicitly out-of-scope user verification checkpoint")
  })

  test("threads Auto Mode only into new normal session prompts", () => {
    const source = promptSource()

    expect(source).toContain('const autoMode = isNewSession && currentMode === "normal" && store.autoMode')
    expect(source).toContain("if (autoMode) sync.permission.enableAutoAccept(sessionID)")
    expect(source).toContain("system: autoMode ? AUTO_MODE_SYSTEM_PROMPT : undefined")
  })
})
