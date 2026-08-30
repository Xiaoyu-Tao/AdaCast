import { describe, expect, test } from "bun:test"
import { readFileSync } from "fs"
import path from "path"

const sidebarPath = path.join(import.meta.dir, "sidebar.tsx")

function sidebarSource() {
  return readFileSync(sidebarPath, "utf-8")
}

describe("session sidebar forecast progress panel", () => {
  test("uses the shared progress resource instead of the legacy Auto Mode block", () => {
    const source = sidebarSource()

    expect(source).toContain("progress: true")
    expect(source).toContain("readForecastProgress")
    expect(source).toContain("defaultForecastProgress")
    expect(source).toContain("<b>Progress</b>")
    expect(source).not.toContain("<b>Auto Mode</b>")
    expect(source).toContain("Auto Mode paused:")
    expect(source).toContain("Auto Mode handoff:")
    expect(source).toMatch(/<b>Progress<\/b>[\s\S]*<b>Context<\/b>/)
  })

  test("renders fixed-order progress rows with wrapped details", () => {
    const source = sidebarSource()

    expect(source).toContain("function progressSymbol")
    expect(source).toContain("function ProgressRow")
    expect(source).toContain("<For each={progress().rows}")
    expect(source).toContain('wrapMode="word"')
    expect(source).not.toMatch(/handoffs\.map|handoff history|Handoff history/)
  })

  test("refreshes the progress resource from synced forecast tool part metadata", () => {
    const source = sidebarSource()

    expect(source).toContain("extractForecastLiveEvents")
    expect(source).toContain("sync.data.part")
    expect(source).toContain("forecastToolParts")
    expect(source).toContain("forecastLiveEvents")
    expect(source).toContain("forecastToolDigest")
    expect(source).toContain("toolDigest: forecastToolDigest()")
    expect(source).toContain("readForecastProgress(input.directory, forecastLiveEvents())")
    expect(source).not.toMatch(/activeTab|selectedTab|currentTab/)
  })
})
