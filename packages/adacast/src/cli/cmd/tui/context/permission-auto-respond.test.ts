import { describe, expect, test } from "bun:test"
import { autoAcceptsPermission, sessionLineage } from "./permission-auto-respond"

describe("TUI permission auto responder", () => {
  test("matches the session itself", () => {
    const autoAccept = new Set(["root"])

    expect(autoAcceptsPermission(autoAccept, [{ id: "root" }], { sessionID: "root" })).toBe(true)
  })

  test("inherits auto accept from a parent session", () => {
    const autoAccept = new Set(["root"])
    const sessions = [
      { id: "root" },
      { id: "child", parentID: "root" },
      { id: "grandchild", parentID: "child" },
    ]

    expect(sessionLineage(sessions, "grandchild")).toEqual(["grandchild", "child", "root"])
    expect(autoAcceptsPermission(autoAccept, sessions, { sessionID: "grandchild" })).toBe(true)
  })

  test("does not match unrelated sessions", () => {
    const autoAccept = new Set(["root"])
    const sessions = [{ id: "root" }, { id: "other" }]

    expect(autoAcceptsPermission(autoAccept, sessions, { sessionID: "other" })).toBe(false)
  })
})
