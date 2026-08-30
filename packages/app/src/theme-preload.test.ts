import { beforeEach, describe, expect, test } from "bun:test"

const src = await Bun.file(new URL("../public/theme-preload.js", import.meta.url)).text()

const run = () => Function(src)()

beforeEach(() => {
  document.head.innerHTML = ""
  document.documentElement.removeAttribute("data-theme")
  document.documentElement.removeAttribute("data-color-scheme")
  localStorage.clear()
  Object.defineProperty(window, "matchMedia", {
    value: () =>
      ({
        matches: false,
      }) as MediaQueryList,
    configurable: true,
  })
})

describe("theme preload", () => {
  test("uses tokyonight when no theme is saved", () => {
    run()

    expect(document.documentElement.dataset.theme).toBe("tokyonight")
    expect(document.documentElement.dataset.colorScheme).toBe("light")
  })

  test("migrates legacy themes to adacast before mount", () => {
    localStorage.setItem("adacast-theme-id", "oc-1")
    localStorage.setItem("adacast-theme-css-light", "--background-base:#fff;")
    localStorage.setItem("adacast-theme-css-dark", "--background-base:#000;")

    run()

    expect(document.documentElement.dataset.theme).toBe("adacast")
    expect(document.documentElement.dataset.colorScheme).toBe("light")
    expect(localStorage.getItem("adacast-theme-id")).toBe("adacast")
    expect(localStorage.getItem("adacast-theme-css-light")).toBeNull()
    expect(localStorage.getItem("adacast-theme-css-dark")).toBeNull()
    expect(document.getElementById("adacast-theme-preload")).toBeNull()
  })

  test("keeps cached css for non-default themes", () => {
    localStorage.setItem("adacast-theme-id", "nightowl")
    localStorage.setItem("adacast-theme-css-light", "--background-base:#fff;")

    run()

    expect(document.documentElement.dataset.theme).toBe("nightowl")
    expect(document.getElementById("adacast-theme-preload")?.textContent).toContain("--background-base:#fff;")
  })
})
