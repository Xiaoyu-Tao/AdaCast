;(function () {
  var key = "adacast-theme-id"
  var themeId = localStorage.getItem(key) || "tokyonight"

  if (themeId === "oc-1" || themeId === "oc-2") {
    themeId = "adacast"
    localStorage.setItem(key, themeId)
    localStorage.removeItem("adacast-theme-css-light")
    localStorage.removeItem("adacast-theme-css-dark")
  }

  var scheme = localStorage.getItem("adacast-color-scheme") || "system"
  var isDark = scheme === "dark" || (scheme === "system" && matchMedia("(prefers-color-scheme: dark)").matches)
  var mode = isDark ? "dark" : "light"

  document.documentElement.dataset.theme = themeId
  document.documentElement.dataset.colorScheme = mode

  if (themeId === "adacast") return

  var css = localStorage.getItem("adacast-theme-css-" + mode)
  if (css) {
    var style = document.createElement("style")
    style.id = "adacast-theme-preload"
    style.textContent =
      ":root{color-scheme:" +
      mode +
      ";--text-mix-blend-mode:" +
      (isDark ? "plus-lighter" : "multiply") +
      ";" +
      css +
      "}"
    document.head.appendChild(style)
  }
})()
