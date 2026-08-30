#!/usr/bin/env bun
import { $ } from "bun"
import pkg from "../package.json"
import { Script } from "@adacast/script"
import { fileURLToPath } from "url"

const dir = fileURLToPath(new URL("..", import.meta.url))
process.chdir(dir)

const binaries: Record<string, string> = {}
for (const filepath of new Bun.Glob("*/package.json").scanSync({ cwd: "./dist" })) {
  const distPkg = await Bun.file(`./dist/${filepath}`).json()
  if (distPkg.name === pkg.name) {
    continue
  }
  binaries[distPkg.name] = distPkg.version
}
console.log("binaries", binaries)
const version = Object.values(binaries)[0]
const npmOnly = process.argv.includes("--npm-only") || process.env.ADACAST_NPM_ONLY === "1"
const publishDelayMs = Number(process.env.ADACAST_PUBLISH_DELAY_MS ?? "3000")
const publishRetries = Number(process.env.ADACAST_PUBLISH_RETRIES ?? "5")
const rateLimitBackoffMs = Number(process.env.ADACAST_PUBLISH_RATE_LIMIT_BACKOFF_MS ?? "120000")

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

async function isPublished(name: string, expectedVersion: string) {
  try {
    const output = await $`npm view ${name} version --json`.quiet().text()
    const version = JSON.parse(output.trim())
    return version === expectedVersion
  } catch {
    return false
  }
}

async function publishPackage(name: string) {
  if (await isPublished(name, version)) {
    console.log(`skip ${name}@${version}, already published`)
    return
  }

  if (process.platform !== "win32") {
    await $`chmod -R 755 .`.cwd(`./dist/${name}`)
  }

  for (let attempt = 1; attempt <= publishRetries; attempt++) {
    try {
      await $`bun pm pack`.cwd(`./dist/${name}`)
      await $`npm publish *.tgz --access public --tag ${Script.channel}`.cwd(`./dist/${name}`)
      if (publishDelayMs > 0) {
        await sleep(publishDelayMs)
      }
      return
    } catch (error) {
      const stderr =
        typeof error === "object" && error && "stderr" in error ? String((error as { stderr?: unknown }).stderr ?? "") : ""
      const message = `${String(error)}\n${stderr}`
      const retryable = message.includes("E429") || message.includes("Too Many Requests") || message.includes("rate limited")
      if (!retryable || attempt === publishRetries) {
        throw error
      }

      const backoffMs = rateLimitBackoffMs * Math.max(1, attempt)
      console.log(`retry ${name}@${version} after ${backoffMs}ms because of rate limit`)
      await sleep(backoffMs)
    }
  }
}

await $`mkdir -p ./dist/${pkg.name}`
await $`cp -r ./bin ./dist/${pkg.name}/bin`
await $`cp ./script/postinstall.mjs ./dist/${pkg.name}/postinstall.mjs`
await Bun.file(`./dist/${pkg.name}/LICENSE`).write(await Bun.file("../../LICENSE").text())
const readmeSource = await Bun.file("../../README.md").text().catch(() => "")
if (readmeSource) {
  await Bun.file(`./dist/${pkg.name}/README.md`).write(readmeSource)
}

await Bun.file(`./dist/${pkg.name}/package.json`).write(
  JSON.stringify(
    {
      name: pkg.name,
      description: pkg.description,
      bin: {
        [pkg.name]: `./bin/${pkg.name}`,
      },
      scripts: {
        postinstall: "node ./postinstall.mjs",
      },
      version: version,
      license: pkg.license,
      files: ["bin", "postinstall.mjs", "LICENSE", "README.md"],
      optionalDependencies: binaries,
    },
    null,
    2,
  ),
)

for (const [name] of Object.entries(binaries)) {
  await publishPackage(name)
}
await publishPackage(pkg.name)

if (npmOnly) {
  process.exit(0)
}

const image = "registry.example.invalid/adacast/adacast"
const platforms = "linux/amd64,linux/arm64"
const tags = [`${image}:${version}`, `${image}:${Script.channel}`]
const tagFlags = tags.flatMap((t) => ["-t", t])
await $`docker buildx build --platform ${platforms} ${tagFlags} --push .`

// registries
if (!Script.preview) {
  // Calculate SHA values
  const arm64Sha = await $`sha256sum ./dist/adacast-linux-arm64.tar.gz | cut -d' ' -f1`.text().then((x) => x.trim())
  const x64Sha = await $`sha256sum ./dist/adacast-linux-x64.tar.gz | cut -d' ' -f1`.text().then((x) => x.trim())
  const macX64Sha = await $`sha256sum ./dist/adacast-darwin-x64.zip | cut -d' ' -f1`.text().then((x) => x.trim())
  const macArm64Sha = await $`sha256sum ./dist/adacast-darwin-arm64.zip | cut -d' ' -f1`.text().then((x) => x.trim())

  const [pkgver, _subver = ""] = Script.version.split(/(-.*)/, 2)

  // arch
  const binaryPkgbuild = [
    "# Maintainer: AdaCast maintainers",
    "",
    "pkgname='adacast-bin'",
    `pkgver=${pkgver}`,
    `_subver=${_subver}`,
    "options=('!debug' '!strip')",
    "pkgrel=1",
    "pkgdesc='The AI coding agent built for the terminal.'",
    "url='https://example.invalid/AdaCast'",
    "arch=('aarch64' 'x86_64')",
    "license=('MIT')",
    "provides=('adacast')",
    "conflicts=('adacast')",
    "depends=('ripgrep')",
    "",
    `source_aarch64=("\${pkgname}_\${pkgver}_aarch64.tar.gz::https://example.invalid/AdaCast/releases/download/v\${pkgver}\${_subver}/adacast-linux-arm64.tar.gz")`,
    `sha256sums_aarch64=('${arm64Sha}')`,

    `source_x86_64=("\${pkgname}_\${pkgver}_x86_64.tar.gz::https://example.invalid/AdaCast/releases/download/v\${pkgver}\${_subver}/adacast-linux-x64.tar.gz")`,
    `sha256sums_x86_64=('${x64Sha}')`,
    "",
    "package() {",
    '  install -Dm755 ./adacast "${pkgdir}/usr/bin/adacast"',
    '  install -d "${pkgdir}/usr/share/adacast"',
    '  cp -r ./app "${pkgdir}/usr/share/adacast/app"',
    '  cp -r ./python "${pkgdir}/usr/share/adacast/python"',
    "}",
    "",
  ].join("\n")

  for (const [pkg, pkgbuild] of [["adacast-bin", binaryPkgbuild]]) {
    for (let i = 0; i < 30; i++) {
      try {
        await $`rm -rf ./dist/aur-${pkg}`
        await $`git clone ssh://aur@aur.archlinux.org/${pkg}.git ./dist/aur-${pkg}`
        await $`cd ./dist/aur-${pkg} && git checkout master`
        await Bun.file(`./dist/aur-${pkg}/PKGBUILD`).write(pkgbuild)
        await $`cd ./dist/aur-${pkg} && makepkg --printsrcinfo > .SRCINFO`
        await $`cd ./dist/aur-${pkg} && git add PKGBUILD .SRCINFO`
        await $`cd ./dist/aur-${pkg} && git commit -m "Update to v${Script.version}"`
        await $`cd ./dist/aur-${pkg} && git push`
        break
      } catch (e) {
        continue
      }
    }
  }

  // Homebrew formula
  const homebrewFormula = [
    "# typed: false",
    "# frozen_string_literal: true",
    "",
    "# This file was generated by GoReleaser. DO NOT EDIT.",
    "class AdaCast < Formula",
    `  desc "The AI coding agent built for the terminal."`,
    `  homepage "https://example.invalid/AdaCast"`,
    `  version "${Script.version.split("-")[0]}"`,
    "",
    `  depends_on "ripgrep"`,
    "",
    "  on_macos do",
    "    if Hardware::CPU.intel?",
    `      url "https://example.invalid/AdaCast/releases/download/v${Script.version}/adacast-darwin-x64.zip"`,
    `      sha256 "${macX64Sha}"`,
    "",
    "      def install",
    '        bin.install "adacast"',
    '        (share/"adacast").install "app"',
    '        (share/"adacast").install "python"',
    "      end",
    "    end",
    "    if Hardware::CPU.arm?",
    `      url "https://example.invalid/AdaCast/releases/download/v${Script.version}/adacast-darwin-arm64.zip"`,
    `      sha256 "${macArm64Sha}"`,
    "",
    "      def install",
    '        bin.install "adacast"',
    '        (share/"adacast").install "app"',
    '        (share/"adacast").install "python"',
    "      end",
    "    end",
    "  end",
    "",
    "  on_linux do",
    "    if Hardware::CPU.intel? and Hardware::CPU.is_64_bit?",
    `      url "https://example.invalid/AdaCast/releases/download/v${Script.version}/adacast-linux-x64.tar.gz"`,
    `      sha256 "${x64Sha}"`,
    "      def install",
    '        bin.install "adacast"',
    '        (share/"adacast").install "app"',
    '        (share/"adacast").install "python"',
    "      end",
    "    end",
    "    if Hardware::CPU.arm? and Hardware::CPU.is_64_bit?",
    `      url "https://example.invalid/AdaCast/releases/download/v${Script.version}/adacast-linux-arm64.tar.gz"`,
    `      sha256 "${arm64Sha}"`,
    "      def install",
    '        bin.install "adacast"',
    '        (share/"adacast").install "app"',
    '        (share/"adacast").install "python"',
    "      end",
    "    end",
    "  end",
    "end",
    "",
    "",
  ].join("\n")

  const token = process.env.GITHUB_TOKEN
  if (!token) {
    console.error("GITHUB_TOKEN is required to update homebrew tap")
    process.exit(1)
  }
  const tap = `https://x-access-token:${token}@github.com/anonymous-submission/homebrew-tap.git`
  await $`rm -rf ./dist/homebrew-tap`
  await $`git clone ${tap} ./dist/homebrew-tap`
  await Bun.file("./dist/homebrew-tap/adacast.rb").write(homebrewFormula)
  await $`cd ./dist/homebrew-tap && git add adacast.rb`
  await $`cd ./dist/homebrew-tap && git commit -m "Update to v${Script.version}"`
  await $`cd ./dist/homebrew-tap && git push`
}
