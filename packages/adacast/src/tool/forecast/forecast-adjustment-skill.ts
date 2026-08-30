import { copyFile, mkdir, readdir, readFile } from "fs/promises"
import path from "path"
import z from "zod"
import { Filesystem } from "@/util/filesystem"
import { Tool } from "../tool"
import DESCRIPTION from "./forecast-adjustment-skill.txt"
import { AdjustmentCandidate, AdjustmentEvalResult, AdjustmentSkillManifest, AdjustmentSourceModelSet } from "./types"
import {
  adjustmentRoot,
  forecastPath,
  projectRelative,
  readJsonFile,
  readPhaseState,
  runIdIsSafe,
  writeJsonFile,
  writePhaseState,
} from "./forecast-adjustment-common"
import {
  collectAcceptedValidationResultPaths,
  collectCompletedValidationResults,
  isUserPriorMirrorCandidate,
  leakageAuditPassed,
  taskRequiresUserPriorMirror,
  validationAuditPath,
  validationEvalPath,
} from "./forecast-adjustment-acceptance"

const parameters = z.object({
  subcommand: z.enum(["list_versions", "read_active", "accept_candidate", "reject_candidate", "lock_active"]),
  candidate_id: z.string().optional(),
  run_id: z.string().optional().describe("Validation run id whose adjusted_eval.json proves acceptance"),
  reason: z.string().optional(),
})

function candidateDir(candidateId: string): string {
  return path.join(adjustmentRoot(), "candidates", candidateId)
}

function versionsRoot(): string {
  return path.join(adjustmentRoot(), "skills", "versions")
}

function activeRoot(): string {
  return path.join(adjustmentRoot(), "skills", "active")
}

async function nextVersion(): Promise<string> {
  const entries = await readdir(versionsRoot(), { withFileTypes: true }).catch(() => [])
  const latest = entries
    .filter((entry) => entry.isDirectory() && /^v\d+$/.test(entry.name))
    .map((entry) => Number(entry.name.slice(1)))
    .reduce((max, value) => Math.max(max, value), 0)
  return `v${String(latest + 1).padStart(3, "0")}`
}

async function writeCandidateStatus(candidateId: string, status: "accepted" | "rejected") {
  const file = path.join(candidateDir(candidateId), "candidate.json")
  const candidate = AdjustmentCandidate.parse(await readJsonFile(file))
  await writeJsonFile(file, { ...candidate, status })
}

async function assertSkillMutationAllowed(operation: string): Promise<void> {
  const state = await readPhaseState()
  if (state.phase === "post-forecast") {
    throw new Error(`forecast_adjustment_skill ${operation} is not allowed after entering post-forecast`)
  }
  if (state.adjustment.forecaster_locked && (state.adjustment.active_skill_version || state.adjustment.no_effective_skill)) {
    throw new Error(`forecast_adjustment_skill ${operation} is not allowed after the adjustment decision is locked`)
  }
}

async function readSourceSet(): Promise<z.infer<typeof AdjustmentSourceModelSet>> {
  return AdjustmentSourceModelSet.parse(await readJsonFile(path.join(adjustmentRoot(), "source-model-set.json")))
}

async function readActiveManifest(): Promise<z.infer<typeof AdjustmentSkillManifest> | null> {
  const manifestPath = path.join(activeRoot(), "manifest.json")
  if (!(await Filesystem.exists(manifestPath))) return null
  return AdjustmentSkillManifest.parse(await readJsonFile(manifestPath))
}

export const ForecastAdjustmentSkillTool = Tool.define("forecast_adjustment_skill", {
  description: DESCRIPTION,
  parameters,
  async execute(params) {
    let result: unknown

    switch (params.subcommand) {
      case "list_versions": {
        const entries = await readdir(versionsRoot(), { withFileTypes: true }).catch(() => [])
        result = { versions: entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort() }
        break
      }

      case "read_active": {
        const manifestPath = path.join(activeRoot(), "manifest.json")
        result = {
          manifest: (await Filesystem.exists(manifestPath)) ? await readJsonFile(manifestPath) : null,
          skill_markdown: await readFile(path.join(activeRoot(), "SKILL.md"), "utf-8").catch(() => null),
        }
        break
      }

      case "reject_candidate": {
        if (!params.candidate_id) throw new Error("reject_candidate requires candidate_id")
        if (!runIdIsSafe(params.candidate_id)) throw new Error(`Invalid candidate_id "${params.candidate_id}"`)
        await assertSkillMutationAllowed("reject_candidate")
        const candidate = AdjustmentCandidate.parse(
          await readJsonFile(path.join(candidateDir(params.candidate_id), "candidate.json")),
        )
        const sourceSet = await readSourceSet()
        const completedValidationResults =
          candidate.source_model_set_id === sourceSet.source_model_set_id
            ? await collectCompletedValidationResults(params.candidate_id, sourceSet)
            : []
        if (
          candidate.source_model_set_id === sourceSet.source_model_set_id &&
          completedValidationResults.some((item) => item.result.acceptance_decision?.accepted)
        ) {
          throw new Error(
            "Candidate cannot be rejected because it already has a validation-accepted result; " +
              "accept and lock it instead, including when it had zero validation triggers",
          )
        }
        if (isUserPriorMirrorCandidate(candidate) && completedValidationResults.length === 0) {
          throw new Error(
            "User-prior mirror cannot be rejected before a held-out validation-back evaluation with a passing leakage audit",
          )
        }
        await writeCandidateStatus(params.candidate_id, "rejected")
        result = { rejected: true, candidate_id: params.candidate_id, reason: params.reason ?? null }
        break
      }

      case "accept_candidate": {
        if (!params.candidate_id) throw new Error("accept_candidate requires candidate_id")
        if (!params.run_id) throw new Error("accept_candidate requires validation run_id")
        if (!runIdIsSafe(params.candidate_id)) throw new Error(`Invalid candidate_id "${params.candidate_id}"`)
        if (!runIdIsSafe(params.run_id)) throw new Error(`Invalid run_id "${params.run_id}"`)
        await assertSkillMutationAllowed("accept_candidate")

        const candidate = AdjustmentCandidate.parse(
          await readJsonFile(path.join(candidateDir(params.candidate_id), "candidate.json")),
        )
        if (candidate.status === "accepted") {
          throw new Error(`Candidate ${params.candidate_id} is already accepted`)
        }
        if ((await taskRequiresUserPriorMirror()) && !isUserPriorMirrorCandidate(candidate)) {
          throw new Error(
            "Tasks with explicit expert experience must retain the first user_prior_mirror as the active test-time skill; " +
              "a later exploratory candidate cannot replace it",
          )
        }
        const activeManifest = await readActiveManifest()
        if (activeManifest?.status === "active" && candidate.parent_skill_version !== activeManifest.version) {
          throw new Error(
            `Candidate parent_skill_version ${candidate.parent_skill_version ?? "null"} does not match active skill ${activeManifest.version}`,
          )
        }
        const sourceSet = await readSourceSet()
        if (sourceSet.source_model_set_id !== candidate.source_model_set_id) {
          throw new Error("Candidate cannot be accepted because it does not match the frozen source model set")
        }
        if (!sourceSet.members.some((member) => member.validation_run_id === params.run_id)) {
          throw new Error("Candidate cannot be accepted from a validation run outside the frozen source model set")
        }
        const requestedValidationEvalPath = validationEvalPath(params.run_id, params.candidate_id)
        const validationEval = AdjustmentEvalResult.parse(await readJsonFile(requestedValidationEvalPath))
        if (
          validationEval.run_id !== params.run_id ||
          validationEval.candidate_id !== params.candidate_id ||
          validationEval.split !== "val"
        ) {
          throw new Error("Candidate validation result does not match the requested validation run and candidate")
        }
        if (!validationEval.acceptance_decision?.accepted) {
          throw new Error("Candidate cannot be accepted because validation acceptance_decision.accepted is not true")
        }
        const auditPath = validationAuditPath(params.run_id, params.candidate_id)
        if (!(await Filesystem.exists(auditPath)) || !(await leakageAuditPassed(auditPath))) {
          throw new Error("Candidate cannot be accepted because the validation leakage audit did not pass")
        }
        const validationResults = await collectAcceptedValidationResultPaths(params.candidate_id, sourceSet)
        if (!validationResults.includes(projectRelative(requestedValidationEvalPath))) {
          throw new Error("Candidate cannot be accepted because the requested validation result was not collected as accepted")
        }

        const version = await nextVersion()
        const versionDir = path.join(versionsRoot(), version)
        await mkdir(versionDir, { recursive: true })
        await mkdir(activeRoot(), { recursive: true })

        const acceptedAt = new Date().toISOString()
        const manifest = AdjustmentSkillManifest.parse({
          version,
          status: "active",
          candidate_id: params.candidate_id,
          source_model_set_id: candidate.source_model_set_id,
          accepted_at: acceptedAt,
          locked_at: null,
          validation_results: validationResults,
          policy_path: forecastPath("adjustments", "policy.json"),
          parent_version: candidate.parent_skill_version,
        })

        await writeJsonFile(path.join(versionDir, "manifest.json"), manifest)
        await copyFile(path.join(candidateDir(params.candidate_id), "SKILL.md"), path.join(versionDir, "SKILL.md"))
        await writeJsonFile(path.join(activeRoot(), "manifest.json"), manifest)
        await copyFile(path.join(versionDir, "SKILL.md"), path.join(activeRoot(), "SKILL.md"))
        await writeCandidateStatus(params.candidate_id, "accepted")

        const state = await readPhaseState()
        await writePhaseState(
          {
            ...state,
            adjustment: {
              ...state.adjustment,
              enabled: true,
              mode: "rule_skill",
              active_skill_version: version,
              active_skill_path: forecastPath("adjustments", "skills", "active", "SKILL.md"),
              no_effective_skill: false,
              no_effective_skill_reason: null,
            },
          },
          `## Adjustment Skill Accepted\n\nAccepted ${params.candidate_id} as ${version} at ${acceptedAt}.`,
        )

        result = { accepted: true, version, manifest }
        break
      }

      case "lock_active": {
        await assertSkillMutationAllowed("lock_active")
        const manifestPath = path.join(activeRoot(), "manifest.json")
        const manifest = AdjustmentSkillManifest.parse(await readJsonFile(manifestPath))
        if (manifest.status !== "active") {
          throw new Error(`Active adjustment skill must be active before locking; current status is ${manifest.status}`)
        }
        const lockedAt = new Date().toISOString()
        const locked = AdjustmentSkillManifest.parse({ ...manifest, status: "locked", locked_at: lockedAt })
        await writeJsonFile(manifestPath, locked)
        await writeJsonFile(path.join(versionsRoot(), locked.version, "manifest.json"), locked)
        const state = await readPhaseState()
        await writePhaseState(
          {
            ...state,
            adjustment: {
              ...state.adjustment,
              enabled: true,
              mode: "rule_skill",
              active_skill_version: locked.version,
              active_skill_path: forecastPath("adjustments", "skills", "active", "SKILL.md"),
              forecaster_locked: true,
            },
          },
          `## Adjustment Skill Locked\n\nLocked ${locked.version} at ${lockedAt}.`,
        )
        result = { locked: true, manifest: locked }
        break
      }
    }

    return {
      title: `Forecast adjustment skill: ${params.subcommand}`,
      metadata: { subcommand: params.subcommand, candidate_id: params.candidate_id },
      output: JSON.stringify(result),
    }
  },
})
