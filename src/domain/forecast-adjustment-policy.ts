import path from "node:path"
import z from "zod"
import { Tool } from "./runtime"
import DESCRIPTION from "./forecast-adjustment-policy.txt"
import { AdjustmentPolicy, parseStoredAdjustmentPolicy } from "./types"
import {
  adjustmentRoot,
  ensureAdjustmentDirs,
  forecastPath,
  readJsonFile,
  readPhaseState,
  writeJsonFile,
  writePhaseState,
} from "./forecast-adjustment-common"
import {
  assertNoRetainableValidationCandidate,
  assertUserPriorMirrorResolved,
} from "./forecast-adjustment-acceptance"

const parameters = z.object({
  subcommand: z.enum(["read", "configure", "lock", "mark_no_effective", "confirm_for_critic"]),
  policy: AdjustmentPolicy.partial().optional().describe("Policy fields for configure"),
  reason: z.string().optional().describe("Reason for no-effective-skill or confirmation operations"),
})

function policyPath(): string {
  return path.join(adjustmentRoot(), "policy.json")
}

async function readPolicy(): Promise<z.infer<typeof AdjustmentPolicy>> {
  try {
    return parseStoredAdjustmentPolicy(await readJsonFile(policyPath()))
  } catch {
    return AdjustmentPolicy.parse({})
  }
}

function adjustmentDecisionLocked(state: Awaited<ReturnType<typeof readPhaseState>>): boolean {
  return Boolean(state.adjustment.forecaster_locked && (state.adjustment.active_skill_version || state.adjustment.no_effective_skill))
}

async function assertPolicyMutationAllowed(operation: string): Promise<void> {
  const state = await readPhaseState()
  if (state.phase === "post-forecast") {
    throw new Error(`forecast_adjustment_policy ${operation} is not allowed after entering post-forecast`)
  }
  if (adjustmentDecisionLocked(state)) {
    throw new Error(`forecast_adjustment_policy ${operation} is not allowed after the adjustment decision is locked`)
  }
}

export const ForecastAdjustmentPolicyTool = Tool.define("forecast_adjustment_policy", {
  description: DESCRIPTION,
  parameters,
  async execute(params) {
    await ensureAdjustmentDirs()
    const now = new Date().toISOString()
    const current = await readPolicy()
    let policy = current
    let state = await readPhaseState()
    let result: unknown

    switch (params.subcommand) {
      case "read": {
        result = { policy, adjustment: state.adjustment, policy_path: forecastPath("adjustments", "policy.json") }
        break
      }

      case "configure": {
        await assertPolicyMutationAllowed("configure")
        policy = AdjustmentPolicy.parse({
          ...current,
          ...params.policy,
          enabled: params.policy?.enabled ?? current.enabled,
          mode:
            params.policy?.mode ??
            (params.policy?.enabled === true ? "rule_skill" : params.policy?.enabled === false ? "off" : current.mode),
          configured_at: current.configured_at ?? now,
        })
        await writeJsonFile(policyPath(), policy)
        state = {
          ...state,
          adjustment: {
            ...state.adjustment,
            enabled: policy.enabled,
            mode: policy.mode,
            strategy: policy.strategy,
            policy_path: forecastPath("adjustments", "policy.json"),
          },
        }
        await writePhaseState(
          state,
          `## Adjustment Policy Configured\n\nAdjustment enabled=${policy.enabled}, mode=${policy.mode} at ${now}.`,
        )
        result = { configured: true, policy, adjustment: state.adjustment }
        break
      }

      case "lock": {
        await assertPolicyMutationAllowed("lock")
        policy = AdjustmentPolicy.parse({ ...current, locked_at: now })
        await writeJsonFile(policyPath(), policy)
        state = {
          ...state,
          adjustment: {
            ...state.adjustment,
            enabled: policy.enabled,
            mode: policy.mode,
            strategy: policy.strategy,
            policy_path: forecastPath("adjustments", "policy.json"),
          },
        }
        await writePhaseState(state, `## Adjustment Policy Locked\n\nLocked at ${now}.`)
        result = { locked: true, policy, adjustment: state.adjustment }
        break
      }

      case "mark_no_effective": {
        await assertPolicyMutationAllowed("mark_no_effective")
        await assertNoRetainableValidationCandidate("forecast_adjustment_policy mark_no_effective")
        await assertUserPriorMirrorResolved("forecast_adjustment_policy mark_no_effective")
        const noEffectiveReason = params.reason ?? "No candidate passed validation acceptance criteria."
        state = {
          ...state,
          adjustment: {
            ...state.adjustment,
            enabled: true,
            mode: "rule_skill",
            policy_path: forecastPath("adjustments", "policy.json"),
            forecaster_locked: true,
            no_effective_skill: true,
            no_effective_skill_reason: noEffectiveReason,
            active_skill_version: null,
            active_skill_path: null,
            test_collection_run_ids: [],
            test_eligible_run_ids: [],
            test_application_records: [],
            test_application_decision: "skip",
            test_application_decision_at: now,
            test_application_decision_actor: state.adjustment.user_decision_actor,
            test_application_decision_reason: noEffectiveReason,
            test_application_run_id: null,
            test_adjustment_locked: false,
            test_adjustment_run_id: null,
            test_adjustment_candidate_id: null,
          },
        }
        await writePhaseState(
          state,
          `## Adjustment Marked No-Effective-Skill\n\n${state.adjustment.no_effective_skill_reason}`,
        )
        result = { marked: true, adjustment: state.adjustment }
        break
      }

      case "confirm_for_critic": {
        if (state.adjustment.test_adjustment_locked) {
          throw new Error("forecast_adjustment_policy confirm_for_critic is too late after test adjustment values are locked")
        }
        if (!state.adjustment.forecaster_locked || !state.adjustment.active_skill_version) {
          throw new Error("forecast_adjustment_policy confirm_for_critic requires a locked active adjustment skill")
        }
        state = {
          ...state,
          adjustment: {
            ...state.adjustment,
            critic_user_confirmed: now,
          },
        }
        await writePhaseState(
          state,
          `## Adjustment Confirmed For Critic\n\nConfirmed at ${now}. ${params.reason ?? ""}`.trim(),
        )
        result = { confirmed: true, adjustment: state.adjustment }
        break
      }
    }

    return {
      title: `Forecast adjustment policy: ${params.subcommand}`,
      metadata: { subcommand: params.subcommand },
      output: JSON.stringify(result),
    }
  },
})
