import { NamedError } from "@adacast/util/error"
import z from "zod"

export const TaskFrozenError = NamedError.create(
  "TaskFrozenError",
  z.object({
    path: z.string(),
  }),
)

export const TaskNotFoundError = NamedError.create(
  "TaskNotFoundError",
  z.object({
    expected_path: z.string(),
  }),
)

export const SpecDuplicateError = NamedError.create(
  "SpecDuplicateError",
  z.object({
    run_id: z.string(),
    eval_path: z.string(),
  }),
)

export const IPCError = NamedError.create(
  "IPCError",
  z.object({
    reason: z.string(),
    stderr_tail: z.string().optional(),
  }),
)

export const PhaseGateError = NamedError.create(
  "PhaseGateError",
  z.object({
    current_phase: z.string(),
    required_phase: z.string(),
    missing_preconditions: z.array(z.string()),
  }),
)

export const SkillNotConfirmedError = NamedError.create(
  "SkillNotConfirmedError",
  z.object({
    state_path: z.string(),
  }),
)
