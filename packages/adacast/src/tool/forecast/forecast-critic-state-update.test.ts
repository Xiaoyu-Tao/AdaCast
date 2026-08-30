import { describe, expect, test } from "bun:test"
import { mkdir } from "fs/promises"
import path from "path"
import { tmpdir } from "../../../test/fixture/fixture"
import { readJsonFile, writeJsonFile } from "./forecast-adjustment-common"
import {
  commitCriticEvidenceTransition,
  commitCriticGap,
  commitCriticSafetyTransition,
} from "./forecast-critic-state-update"
import { computeFileSha256 } from "./model-artifact"

describe("critic state update", () => {
  test("commits an append-only content-addressed evidence transition", async () => {
    await using tmp = await tmpdir()
    const statesDir = path.join(tmp.path, "states", "window-000007")
    await mkdir(statesDir, { recursive: true })
    const initial = {
      version: 1 as const,
      window_index: 7,
      evidences: [],
      predictions: [{ horizon_index: 0, timestamp: "2026-01-01 00:00:00", value: 10 }],
      evidence_gap: 1 as const,
      needed_evidence: "Need current-window context.",
    }
    const initialPath = path.join(statesDir, "initial.json")
    const candidatePath = path.join(tmp.path, "context.json")
    await writeJsonFile(initialPath, initial)
    await writeJsonFile(candidatePath, { candidate_type: "context", evidence_candidate_id: "context:7" })
    const initialArtifact = { path: initialPath, sha256: await computeFileSha256(initialPath) }
    const candidateArtifact = { path: candidatePath, sha256: await computeFileSha256(candidatePath) }

    const committed = await commitCriticEvidenceTransition({
      statesDir,
      previousState: initial,
      previousArtifact: initialArtifact,
      selectedEvidence: {
        evidence_candidate_id: "context:7",
        candidate_type: "context",
        artifact_path: candidatePath,
        artifact_sha256: candidateArtifact.sha256,
      },
      expectedCandidateArtifact: candidateArtifact,
      predictions: [{ horizon_index: 0, timestamp: "2026-01-01 00:00:00", value: 12 }],
      evidenceGap: 0,
      neededEvidence: "",
    })

    expect(committed.state.evidences.map((item) => item.evidence_candidate_id)).toEqual(["context:7"])
    expect(committed.state.predictions[0]?.value).toBe(12)
    expect(await readJsonFile<{ path: string; sha256: string }>(path.join(statesDir, "current.json"))).toEqual(
      committed.artifact,
    )
    expect(await computeFileSha256(committed.artifact.path)).toBe(committed.artifact.sha256)
  })

  test("rejects duplicate evidence, layout changes, and inconsistent gaps", async () => {
    await using tmp = await tmpdir()
    const statesDir = path.join(tmp.path, "states")
    await mkdir(statesDir, { recursive: true })
    const candidatePath = path.join(tmp.path, "experience.json")
    await writeJsonFile(candidatePath, { candidate_type: "experience", evidence_candidate_id: "experience:1" })
    const candidateArtifact = { path: candidatePath, sha256: await computeFileSha256(candidatePath) }
    const state = {
      version: 1 as const,
      window_index: 1,
      evidences: [
        {
          evidence_candidate_id: "experience:1",
          candidate_type: "experience" as const,
          artifact_path: candidatePath,
          artifact_sha256: candidateArtifact.sha256,
        },
      ],
      predictions: [{ horizon_index: 0, timestamp: "t0", value: 1 }],
      evidence_gap: 1 as const,
      needed_evidence: "Need context.",
    }
    const statePath = path.join(statesDir, "previous.json")
    await writeJsonFile(statePath, state)
    const previousArtifact = { path: statePath, sha256: await computeFileSha256(statePath) }
    const transition = {
      statesDir,
      previousState: state,
      previousArtifact,
      selectedEvidence: state.evidences[0]!,
      expectedCandidateArtifact: candidateArtifact,
      predictions: state.predictions,
      evidenceGap: 1 as const,
      neededEvidence: "Need context.",
    }
    await expect(commitCriticEvidenceTransition(transition)).rejects.toThrow("more than once")
    await expect(
      commitCriticGap({ ...transition, evidenceGap: 1, neededEvidence: "" }),
    ).rejects.toThrow("must describe")
  })

  test("commits a post-loop safety transition without changing evidence or gap state", async () => {
    await using tmp = await tmpdir()
    const statesDir = path.join(tmp.path, "states")
    await mkdir(statesDir, { recursive: true })
    const state = {
      version: 1 as const,
      window_index: 2,
      evidences: [],
      predictions: [{ horizon_index: 0, timestamp: "t0", value: -1 }],
      evidence_gap: 0 as const,
      needed_evidence: "",
    }
    const statePath = path.join(statesDir, "before-safety.json")
    await writeJsonFile(statePath, state)
    const previousArtifact = { path: statePath, sha256: await computeFileSha256(statePath) }

    const committed = await commitCriticSafetyTransition({
      statesDir,
      previousState: state,
      previousArtifact,
      predictions: [{ horizon_index: 0, timestamp: "t0", value: 0 }],
    })

    expect(committed.state.predictions[0]?.value).toBe(0)
    expect(committed.state.evidences).toEqual([])
    expect(committed.state.evidence_gap).toBe(0)
    expect(committed.state.needed_evidence).toBe("")
  })
})
