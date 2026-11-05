"""Shared evidence-candidate boundaries and leakage-safe data knowledge loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EVIDENCE_CANDIDATE_BOUNDARIES = {
    "experience": (
        "Historically or validation-derived guidance about what adjustment worked under stated conditions. "
        "It may guide this sample, but it is not a current observation or a timeless fact."
    ),
    "context": (
        "Time-local facts for this forecast sample, including current/future-known exogenous values, "
        "calendar or event state, and their changes. It must not be generalized into a reusable rule here."
    ),
    "knowledge": (
        "Stable domain/task semantics and empirical data analysis: qualitative findings, descriptive statistics, "
        "trend, seasonality, stationarity, volatility, data quality, and distribution structure. Empirical data "
        "knowledge must retain its analysis scope and uncertainty; it excludes model residual/performance findings "
        "and the current sample's changing values."
    ),
}

MAX_QUALITATIVE_ANALYSIS_CHARS = 20_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data_analysis_knowledge(
    task: dict[str, Any],
    task_path: str | Path,
) -> dict[str, Any]:
    """Load pre-forecast analysis as scoped empirical knowledge, never as model experience."""
    task_file = Path(task_path)
    reports_dir = task_file.parent / "reports"
    result: dict[str, Any] = {
        "epistemic_status": "empirical_data_knowledge",
        "boundary": (
            "Descriptive and structural findings may inform forecasting within their recorded scope. "
            "Model rankings, residual behavior, correction magnitudes, and validation gains remain experience. "
            "Current-window values remain context."
        ),
        "quantitative_analysis": None,
        "qualitative_analysis": None,
    }

    quantitative_path = reports_dir / "quantitative.json"
    if quantitative_path.is_file():
        try:
            payload = json.loads(quantitative_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            payload = None
        if (
            isinstance(payload, dict)
            and str(payload.get("target_col", "")) == str(task.get("target_col", ""))
            and str(payload.get("time_col", "")) == str(task.get("time_col", ""))
        ):
            result["quantitative_analysis"] = {
                "source_path": str(quantitative_path),
                "source_sha256": _sha256(quantitative_path),
                "analysis_scope": payload.get("analysis_scope"),
                "analysis": payload.get("analysis"),
                "split_time_ranges": payload.get("split_time_ranges"),
                "analyzer_version": payload.get("analyzer_version"),
                "analysis_timestamp": payload.get("analysis_timestamp"),
            }

    qualitative_path = reports_dir / "pre-forecast.md"
    if not qualitative_path.is_file():
        fallback_path = reports_dir / "qualitative-research.md"
        qualitative_path = fallback_path if fallback_path.is_file() else qualitative_path
    if qualitative_path.is_file():
        try:
            content = qualitative_path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError):
            content = ""
        if content.strip():
            result["qualitative_analysis"] = {
                "source_path": str(qualitative_path),
                "source_sha256": _sha256(qualitative_path),
                "content": content[:MAX_QUALITATIVE_ANALYSIS_CHARS],
                "truncated": len(content) > MAX_QUALITATIVE_ANALYSIS_CHARS,
                "usage_constraint": (
                    "Treat domain and dataset-structure conclusions as candidate knowledge. Treat any model ranking, "
                    "expected metric, or adjustment recommendation in this prose as an unvalidated hypothesis, not knowledge."
                ),
            }

    return result
