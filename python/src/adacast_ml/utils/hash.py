"""SHA-256 hashing utilities for spec identity and evaluator integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EVALUATOR_PATH = Path(__file__).resolve().parent.parent / "evaluator.py"


def canonical_spec_json(spec: dict) -> str:
    """Return the cross-runtime canonical JSON used for experiment identity."""
    return json.dumps(
        spec,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def compute_spec_hash(spec: dict) -> str:
    """Compute a stable SHA-256 digest for a JSON-serializable spec."""
    canonical = canonical_spec_json(spec)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_file_hash(path: Path) -> str:
    """Compute a SHA-256 digest for a file's raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_evaluator(evaluator_path: Path, store_path: Path) -> str:
    """Persist and verify the evaluator hash to prevent evaluation drift."""
    current_hash = compute_file_hash(evaluator_path)

    if not store_path.exists():
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text(current_hash, encoding="utf-8")
        return current_hash

    stored_hash = store_path.read_text(encoding="utf-8").strip()
    if current_hash != stored_hash:
        raise SystemExit(
            "INTEGRITY VIOLATION: evaluator.py has been modified.\n"
            f"  Stored hash: {stored_hash}\n"
            f"  Current hash: {current_hash}\n"
            "Evaluation aborted to prevent evaluation drift."
        )

    return current_hash
