"""Hugging Face cache inspection helpers for foundation model diagnostics."""

from __future__ import annotations


def inspect_repo_cache(repo_id: str, hub_cache: str | None) -> dict[str, str]:
    """Inspect whether a Hugging Face repo is already present in the local cache."""
    try:
        from huggingface_hub import scan_cache_dir
    except Exception as exc:  # noqa: BLE001
        return {
            "type": "cache_warning",
            "code": "cache_scan_failed",
            "message": f"Could not inspect Hugging Face cache for {repo_id}: {exc}",
        }

    try:
        kwargs = {"cache_dir": hub_cache} if hub_cache else {}
        cache_info = scan_cache_dir(**kwargs)
        for repo in getattr(cache_info, "repos", []):
            if getattr(repo, "repo_id", None) == repo_id:
                return {"type": "cache_hit", "repo_id": repo_id}
        return {"type": "cache_miss", "repo_id": repo_id}
    except Exception as exc:  # noqa: BLE001
        return {
            "type": "cache_warning",
            "code": "cache_scan_failed",
            "message": f"Could not inspect Hugging Face cache for {repo_id}: {exc}",
        }


__all__ = ["inspect_repo_cache"]
