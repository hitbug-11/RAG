"""Download pinned project models into the server project cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path("/data/haojiachen/rag")
MODELS = [
    {
        "repo_id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    },
    {
        "repo_id": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    },
]
REQUIRED_PROJECT_PATHS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_XET_CACHE",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
    "TMPDIR",
)


def require_project_path(variable_name: str) -> Path:
    raw_value = os.environ.get(variable_name)
    if not raw_value:
        raise RuntimeError(f"{variable_name} must be set by run_server_python.sh")
    path = Path(raw_value).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise RuntimeError(f"{variable_name} escapes the project root: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def directory_stats(snapshot_path: Path) -> dict[str, Any]:
    files = [path for path in snapshot_path.rglob("*") if path.is_file()]
    return {
        "file_count": len(files),
        "logical_size_bytes": sum(path.stat().st_size for path in files),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/server_model_downloads.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_paths = {
        variable_name: str(require_project_path(variable_name))
        for variable_name in REQUIRED_PROJECT_PATHS
    }
    hub_cache = Path(resolved_paths["HF_HUB_CACHE"])

    downloads = []
    for model in MODELS:
        snapshot_path = Path(
            snapshot_download(
                repo_id=model["repo_id"],
                revision=model["revision"],
                cache_dir=hub_cache,
            )
        ).resolve()
        if not snapshot_path.is_relative_to(PROJECT_ROOT):
            raise AssertionError(f"Downloaded snapshot escaped project root: {snapshot_path}")
        downloads.append(
            {
                **model,
                "snapshot_path": str(snapshot_path),
                **directory_stats(snapshot_path),
            }
        )

    summary = {
        "project_root": str(PROJECT_ROOT),
        "environment_paths": resolved_paths,
        "downloads": downloads,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
