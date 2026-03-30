from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from transformers import set_seed as transformers_set_seed


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "item"


def batched(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")
    for start_index in range(0, len(items), batch_size):
        yield items[start_index : start_index + batch_size]


def find_first_existing_path(candidates: Iterable[str]) -> Path | None:
    for raw_path in candidates:
        path = Path(os.path.expandvars(raw_path)).expanduser()
        if path.exists():
            return path.resolve()
    return None


def resolve_artifact_source(local_paths: list[str], hf_repo_id: str | None) -> dict[str, Any]:
    local_path = find_first_existing_path(local_paths)
    if local_path is not None:
        return {
            "kind": "local",
            "value": str(local_path),
            "resolved_local_path": str(local_path),
        }
    if hf_repo_id:
        return {
            "kind": "hf_hub",
            "value": hf_repo_id,
            "resolved_local_path": None,
        }
    raise FileNotFoundError(
        "Khong tim thay local path nao hop le va cung khong co hf_repo_id de fallback."
    )


def set_reproducibility(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers_set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def get_package_versions() -> dict[str, str | None]:
    packages = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "numpy",
        "tqdm",
        "sentencepiece",
    ]
    versions: dict[str, str | None] = {}
    for package_name in packages:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def get_git_metadata(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit = None

    try:
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).splitlines()
    except Exception:
        status = []

    return {
        "commit": commit,
        "is_dirty": bool(status),
        "status_short": status,
    }


def collect_environment_metadata(project_root: str | Path) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "captured_at_utc": now.isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
        "package_versions": get_package_versions(),
        "git": get_git_metadata(project_root),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_directory(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2)
