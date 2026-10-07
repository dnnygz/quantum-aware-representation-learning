"""Shared filesystem, configuration, logging, and provenance helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


LOGGER = logging.getLogger("tcga_brca_data")


def configure_logging(verbose: bool = False) -> None:
    """Configure consistent command-line logging."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and minimally validate the declarative data configuration.

    Returns the configuration and repository root. Paths in the YAML are
    interpreted relative to the repository root, not the caller's CWD.
    """

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")

    required = {"project_id", "gdc", "query", "clinical", "expression", "splits", "paths"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Configuration missing sections: {missing}")

    repo_root = config_path.parent.parent
    return config, repo_root


def resolve_path(repo_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else repo_root / path


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def write_json_atomic(path: Path, value: Any) -> None:
    """Atomically write JSON to a mutable generated-data location."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json_immutable(path: Path, value: Any) -> None:
    """Create an immutable JSON artifact or verify that it is unchanged."""

    payload = canonical_json_bytes(value)
    if path.exists():
        existing = canonical_json_bytes(json.loads(path.read_text(encoding="utf-8")))
        if existing != payload:
            raise FileExistsError(
                f"Refusing to overwrite immutable raw metadata with different content: {path}. "
                "A new GDC release must use a new snapshot directory."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    # MD5 is used only to match the integrity checksum supplied by GDC.
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_slug(status: Mapping[str, Any]) -> str:
    version = status.get("data_release_version", {})
    major = version.get("major")
    minor = version.get("minor")
    if major is None:
        raise ValueError("GDC /status did not include data_release_version.major")
    return f"gdc_release_{major}.{0 if minor is None else minor}"


def git_state(repo_root: Path) -> dict[str, Any]:
    """Return Git revision metadata without requiring a clean worktree."""

    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def dependency_versions(names: list[str] | None = None) -> dict[str, str | None]:
    names = names or ["numpy", "pandas", "pyarrow", "PyYAML", "requests", "scikit-learn"]
    result: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result
