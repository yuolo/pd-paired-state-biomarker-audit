"""Small caching helpers for reproducible command-line pipeline steps."""

from __future__ import annotations

import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def file_exists_and_nonempty(path: str | Path) -> bool:
    """Return True if a file exists and has nonzero size."""

    file_path = Path(path)
    return file_path.exists() and file_path.is_file() and file_path.stat().st_size > 0


def output_is_newer_than_inputs(output: str | Path, inputs: Iterable[str | Path]) -> bool:
    """Return True when output exists and is newer than every existing input."""

    output_path = Path(output)
    if not file_exists_and_nonempty(output_path):
        return False
    output_mtime = output_path.stat().st_mtime
    for input_path in [Path(item) for item in inputs]:
        if input_path.exists() and input_path.stat().st_mtime > output_mtime:
            return False
    return True


def compute_file_hash(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA256 hash for a local file."""

    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return ""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten_existing(paths: Iterable[str | Path]) -> list[Path]:
    """Return existing path list, expanding directories one level recursively."""

    result: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            result.extend(sorted(child for child in path.rglob("*") if child.is_file()))
        elif path.exists():
            result.append(path)
    return result


def write_run_metadata(
    output_dir: str | Path,
    command: str | list[str] | None = None,
    config_hash: str = "",
    input_hashes: dict[str, str] | None = None,
    timestamp: str | None = None,
    filename: str = "run_metadata.json",
) -> Path:
    """Write reproducibility metadata for a pipeline step."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if command is None:
        command_text = " ".join(shlex.quote(part) for part in sys.argv)
    elif isinstance(command, list):
        command_text = " ".join(shlex.quote(str(part)) for part in command)
    else:
        command_text = str(command)
    metadata = {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "command": command_text,
        "config_hash": config_hash,
        "input_hashes": input_hashes or {},
    }
    path = output_path / filename
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def should_skip_step(
    outputs: Iterable[str | Path],
    inputs: Iterable[str | Path],
    force: bool = False,
) -> bool:
    """Return True if all outputs exist and are newer than all existing inputs."""

    if force:
        return False
    output_paths = [Path(output) for output in outputs]
    if not output_paths or not all(file_exists_and_nonempty(path) for path in output_paths):
        return False
    input_paths = _flatten_existing(inputs)
    if not input_paths:
        return False
    newest_input = max(path.stat().st_mtime for path in input_paths)
    oldest_output = min(path.stat().st_mtime for path in output_paths)
    return oldest_output >= newest_input


def hash_inputs(paths: Iterable[str | Path]) -> dict[str, str]:
    """Return SHA256 hashes for existing input files."""

    return {str(path): compute_file_hash(path) for path in _flatten_existing(paths)}

