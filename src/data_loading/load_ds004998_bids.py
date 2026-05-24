"""BIDS inspection helpers for OpenNeuro ds004998.

This module inspects local ds004998 files only. It does not download data and
does not use HCP-derived information as an electrophysiological reference.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

_default_cache = Path.cwd() / ".cache" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(_default_cache))
os.environ.setdefault("MPLBACKEND", "Agg")
_default_cache.mkdir(parents=True, exist_ok=True)

try:
    import mne
except ImportError:  # pragma: no cover - exercised only in missing dependency envs.
    mne = None


RECORDING_COLUMNS = [
    "subject",
    "session",
    "medication",
    "condition",
    "task",
    "run",
    "file_path",
    "meg_available",
    "lfp_stn_available",
    "duration_sec",
    "sampling_rate_hz",
    "n_channels",
    "n_meg_channels",
    "n_lfp_stn_channels",
    "usable",
    "qc_status",
]

LOGICAL_RECORDING_COLUMNS = [
    "subject",
    "session",
    "task",
    "side",
    "acq",
    "medication",
    "run",
    "is_split",
    "split_index",
    "canonical_path",
    "split_paths",
    "physical_file_count",
    "logical_recording_id",
]

USABLE_PAIR_COLUMNS = [
    "subject",
    "task",
    "side",
    "task_family",
    "medoff_logical_recording_id",
    "medon_logical_recording_id",
    "medoff_run",
    "medon_run",
    "medoff_canonical_path",
    "medon_canonical_path",
    "medoff_split_count",
    "medon_split_count",
    "selection_reason",
    "pair_id",
]

INCOMPLETE_GROUP_COLUMNS = [
    "subject",
    "task",
    "side",
    "available_acq",
    "has_medoff",
    "has_medon",
    "reason",
]

EXCLUDED_SUBJECT_COLUMNS = [
    "subject",
    "reason",
    "available_tasks",
    "available_holdmove_tasks",
]

VALID_HOLDMOVE_TASKS = {"HoldL", "HoldR", "MoveL", "MoveR"}

CHANNEL_COLUMNS = [
    "subject",
    "session",
    "task",
    "run",
    "file_path",
    "channel_name",
    "channel_type",
    "is_meg",
    "is_lfp_stn",
    "lfp_label",
    "lfp_side",
    "lfp_contact_index",
    "is_bad",
    "sampling_rate_hz",
]

QC_COLUMNS = [
    "subject",
    "session",
    "task",
    "run",
    "file_path",
    "usable",
    "qc_status",
    "duration_sec",
    "sampling_rate_hz",
    "n_channels",
    "n_meg_channels",
    "n_lfp_stn_channels",
    "notes",
]


def empty_recording_manifest() -> pd.DataFrame:
    """Return an empty recording manifest with stable columns."""

    return pd.DataFrame(columns=RECORDING_COLUMNS)


def empty_channel_manifest() -> pd.DataFrame:
    """Return an empty channel manifest with stable columns."""

    return pd.DataFrame(columns=CHANNEL_COLUMNS)


def empty_quality_control() -> pd.DataFrame:
    """Return an empty QC table with stable columns."""

    return pd.DataFrame(columns=QC_COLUMNS)


def parse_bids_entities(path: str | Path) -> dict[str, str]:
    """Parse common BIDS entities from a file path."""

    path = Path(path)
    entities = {
        "subject": "",
        "session": "",
        "task": "",
        "run": "",
        "split": "",
        "acquisition": "",
        "medication": "",
        "condition": "",
    }
    for token in re.split(r"[_/]", str(path)):
        if token.startswith("sub-"):
            entities["subject"] = token
        elif token.startswith("ses-"):
            entities["session"] = token
        elif token.startswith("task-"):
            entities["task"] = token.removeprefix("task-")
        elif token.startswith("run-"):
            entities["run"] = token.removeprefix("run-")
        elif token.startswith("split-"):
            entities["split"] = token.removeprefix("split-")
        elif token.startswith("acq-"):
            entities["acquisition"] = token.removeprefix("acq-")
        elif token.startswith("med-"):
            entities["medication"] = normalize_medication(token.removeprefix("med-"))
    entities["medication"] = (
        entities["medication"]
        or normalize_medication(entities["acquisition"])
        or infer_medication(path)
    )
    if entities["medication"] == "unknown":
        entities["medication"] = infer_medication(path)
    entities["condition"] = infer_condition(path, entities["task"])
    return entities


def derive_task_side(task: str | Any) -> str:
    """Return L/R/none side from Hold/Move task suffix."""

    text = str(task or "")
    if text.endswith("L"):
        return "L"
    if text.endswith("R"):
        return "R"
    if text.lower().startswith("rest"):
        return "none"
    return "unknown"


def derive_task_family(task: str | Any) -> str:
    """Return Hold/Move/Rest task family from task name."""

    text = str(task or "")
    lower = text.lower()
    if lower.startswith("hold"):
        return "Hold"
    if lower.startswith("move"):
        return "Move"
    if lower.startswith("rest"):
        return "Rest"
    return text or "unknown"


def _discover_physical_meg_fif_paths(data_root: str | Path) -> list[Path]:
    """Discover physical or symlinked MEG FIF paths by filename."""

    root = Path(data_root)
    if not root.exists():
        return []
    candidates: set[Path] = set()
    for pattern in ("sub-*/ses-*/meg/*_meg.fif", "sub-*/meg/*_meg.fif", "**/*_meg.fif"):
        candidates.update(root.glob(pattern))
    return sorted(
        path
        for path in candidates
        if path.name.endswith("_meg.fif")
        and ".git" not in path.parts
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def empty_logical_recording_manifest() -> pd.DataFrame:
    """Return an empty logical recording manifest."""

    return pd.DataFrame(columns=LOGICAL_RECORDING_COLUMNS)


def _run_sort_value(run: str | Any) -> tuple[int, str]:
    """Sort run labels numerically when possible."""

    text = str(run or "")
    try:
        return (int(text), text)
    except ValueError:
        return (9999, text)


def build_logical_recording_manifest(
    data_root: str | Path,
    require_existing: bool = True,
) -> pd.DataFrame:
    """Build one row per logical BIDS MEG recording, collapsing split FIF parts.

    Split FIF files are grouped by subject/session/task/acquisition/run. The
    canonical path is the first available split path when a recording is split,
    or the non-split path otherwise. By default only locally resolvable FIF
    files are counted, so broken git-annex symlinks do not create usable
    recordings.
    """

    paths = _discover_physical_meg_fif_paths(data_root)
    rows: list[dict[str, object]] = []
    for path in paths:
        if require_existing and not path.exists():
            continue
        entities = parse_bids_entities(path)
        acq = entities["acquisition"] or ("MedOff" if entities["medication"] == "off" else "MedOn" if entities["medication"] == "on" else "")
        if not entities["subject"] or not entities["task"]:
            continue
        rows.append(
            {
                "subject": entities["subject"],
                "session": entities["session"],
                "task": entities["task"],
                "side": derive_task_side(entities["task"]),
                "acq": acq,
                "medication": entities["medication"],
                "run": entities["run"],
                "split_index": entities["split"],
                "path": str(path),
            }
        )
    if not rows:
        return empty_logical_recording_manifest()

    physical = pd.DataFrame(rows)
    logical_rows: list[dict[str, object]] = []
    group_cols = ["subject", "session", "task", "side", "acq", "medication", "run"]
    for keys, group in physical.groupby(group_cols, dropna=False, sort=True):
        group = group.copy()
        group["_split_sort"] = group["split_index"].apply(lambda value: str(value or "00"))
        group = group.sort_values(["_split_sort", "path"], kind="mergesort")
        paths_sorted = group["path"].astype(str).tolist()
        split_indices = [str(value) for value in group["split_index"].fillna("").tolist() if str(value)]
        is_split = bool(split_indices or len(paths_sorted) > 1)
        canonical_path = paths_sorted[0]
        subject, session, task, side, acq, medication, run = keys
        logical_id = f"{subject}_{session}_task-{task}_acq-{acq}_run-{run}"
        logical_rows.append(
            {
                "subject": subject,
                "session": session,
                "task": task,
                "side": side,
                "acq": acq,
                "medication": medication,
                "run": str(run),
                "is_split": is_split,
                "split_index": split_indices[0] if split_indices else "",
                "canonical_path": canonical_path,
                "split_paths": "|".join(paths_sorted),
                "physical_file_count": int(len(paths_sorted)),
                "logical_recording_id": logical_id,
            }
        )
    return pd.DataFrame(logical_rows, columns=LOGICAL_RECORDING_COLUMNS).sort_values(
        ["subject", "task", "acq", "run", "canonical_path"],
        kind="mergesort",
    )


def _pair_selection_key(off_row: pd.Series, on_row: pd.Series) -> tuple[object, ...]:
    """Return deterministic sort key for MedOff/MedOn logical pair selection."""

    matched_run = str(off_row["run"]) == str(on_row["run"])
    both_non_split = not bool(off_row["is_split"]) and not bool(on_row["is_split"])
    both_run2_non_split = matched_run and str(off_row["run"]) == "2" and both_non_split
    return (
        0 if both_run2_non_split else 1,
        0 if matched_run else 1,
        0 if both_non_split else 1,
        _run_sort_value(off_row["run"]),
        _run_sort_value(on_row["run"]),
        int(off_row["physical_file_count"]) + int(on_row["physical_file_count"]),
        str(off_row["canonical_path"]),
        str(on_row["canonical_path"]),
    )


def select_canonical_medoff_medon_pair(group: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    """Select one deterministic MedOff/MedOn logical pair for a subject/task."""

    off = group[group["acq"].astype(str).eq("MedOff")].copy()
    on = group[group["acq"].astype(str).eq("MedOn")].copy()
    if off.empty or on.empty:
        raise ValueError("Cannot select canonical pair without both MedOff and MedOn.")
    candidates: list[tuple[tuple[object, ...], pd.Series, pd.Series]] = []
    for _, off_row in off.iterrows():
        for _, on_row in on.iterrows():
            candidates.append((_pair_selection_key(off_row, on_row), off_row, on_row))
    candidates.sort(key=lambda item: item[0])
    _, off_row, on_row = candidates[0]
    if str(off_row["run"]) == "2" and str(on_row["run"]) == "2" and not bool(off_row["is_split"]) and not bool(on_row["is_split"]):
        reason = "preferred_non_split_run_2_for_both_states"
    elif str(off_row["run"]) == str(on_row["run"]):
        reason = "preferred_matched_run_numbers"
    else:
        reason = "fallback_first_valid_medoff_medon_after_sort"
    return off_row, on_row, reason


def build_holdmove_usable_pair_inventory(
    logical_manifest: pd.DataFrame,
    valid_tasks: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build complete Hold/Move MedOff-MedOn pair inventory from logical recordings."""

    valid_tasks = valid_tasks or VALID_HOLDMOVE_TASKS
    if logical_manifest.empty:
        return (
            pd.DataFrame(columns=USABLE_PAIR_COLUMNS),
            pd.DataFrame(columns=INCOMPLETE_GROUP_COLUMNS),
            pd.DataFrame(columns=EXCLUDED_SUBJECT_COLUMNS),
        )

    manifest = logical_manifest.copy()
    manifest["task"] = manifest["task"].astype(str)
    manifest["acq"] = manifest["acq"].astype(str)
    holdmove = manifest[manifest["task"].isin(valid_tasks) & manifest["acq"].isin(["MedOff", "MedOn"])].copy()
    pair_rows: list[dict[str, object]] = []
    incomplete_rows: list[dict[str, object]] = []
    for (subject, task), group in holdmove.groupby(["subject", "task"], dropna=False, sort=True):
        acqs = sorted(group["acq"].dropna().astype(str).unique().tolist())
        has_off = "MedOff" in acqs
        has_on = "MedOn" in acqs
        side = derive_task_side(task)
        if has_off and has_on:
            off_row, on_row, reason = select_canonical_medoff_medon_pair(group)
            pair_rows.append(
                {
                    "subject": subject,
                    "task": task,
                    "side": side,
                    "task_family": derive_task_family(task),
                    "medoff_logical_recording_id": off_row["logical_recording_id"],
                    "medon_logical_recording_id": on_row["logical_recording_id"],
                    "medoff_run": str(off_row["run"]),
                    "medon_run": str(on_row["run"]),
                    "medoff_canonical_path": off_row["canonical_path"],
                    "medon_canonical_path": on_row["canonical_path"],
                    "medoff_split_count": int(off_row["physical_file_count"]),
                    "medon_split_count": int(on_row["physical_file_count"]),
                    "selection_reason": reason,
                    "pair_id": f"{subject}_{task}",
                }
            )
        else:
            missing = "missing_medoff" if not has_off else "missing_medon"
            incomplete_rows.append(
                {
                    "subject": subject,
                    "task": task,
                    "side": side,
                    "available_acq": ";".join(acqs),
                    "has_medoff": bool(has_off),
                    "has_medon": bool(has_on),
                    "reason": missing,
                }
            )

    excluded_rows: list[dict[str, object]] = []
    all_subjects = sorted(manifest["subject"].dropna().astype(str).unique().tolist())
    complete_subjects = {str(row["subject"]) for row in pair_rows}
    incomplete_subjects = {str(row["subject"]) for row in incomplete_rows}
    for subject in all_subjects:
        subject_tasks = sorted(manifest.loc[manifest["subject"].astype(str).eq(subject), "task"].dropna().astype(str).unique().tolist())
        holdmove_tasks = sorted(task for task in subject_tasks if task in valid_tasks)
        if not holdmove_tasks and any(str(task).lower().startswith("rest") for task in subject_tasks):
            excluded_rows.append(
                {
                    "subject": subject,
                    "reason": "rest_only_excluded_from_main_holdmove_v2a",
                    "available_tasks": ";".join(subject_tasks),
                    "available_holdmove_tasks": "",
                }
            )
        elif subject in incomplete_subjects and subject not in complete_subjects:
            excluded_rows.append(
                {
                    "subject": subject,
                    "reason": "incomplete_holdmove_medoff_medon_groups",
                    "available_tasks": ";".join(subject_tasks),
                    "available_holdmove_tasks": ";".join(holdmove_tasks),
                }
            )

    pairs = pd.DataFrame(pair_rows, columns=USABLE_PAIR_COLUMNS).sort_values(["subject", "task"], kind="mergesort")
    incomplete = pd.DataFrame(incomplete_rows, columns=INCOMPLETE_GROUP_COLUMNS).sort_values(["subject", "task"], kind="mergesort")
    excluded = pd.DataFrame(excluded_rows, columns=EXCLUDED_SUBJECT_COLUMNS).sort_values(["subject"], kind="mergesort")
    return pairs, incomplete, excluded


def normalize_medication(value: str | Any) -> str:
    """Normalize medication labels to off/on/unknown when possible."""

    text = str(value).strip().lower()
    if text in {"off", "medoff", "dopamineoff", "withoutmed", "without_med"}:
        return "off"
    if text in {"on", "medon", "dopamineon", "withmed", "with_med"}:
        return "on"
    if "off" in text and "on" not in text:
        return "off"
    if "on" in text and "off" not in text:
        return "on"
    return "unknown"


def _strip_bids_prefix(value: str, prefix: str) -> str:
    """Return an entity value without BIDS prefix."""

    return value.removeprefix(prefix)


def infer_montage_path(recording_path: str | Path) -> Path:
    """Infer the subject/session montage TSV path for a recording."""

    path = Path(recording_path)
    entities = parse_bids_entities(path)
    subject = entities["subject"]
    session = entities["session"]
    if not subject or not session:
        return Path("__missing_montage__.tsv")
    subject_dir = path
    while subject_dir.name != subject and subject_dir.parent != subject_dir:
        subject_dir = subject_dir.parent
    if subject_dir.name != subject:
        return Path("__missing_montage__.tsv")
    session_dir = subject_dir / session
    return session_dir / "montage" / f"{subject}_{session}_montage.tsv"


def _parse_lfp_label(label: str) -> tuple[str, int | str]:
    """Parse labels like LFP-right-0 into side/contact index."""

    match = re.search(r"lfp[-_](left|right)[-_](\d+)", str(label), flags=re.IGNORECASE)
    if not match:
        return "", ""
    return match.group(1).lower(), int(match.group(2))


def load_lfp_montage_mapping(recording_path: str | Path) -> dict[str, dict[str, str | int]]:
    """Load ds004998 EEG-to-LFP mapping from a subject montage TSV."""

    montage_path = infer_montage_path(recording_path)
    if not montage_path.exists():
        return {}
    montage = pd.read_csv(montage_path, sep="\t")
    mapping: dict[str, dict[str, str | int]] = {}
    for _, row in montage.iterrows():
        for old_col, new_col in (
            ("right_contacts_old", "right_contacts_new"),
            ("left_contacts_old", "left_contacts_new"),
        ):
            if old_col not in montage or new_col not in montage:
                continue
            old_name = str(row.get(old_col, "")).strip()
            new_label = str(row.get(new_col, "")).strip()
            if not old_name or old_name.lower() == "nan" or not new_label or new_label.lower() == "nan":
                continue
            side, contact_index = _parse_lfp_label(new_label)
            mapping[old_name] = {
                "lfp_label": new_label,
                "lfp_side": side,
                "lfp_contact_index": contact_index,
                "montage_path": str(montage_path),
            }
    return mapping


def infer_medication(path: str | Path) -> str:
    """Infer medication state from BIDS entities or folder names."""

    text = str(path).lower()
    patterns = {
        "off": [r"med-?off", r"medication-?off", r"\boff\b", r"ses-?off"],
        "on": [r"med-?on", r"medication-?on", r"\bon\b", r"ses-?on"],
    }
    for label, label_patterns in patterns.items():
        if any(re.search(pattern, text) for pattern in label_patterns):
            return label
    return "unknown"


def infer_condition(path: str | Path, task: str = "") -> str:
    """Infer rest/hold/move condition when present."""

    text = f"{path} {task}".lower()
    if "rest" in text:
        return "rest"
    if "hold" in text:
        return "hold"
    if "move" in text or "movement" in text:
        return "move"
    return task or "unknown"


def discover_recording_files(data_root: str | Path, max_recordings: int | None = None) -> list[Path]:
    """Discover likely ds004998 MEG FIF recordings."""

    manifest = build_logical_recording_manifest(data_root, require_existing=True)
    if manifest.empty:
        return []
    files = [Path(path) for path in manifest["canonical_path"].astype(str).tolist()]
    if max_recordings is not None:
        return files[:max_recordings]
    return files


def read_raw_header(file_path: str | Path):
    """Read a FIF header without preloading data."""

    if mne is None:
        raise ImportError("mne is required for ds004998 inspection.")
    return mne.io.read_raw_fif(file_path, preload=False, verbose="ERROR")


def channel_type_table(
    raw,
    entities: dict[str, str],
    file_path: Path,
    lfp_mapping: dict[str, dict[str, str | int]] | None = None,
) -> pd.DataFrame:
    """Build a channel manifest for one raw recording."""

    channel_types = raw.get_channel_types()
    bads = set(raw.info.get("bads", []))
    lfp_mapping = lfp_mapping or load_lfp_montage_mapping(file_path)
    rows = []
    for name, ch_type in zip(raw.ch_names, channel_types, strict=True):
        is_meg = ch_type in {"mag", "grad", "meg"}
        lower_name = name.lower()
        mapped_lfp = lfp_mapping.get(name, {})
        keyword_lfp = (
            ("stn" in lower_name or "lfp" in lower_name or "dbs" in lower_name)
            and ch_type not in {"stim", "eog", "ecg"}
        ) or ch_type in {"dbs"}
        is_lfp_stn = bool(mapped_lfp or keyword_lfp)
        rows.append(
            {
                "subject": entities["subject"],
                "session": entities["session"],
                "task": entities["task"],
                "run": entities["run"],
                "file_path": str(file_path),
                "channel_name": name,
                "channel_type": ch_type,
                "is_meg": bool(is_meg),
                "is_lfp_stn": bool(is_lfp_stn),
                "lfp_label": mapped_lfp.get("lfp_label", name if keyword_lfp else ""),
                "lfp_side": mapped_lfp.get("lfp_side", ""),
                "lfp_contact_index": mapped_lfp.get("lfp_contact_index", ""),
                "is_bad": name in bads,
                "sampling_rate_hz": float(raw.info["sfreq"]),
            }
        )
    return pd.DataFrame(rows, columns=CHANNEL_COLUMNS)


def inspect_recording(file_path: str | Path) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Inspect one recording and return recording, channel, and QC records."""

    path = Path(file_path)
    entities = parse_bids_entities(path)
    base = {
        "subject": entities["subject"],
        "session": entities["session"],
        "medication": entities["medication"],
        "condition": entities["condition"],
        "task": entities["task"],
        "run": entities["run"],
        "file_path": str(path),
    }
    try:
        raw = read_raw_header(path)
        lfp_mapping = load_lfp_montage_mapping(path)
        channels = channel_type_table(raw, entities, path, lfp_mapping=lfp_mapping)
        duration = float(raw.n_times / raw.info["sfreq"]) if raw.info["sfreq"] else 0.0
        n_meg = int(channels["is_meg"].sum())
        n_lfp = int(channels["is_lfp_stn"].sum())
        usable = bool(duration > 1.0 and n_lfp > 0 and n_meg > 0)
        qc_status = "ok" if usable else "missing_meg_or_lfp"
        recording = {
            **base,
            "meg_available": bool(n_meg > 0),
            "lfp_stn_available": bool(n_lfp > 0),
            "duration_sec": duration,
            "sampling_rate_hz": float(raw.info["sfreq"]),
            "n_channels": int(len(raw.ch_names)),
            "n_meg_channels": n_meg,
            "n_lfp_stn_channels": n_lfp,
            "usable": usable,
            "qc_status": qc_status,
        }
        qc = {
            "subject": entities["subject"],
            "session": entities["session"],
            "task": entities["task"],
            "run": entities["run"],
            "file_path": str(path),
            "usable": usable,
            "qc_status": qc_status,
            "duration_sec": duration,
            "sampling_rate_hz": float(raw.info["sfreq"]),
            "n_channels": int(len(raw.ch_names)),
            "n_meg_channels": n_meg,
            "n_lfp_stn_channels": n_lfp,
            "notes": (
                "Header inspection only. "
                f"Montage LFP channels: {len(lfp_mapping)}."
            ),
        }
        return recording, channels, qc
    except Exception as exc:  # noqa: BLE001 - inspection should continue across files.
        recording = {
            **base,
            "meg_available": False,
            "lfp_stn_available": False,
            "duration_sec": 0.0,
            "sampling_rate_hz": 0.0,
            "n_channels": 0,
            "n_meg_channels": 0,
            "n_lfp_stn_channels": 0,
            "usable": False,
            "qc_status": f"read_error:{type(exc).__name__}",
        }
        qc = {
            "subject": entities["subject"],
            "session": entities["session"],
            "task": entities["task"],
            "run": entities["run"],
            "file_path": str(path),
            "usable": False,
            "qc_status": f"read_error:{type(exc).__name__}",
            "duration_sec": 0.0,
            "sampling_rate_hz": 0.0,
            "n_channels": 0,
            "n_meg_channels": 0,
            "n_lfp_stn_channels": 0,
            "notes": str(exc),
        }
        return recording, empty_channel_manifest(), qc


def inspect_ds004998_bids(
    data_root: str | Path,
    max_recordings: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inspect a local ds004998 BIDS root."""

    files = discover_recording_files(data_root, max_recordings=max_recordings)
    if not files:
        qc = empty_quality_control()
        qc.loc[len(qc)] = {
            "subject": "",
            "session": "",
            "task": "",
            "run": "",
            "file_path": str(data_root),
            "usable": False,
            "qc_status": "no_recordings_found",
            "duration_sec": 0.0,
            "sampling_rate_hz": 0.0,
            "n_channels": 0,
            "n_meg_channels": 0,
            "n_lfp_stn_channels": 0,
            "notes": "No FIF recordings were found at the supplied data root.",
        }
        return empty_recording_manifest(), empty_channel_manifest(), qc

    recording_rows = []
    channel_tables = []
    qc_rows = []
    for file_path in files:
        recording, channels, qc = inspect_recording(file_path)
        recording_rows.append(recording)
        if not channels.empty:
            channel_tables.append(channels)
        qc_rows.append(qc)

    recordings = pd.DataFrame(recording_rows, columns=RECORDING_COLUMNS)
    channels = (
        pd.concat(channel_tables, ignore_index=True)
        if channel_tables
        else empty_channel_manifest()
    )
    qc = pd.DataFrame(qc_rows, columns=QC_COLUMNS)
    return recordings, channels, qc


def write_inspection_outputs(
    recordings: pd.DataFrame,
    channels: pd.DataFrame,
    qc: pd.DataFrame,
    output_dir: str | Path = "outputs/tables",
) -> dict[str, Path]:
    """Write ds004998 inspection tables."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "recording_manifest": output_dir / "ds004998_recording_manifest.csv",
        "channel_manifest": output_dir / "ds004998_channel_manifest.csv",
        "quality_control": output_dir / "ds004998_quality_control.csv",
    }
    recordings.to_csv(paths["recording_manifest"], index=False)
    channels.to_csv(paths["channel_manifest"], index=False)
    qc.to_csv(paths["quality_control"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local OpenNeuro ds004998 BIDS files.")
    parser.add_argument("--data-root", default="data/raw/ds004998")
    parser.add_argument("--output-dir", default="outputs/tables")
    parser.add_argument("--max-recordings", type=int, default=None)
    args = parser.parse_args()

    recordings, channels, qc = inspect_ds004998_bids(args.data_root, args.max_recordings)
    paths = write_inspection_outputs(recordings, channels, qc, args.output_dir)
    print(f"Wrote {len(recordings)} recording rows to {paths['recording_manifest']}.")
    print(f"Wrote {len(channels)} channel rows to {paths['channel_manifest']}.")
    print(f"Wrote {len(qc)} QC rows to {paths['quality_control']}.")


if __name__ == "__main__":
    main()
