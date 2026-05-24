"""Channel selection utilities for ds004998 MEG + STN LFP recordings."""

from __future__ import annotations

import re


def is_meg_channel(channel_type: str) -> bool:
    """Return True for MNE MEG channel types."""

    return channel_type in {"mag", "grad", "meg"}


def is_stn_lfp_channel(
    channel_name: str,
    channel_type: str,
    lfp_mapping: dict[str, dict] | None = None,
) -> bool:
    """Return True for likely STN/LFP/DBS channels."""

    if lfp_mapping and channel_name in lfp_mapping:
        return True
    lower = channel_name.lower()
    return (
        channel_type == "dbs"
        or ("stn" in lower or "lfp" in lower or "dbs" in lower)
        and channel_type not in {"stim", "eog", "ecg"}
    )


def meg_picks(raw) -> list[int]:
    """Return MEG sensor picks."""

    types = raw.get_channel_types()
    return [idx for idx, ch_type in enumerate(types) if is_meg_channel(ch_type)]


def stn_lfp_picks(raw, lfp_mapping: dict[str, dict] | None = None) -> list[int]:
    """Return likely STN/LFP channel picks."""

    types = raw.get_channel_types()
    return [
        idx
        for idx, (name, ch_type) in enumerate(zip(raw.ch_names, types, strict=True))
        if is_stn_lfp_channel(name, ch_type, lfp_mapping=lfp_mapping)
    ]


def stn_lfp_picks_by_side(raw, lfp_mapping: dict[str, dict] | None = None) -> dict[str, list[int]]:
    """Return likely STN/LFP picks split by montage side."""

    lfp_mapping = lfp_mapping or {}
    by_side = {"left": [], "right": [], "unknown": []}
    types = raw.get_channel_types()
    for idx, (name, ch_type) in enumerate(zip(raw.ch_names, types, strict=True)):
        if not is_stn_lfp_channel(name, ch_type, lfp_mapping=lfp_mapping):
            continue
        side = str(lfp_mapping.get(name, {}).get("lfp_side", "")).lower() or "unknown"
        by_side.setdefault(side, []).append(idx)
    return by_side


def motor_cortical_proxy_picks(raw, max_channels: int = 24) -> list[int]:
    """Return a transparent motor-cortical sensor proxy.

    Without subject-specific source localization, this selects MEG channels
    whose names suggest motor coverage when possible. If names are generic,
    it falls back to the first MEG channels and labels the result as a sensor
    proxy, not an anatomical source estimate.
    """

    all_meg = meg_picks(raw)
    name_patterns = re.compile(r"(motor|m1|sma|premotor|central|lh|rh)", re.IGNORECASE)
    named = [idx for idx in all_meg if name_patterns.search(raw.ch_names[idx])]
    selected = named or all_meg[:max_channels]
    return selected[:max_channels]
