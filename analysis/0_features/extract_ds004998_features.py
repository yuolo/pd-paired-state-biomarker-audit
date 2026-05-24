"""Extract real ds004998 MEG/STN LFP proxy features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.load_ds004998_bids import write_inspection_outputs  # noqa: E402
from src.pathology_model.build_real_state_vectors import write_real_state_outputs  # noqa: E402
from src.preprocessing.meg_lfp_preprocessing import (  # noqa: E402
    build_extraction_failure_log,
    extract_ds004998_features,
)
from src.utils.cache_utils import (  # noqa: E402
    compute_file_hash,
    hash_inputs,
    should_skip_step,
    write_run_metadata,
)


def load_config(path: str | Path) -> dict:
    """Load YAML config."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ds004998 real-data proxy features.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output-dir", default="outputs/tables")
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--max-duration-sec", type=float, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--parallel-backend", choices=["process", "thread", "sequential"], default="process")
    parser.add_argument("--recording-batch-size", type=int, default=None)
    parser.add_argument("--worker-inner-n-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--cache-dir", default="outputs/cache")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    ds_cfg = config.get("real_data", {}).get("ds004998", {})
    max_recordings = args.max_recordings
    if max_recordings is None:
        max_recordings = ds_cfg.get("max_recordings_first_pass", 1)
    max_duration_sec = args.max_duration_sec
    if max_duration_sec is None:
        max_duration_sec = ds_cfg.get("max_duration_sec", 120.0)
    max_windows = args.max_windows
    if max_windows is None:
        max_windows = ds_cfg.get("max_windows_per_recording", 12)

    output_dir = Path(args.output_dir)
    expected_outputs = [
        output_dir / "ds004998_window_features.csv",
        output_dir / "ds004998_state_vectors.csv",
        output_dir / "ds004998_window_features_enhanced_coupling.csv",
        output_dir / "ds004998_state_vectors_enhanced_coupling.csv",
        output_dir / "cortico_stn_coupling_feature_inventory.csv",
        output_dir / "extraction_timing_by_recording.csv",
    ]
    cache_inputs = [
        args.config,
        args.data_root,
        "src/preprocessing/meg_lfp_preprocessing.py",
        "src/preprocessing/spectral_features.py",
        "src/preprocessing/coupling_features.py",
        "src/pathology_model/build_real_state_vectors.py",
    ]
    if (args.use_cache or args.skip_existing) and should_skip_step(expected_outputs, cache_inputs, force=args.force):
        print(f"SKIP cached output: {', '.join(str(path) for path in expected_outputs)}")
        write_run_metadata(
            args.cache_dir,
            config_hash=compute_file_hash(args.config),
            input_hashes=hash_inputs(cache_inputs[:1] + cache_inputs[2:]),
            filename="extract_ds004998_features_metadata.json",
        )
        return

    recordings, channels, qc, features, timing = extract_ds004998_features(
        args.data_root,
        max_recordings=max_recordings,
        max_duration_sec=max_duration_sec,
        max_windows=max_windows,
        n_jobs=args.n_jobs,
        parallel_backend=args.parallel_backend,
        recording_batch_size=args.recording_batch_size,
        return_timing=True,
    )
    inspect_paths = write_inspection_outputs(recordings, channels, qc, args.output_dir)
    state_paths = write_real_state_outputs(features, args.output_dir)
    failure_log = build_extraction_failure_log(recordings, qc, features)
    failure_log_path = Path(args.output_dir) / "extraction_failure_log.csv"
    failure_log.to_csv(failure_log_path, index=False)
    timing_path = Path(args.output_dir) / "extraction_timing_by_recording.csv"
    timing.to_csv(timing_path, index=False)
    write_run_metadata(
        args.cache_dir,
        config_hash=compute_file_hash(args.config),
        input_hashes=hash_inputs(cache_inputs[:1] + cache_inputs[2:]),
        filename="extract_ds004998_features_metadata.json",
    )
    print(f"Recording manifest: {inspect_paths['recording_manifest']} ({len(recordings)} rows)")
    print(f"Channel manifest: {inspect_paths['channel_manifest']} ({len(channels)} rows)")
    print(f"Quality control: {inspect_paths['quality_control']} ({len(qc)} rows)")
    print(f"Window features: {state_paths['window_features']} ({len(features)} rows)")
    print(f"State vectors: {state_paths['state_vectors']}")
    print(f"Extraction failure log: {failure_log_path} ({len(failure_log)} rows)")
    print(f"Extraction timing: {timing_path} ({len(timing)} rows)")


if __name__ == "__main__":
    main()
