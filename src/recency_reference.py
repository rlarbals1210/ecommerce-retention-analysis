"""Restore the reference-day recency producer from 94c2556, notebook 05 cell 2.

The original normalize -> deduplicate -> groupby max calculation is retained.
Chunk-local maxima and balanced merges replace the six-month activity-day table;
UTC parsing and an explicit exclusive cutoff prevent outcome-window leakage.

Preparation only (no raw CSV access): python -m src.recency_reference prepare
Scanning requires a separate, explicit command and approval. A scan produces an
UNVALIDATED candidate, never replaces the existing cache or publishes results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config


def utc_reference(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Reference date must not be missing")
    return (stamp.tz_localize("UTC") if stamp.tzinfo is None
            else stamp.tz_convert("UTC")).normalize()


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return {"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_digest(ids):
    # Explicit little-endian serialization is independent of the host platform.
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def select_sample(labels, size=config.RECENCY_SAMPLE_SIZE, seed=config.RANDOM_SEED):
    """Select before scanning; sort IDs so label row order cannot affect selection."""
    if size < 1:
        raise ValueError("Sample size must be positive")
    ids = labels["user_id"]
    if ids.isna().any() or ids.duplicated().any() or not pd.api.types.is_integer_dtype(ids):
        raise ValueError("Label user IDs must be unique, nonmissing integers")
    if not labels["status"].isin(["retained", "churned", "censored"]).all():
        raise ValueError("Unexpected/missing label status")
    eligible = np.sort(labels.loc[labels["status"].isin(["retained", "churned"]),
                                  "user_id"].to_numpy(dtype="int64"))
    if size > len(eligible):
        raise ValueError("Sample exceeds the eligible population")
    selected = np.sort(np.random.default_rng(seed).choice(eligible, size=size, replace=False))
    return pd.DataFrame({"user_id": selected}), len(eligible)


def write_json_new(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def prepare_sample(labels_path, run_dir, size=config.RECENCY_SAMPLE_SIZE,
                   seed=config.RANDOM_SEED):
    """Write only a fixed sample and provenance, refusing to replace any run."""
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    before = file_identity(labels_path)
    labels_hash = sha256_file(labels_path)
    labels = pd.read_parquet(labels_path, columns=["user_id", "status"])
    sample, count = select_sample(labels, size, seed)
    if file_identity(labels_path) != before:
        raise ValueError("Label file changed during sample preparation")
    manifest = {
        "state": "sample_prepared_raw_scan_not_started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_date": config.AB_REFERENCE_DATE,
        "observation_start": config.AB_OBSERVATION_START,
        "timezone": "UTC", "sample_size": size, "seed": seed,
        "selection": "NumPy default_rng.choice without replacement from sorted eligible IDs",
        "numpy_version": np.__version__, "eligible_users": count,
        "sample_ids_sha256": sample_digest(sample["user_id"]),
        "label_source": {**before, "sha256": labels_hash},
        "source_cell": "94c2556:notebooks/05_ab_test_design.ipynb:cell[2]",
        "planned_input_files": config.RAW_FILES[:6],
        "candidate_file": "recency_at_reference.candidate.parquet",
        "sample_events_file": "sample_events.parquet",
        "diagnostics_file": "scan_diagnostics.json",
        "validation": {"real_data_b_c_d": "not_run", "cache_comparison": "not_run"},
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "sample_users.parquet").open("xb") as stream:
        sample.to_parquet(stream, index=False)
    write_json_new(run_dir / "preparation.json", manifest)
    return manifest


class _Maxima:
    """Balanced reduction: each level holds at most one partial user/date series."""

    def __init__(self):
        self.levels = []

    def add(self, series):
        if series.empty:
            return
        level = 0
        while level < len(self.levels) and self.levels[level] is not None:
            series = pd.concat([self.levels[level], series]).groupby(level=0, sort=False).max()
            self.levels[level] = None
            level += 1
        if level == len(self.levels):
            self.levels.append(series)
        else:
            self.levels[level] = series

    def finish(self):
        parts = [part for part in self.levels if part is not None]
        if not parts:
            return pd.Series([], dtype="datetime64[ns, UTC]",
                             index=pd.Index([], dtype="int64", name="user_id"))
        return pd.concat(parts).groupby(level=0, sort=True).max()


def scan_recency(paths, reference_date, sample_ids=(), chunksize=config.RECENCY_CHUNKSIZE,
                 progress=False):
    """Read two raw columns once. Return a candidate, pre-filter diagnostics and
    unfiltered sampled events; this function does not assert B validation passed.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    paths = [Path(path) for path in paths]
    if not paths or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Input paths must be nonempty and distinct")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files: " + ", ".join(missing))
    reference = utc_reference(reference_date)
    cutoff = reference + pd.Timedelta(days=1)
    accumulator = _Maxima()
    diagnostics, samples = [], []
    sample_ids = pd.Index(sample_ids, dtype="int64")
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    last_progress = started
    for path in paths:
        identity = file_identity(path)
        file_started = time.monotonic()
        if progress:
            print(json.dumps({"event": "file_started", "file": path.name,
                              "elapsed_seconds": round(file_started - started, 2)}), flush=True)
        diag = {**identity, "rows": 0, "valid_timestamp_rows": 0,
                "parse_errors": 0, "cutoff_excluded_rows": 0,
                "included_rows": 0, "min_time_utc": None, "max_time_utc": None}
        minimum, maximum = None, None
        row_offset = 0
        for chunk in pd.read_csv(path, usecols=["user_id", "event_time"],
                                 dtype={"user_id": config.DTYPES["user_id"],
                                        "event_time": "string"}, chunksize=chunksize):
            # Keep original strings BEFORE parsing/filtering for the independent oracle.
            sampled = chunk["user_id"].isin(sample_ids)
            selected = chunk.loc[sampled].copy()
            if not selected.empty:
                selected["source_file"] = path.name
                selected["source_row"] = np.flatnonzero(sampled) + row_offset + 1
                samples.append(selected)
            times = pd.to_datetime(chunk["event_time"], utc=True, format="mixed", errors="coerce")
            valid = times.notna()
            diag["rows"] += len(chunk)
            diag["parse_errors"] += int((~valid).sum())
            diag["valid_timestamp_rows"] += int(valid.sum())
            diag["cutoff_excluded_rows"] += int((valid & times.ge(cutoff)).sum())
            if valid.any():
                lo, hi = times[valid].min(), times[valid].max()
                minimum = lo if minimum is None else min(minimum, lo)
                maximum = hi if maximum is None else max(maximum, hi)
            keep = valid & times.lt(cutoff)
            diag["included_rows"] += int(keep.sum())
            daily = pd.DataFrame({"user_id": chunk.loc[keep, "user_id"],
                                  "date": times.loc[keep].dt.normalize()}).drop_duplicates()
            accumulator.add(daily.groupby("user_id", sort=False)["date"].max())
            row_offset += len(chunk)
            now = time.monotonic()
            if progress and now - last_progress >= 30:
                print(json.dumps({"event": "scan_progress", "file": path.name,
                                  "file_rows": row_offset,
                                  "elapsed_seconds": round(now - started, 2),
                                  "parse_errors_in_file": diag["parse_errors"]}), flush=True)
                last_progress = now
        diag["min_time_utc"] = minimum.isoformat() if minimum is not None else None
        diag["max_time_utc"] = maximum.isoformat() if maximum is not None else None
        diag["input_unchanged_during_scan"] = file_identity(path) == identity
        diag["elapsed_seconds"] = round(time.monotonic() - file_started, 3)
        diagnostics.append(diag)
        if progress:
            print(json.dumps({"event": "file_completed", **diag}), flush=True)
    last_date = accumulator.finish()
    recency = (reference - last_date).dt.days.rename("recency_at_ref").reset_index()
    events = (pd.concat(samples, ignore_index=True) if samples else
              pd.DataFrame(columns=["user_id", "event_time", "source_file", "source_row"]))
    audit = {
        "state": "candidate_only_not_validated",
        "reference_date": reference.date().isoformat(), "timezone": "UTC",
        "cutoff_exclusive_utc": cutoff.isoformat(), "chunksize": chunksize,
        "files": diagnostics,
        "parse_errors": sum(d["parse_errors"] for d in diagnostics),
        "cutoff_excluded_rows": sum(d["cutoff_excluded_rows"] for d in diagnostics),
        "input_changed": any(not d["input_unchanged_during_scan"] for d in diagnostics),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "pandas_version": pd.__version__, "numpy_version": np.__version__,
    }
    return recency, audit, events


def independent_sample_recency(events, sample_ids, reference_date):
    """Stdlib-only time/max oracle; does not reuse the production parsing/reduction.

    Accept the source's UTC timestamps and ISO offsets. Unsupported/missing values
    raise, rather than being silently dropped. All selected IDs must be explained.
    """
    reference = datetime.fromisoformat(str(reference_date)).date()
    wanted = set(map(int, sample_ids))
    last = {}
    for row in events.itertuples(index=False):
        if int(row.user_id) not in wanted:
            raise ValueError("Unexpected user in sampled events")
        value = str(row.event_time).replace(" UTC", "+00:00").replace("Z", "+00:00")
        stamp = datetime.fromisoformat(value)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        day = stamp.astimezone(timezone.utc).date()
        if day <= reference:
            uid = int(row.user_id)
            last[uid] = max(last.get(uid, day), day)
    if set(last) != wanted:
        raise ValueError("Sample contains users without a pre-reference event")
    return {uid: (reference - last[uid]).days for uid in sorted(wanted)}


def scan_prepared(run_dir, chunksize=config.RECENCY_CHUNKSIZE, progress=False):
    """Explicit candidate production only; later validation/adoption is separate."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "preparation.json").read_text())
    if manifest["reference_date"] != config.AB_REFERENCE_DATE:
        raise ValueError("Prepared sample reference date differs from config")
    outputs = [run_dir / manifest[key] for key in
               ("candidate_file", "sample_events_file", "diagnostics_file")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Candidate run outputs already exist; do not overwrite")
    sample = pd.read_parquet(run_dir / "sample_users.parquet")
    ids = sample["user_id"]
    if (len(ids) != manifest["sample_size"] or ids.isna().any() or ids.duplicated().any()
            or sample_digest(ids) != manifest["sample_ids_sha256"]):
        raise ValueError("Prepared sample does not match its manifest")
    candidate, audit, events = scan_recency(config.RAW_PATHS[:6], config.AB_REFERENCE_DATE,
                                           ids, chunksize, progress=progress)
    audit["sample_ids_sha256"] = manifest["sample_ids_sha256"]
    audit["producer_sha256"] = sha256_file(Path(__file__))
    audit["config_sha256"] = sha256_file(Path(config.__file__))
    for frame, output in zip([candidate, events], outputs[:2]):
        with output.open("xb") as stream:
            frame.to_parquet(stream, index=False)
    write_json_new(outputs[2], audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "scan"])
    parser.add_argument("--run-dir", type=Path, default=config.RECENCY_REBUILD_DIR)
    parser.add_argument("--chunksize", type=int, default=config.RECENCY_CHUNKSIZE)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare_sample(config.PROC_DIR / "churn_labels.parquet", args.run_dir)
    else:
        result = scan_prepared(args.run_dir, args.chunksize, progress=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
