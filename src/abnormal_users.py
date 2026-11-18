"""First stop of the abnormal-user sensitivity analysis, using pre-reference events."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import subprocess
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
from src.recency_reference import file_identity, sha256_file, write_json_new


BASE = "ce81d18"
EXPECTED_USERS = 13_347_976
EXPECTED_GAPS = 36_655_640
CUTOFF = pd.Timestamp("2020-03-27T00:00:00Z")
MONTHS = ("2019-10", "2019-11", "2019-12", "2020-01", "2020-02", "2020-03")
RUN_DIR = config.PROC_DIR / "runs" / "abnormal_users_ref20200326"
SUMMARY = config.REPORTS_DIR / "sensitivity" / "abnormal_users" / "first_stop_ref20200326.json"
CHUNK_ROWS = 500_000
PARTITIONS = 16
GIB = 1024**3
SPOOL_LIMIT = 2 * GIB
SPILL_LIMIT_GIB = 4
DISK_RESERVE = 6 * GIB
MIN_INITIAL_FREE = 14 * GIB
MINUTE_SCHEMA = pa.schema([("user_id", pa.int64()), ("minute", pa.int32()), ("rows", pa.int32())])
PROBABILITIES = {"p50": .50, "p90": .90, "p99": .99, "p99.9": .999}


class ScanRejected(ValueError):
    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


def duck_connection(spill):
    spill.mkdir()
    db = duckdb.connect(config={
        "memory_limit": "2GiB", "threads": "2", "preserve_insertion_order": "false",
        "temp_directory": str(spill), "max_temp_directory_size": f"{SPILL_LIMIT_GIB}GiB",
        "TimeZone": "UTC",
    })
    return db


def timestamps(strings):
    parsed = pd.to_datetime(strings, format="%Y-%m-%d %H:%M:%S UTC", utc=True, errors="coerce")
    bad = parsed.isna()
    if bad.any():
        parsed.loc[bad] = pd.to_datetime(strings.loc[bad], format="mixed", utc=True, errors="coerce")
    return parsed


def append_partials(frame, month, scratch, writers, stack):
    minute = (frame.time.dt.as_unit("s").astype("int64") - int(month.timestamp())) // 60
    counts = frame.groupby([frame.user_id, minute.rename("minute")], sort=False).size().reset_index(name="rows")
    uid = counts.user_id.to_numpy()
    buckets = uid % PARTITIONS
    order = np.argsort(buckets, kind="stable")
    ordered_buckets = buckets[order]
    starts = np.searchsorted(ordered_buckets, np.arange(PARTITIONS + 1))
    for bucket, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        if start == end:
            continue
        part = counts.iloc[order[start:end]]
        if bucket not in writers:
            writers[bucket] = stack.enter_context(pq.ParquetWriter(
                scratch / f"part-{bucket:02}.parquet", MINUTE_SCHEMA, compression="snappy"))
        writers[bucket].write_table(pa.Table.from_pandas(part, schema=MINUTE_SCHEMA, preserve_index=False))


def scan_month(path, month_label, scratch, db, accept, *, chunksize=CHUNK_ROWS, progress=False):
    """Count each raw row once, validate the whole file, then reduce disjoint user partitions."""
    scratch.mkdir()
    start = pd.Timestamp(month_label + "-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    initial = file_identity(path)
    counts = {**initial, "utc_month": month_label, "raw_rows": 0, "parse_errors": 0,
              "outside_month_rows": 0, "cutoff_excluded_rows": 0, "included_rows": 0}
    minimum = maximum = None
    clock = time.monotonic()
    last_notification = clock
    try:
        with ExitStack() as stack:
            writers = {}
            for chunk in pd.read_csv(path, usecols=["user_id", "event_time"],
                                     dtype={"user_id": "int64", "event_time": "string"}, chunksize=chunksize):
                observed = timestamps(chunk.event_time)
                valid = observed.notna()
                counts["raw_rows"] += len(chunk)
                counts["parse_errors"] += int((~valid).sum())
                counts["cutoff_excluded_rows"] += int(observed.ge(CUTOFF).sum())
                outside = observed.lt(start) | observed.ge(end)
                counts["outside_month_rows"] += int(outside.sum())
                if valid.any():
                    low, high = observed.min(), observed.max()
                    minimum = low if minimum is None else min(minimum, low)
                    maximum = high if maximum is None else max(maximum, high)
                included = observed.lt(CUTOFF)
                counts["included_rows"] += int(included.sum())
                # The rest of an invalid file still contributes to diagnostics.
                if not (counts["parse_errors"] or counts["outside_month_rows"]) and included.any():
                    frame = pd.DataFrame({"user_id": chunk.loc[included, "user_id"], "time": observed.loc[included]})
                    append_partials(frame, start, scratch, writers, stack)
                spool_bytes = sum(p.stat().st_size for p in scratch.glob("*.parquet"))
                if spool_bytes > SPOOL_LIMIT or shutil.disk_usage(scratch).free < DISK_RESERVE:
                    raise ScanRejected("Temporary storage limit reached", counts)
                if progress and time.monotonic() - last_notification >= 25:
                    print(f"{path.name}: scanned {counts['raw_rows']:,} rows", flush=True)
                    last_notification = time.monotonic()

        counts.update({
            "min_time_utc": minimum.isoformat() if minimum is not None else None,
            "max_time_utc": maximum.isoformat() if maximum is not None else None,
            "metadata_unchanged": file_identity(path) == initial,
            "spool_bytes": sum(p.stat().st_size for p in scratch.glob("*.parquet")),
        })
        if (minimum is None or counts["parse_errors"] or counts["outside_month_rows"]
                or not counts["metadata_unchanged"]):
            raise ScanRejected("File rejected by full-file UTC month/parse/identity checks", counts)
        if counts["raw_rows"] != counts["included_rows"] + counts["cutoff_excluded_rows"]:
            raise ScanRejected("Raw rows failed cutoff reconciliation", counts)
        if counts["spool_bytes"] > SPOOL_LIMIT:
            raise ScanRejected("Temporary partial-count limit reached", counts)
        reduced_events = 0
        for part in sorted(scratch.glob("*.parquet")):
            if shutil.disk_usage(scratch).free < DISK_RESERVE + SPILL_LIMIT_GIB * GIB:
                raise ScanRejected("Insufficient free space before reduction", counts)
            result = db.execute("""
                SELECT user_id, CAST(SUM(n) AS BIGINT) AS total, MAX(n) AS peak
                FROM (
                    SELECT user_id, minute, CAST(SUM(rows) AS BIGINT) AS n
                    FROM read_parquet(?) GROUP BY user_id, minute
                ) GROUP BY user_id
                """, [str(part)]).fetchnumpy()
            reduced_events += int(result["total"].sum())
            accept(result["user_id"], result["total"], result["peak"])
        if reduced_events != counts["included_rows"]:
            raise ScanRejected("Minute counts failed event-total reconciliation", counts)
        counts["elapsed_seconds"] = round(time.monotonic() - clock, 2)
        return counts
    finally:
        shutil.rmtree(scratch)


def aligned_indices(cached_ids, ids):
    positions = np.searchsorted(cached_ids, ids)
    if np.any(positions >= len(cached_ids)) or np.any(cached_ids[positions] != ids):
        raise ValueError("Scan contains extra users absent from the activity-day cache")
    return positions


def validate_population(days, events):
    if np.any(days <= 0):
        raise ValueError("Cache contains a zero or invalid activity-day denominator")
    if np.any(events <= 0):
        raise ValueError("Activity-day cache contains users absent from the prescan")
    if np.any(events < days):
        raise ValueError("Prescan event totals cannot account for cached activity-day denominators")


def distribution(values):
    q = np.quantile(values, list(PROBABILITIES.values()), method="linear")
    return {**dict(zip(PROBABILITIES, map(float, q))), "max": float(np.max(values))}


def score_scenarios(ratios, peaks, events):
    a, b = distribution(ratios), distribution(peaks)
    thresholds = {"S1": a["p99"], "S2": a["p99.9"], "S3": b["p99.9"]}
    masks = {"S1": ratios > thresholds["S1"], "S2": ratios > thresholds["S2"],
             "S3": peaks > thresholds["S3"]}
    masks["S4"] = masks["S2"] | masks["S3"]
    ties = {"S1": ratios == thresholds["S1"], "S2": ratios == thresholds["S2"],
            "S3": peaks == thresholds["S3"]}
    rows = int(events.sum())
    counts = {}
    for key, mask in masks.items():
        users, event_rows = int(mask.sum()), int(events[mask].sum())
        counts[key] = {"users": users, "user_share": users / len(events),
                       "event_rows": event_rows, "event_share": event_rows / rows}
        if key in thresholds:
            counts[key].update(threshold=thresholds[key], ties=int(ties[key].sum()))
        else:
            counts[key].update(
                threshold_components={"S2": thresholds["S2"], "S3": thresholds["S3"]},
                ties_components={"S2": int(ties["S2"].sum()), "S3": int(ties["S3"].sum())},
                ties_either=int((ties["S2"] | ties["S3"]).sum()),
            )
    return {"A": a, "B": b, "scenarios": counts, "denominator_event_rows": rows,
            "overlap": {"S2_and_S3": int((masks["S2"] & masks["S3"]).sum()),
                        "S1_and_S3": int((masks["S1"] & masks["S3"]).sum())}}, masks


def activity_days_profile(days):
    if not len(days):
        return {"users": 0, "distribution": None, "counts_by_activity_days": {}}
    histogram = np.bincount(days)
    return {"users": len(days), "distribution": distribution(days), "min": int(days.min()),
            "counts_by_activity_days": {str(i): int(n) for i, n in enumerate(histogram) if n}}


def main():
    started = time.monotonic()
    head = subprocess.check_output(
        ["git", "--no-optional-locks", "rev-parse", "--short", "HEAD"],
        cwd=config.PROJECT_ROOT, text=True).strip()
    if head != BASE or config.AB_FEATURE_END_EXCLUSIVE != "2020-03-27T00:00:00Z":
        raise ValueError("Unexpected public baseline or cutoff")
    paths = config.RAW_PATHS[:len(MONTHS)]
    if len(paths) != len(MONTHS) or not all(p.is_file() for p in paths):
        raise FileNotFoundError("All six SSD inputs must be present")
    if RUN_DIR.exists() or SUMMARY.exists():
        raise FileExistsError("Existing first-stop run or summary must be preserved")
    if shutil.disk_usage(config.PROC_DIR).free < MIN_INITIAL_FREE:
        raise ValueError("Less than 14 GiB free for bounded scratch and local outputs")

    cache_path = config.RUN_PROC_DIR / "user_activity_days.parquet"
    previous = json.loads((config.RECENCY_REBUILD_DIR / "scan_diagnostics.json").read_text())
    evidence = json.loads(config.RISK_BASIS_PATH.read_text())
    cached_expected = evidence["input_files"][cache_path.relative_to(config.PROJECT_ROOT).as_posix()]["sha256"]
    if sha256_file(cache_path) != cached_expected:
        raise ValueError("Activity-day cache differs from the verified unique user-day input")

    RUN_DIR.mkdir(parents=True)
    scratch = RUN_DIR / "scratch"
    scratch.mkdir()
    diagnostics = {"status": "scanning", "files": []}
    write_progress = lambda: (RUN_DIR / "progress.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    db = duck_connection(scratch / "duckdb")
    try:
        source = db.execute("""
            SELECT user_id, CAST(COUNT(*) AS INTEGER) AS active_days FROM read_parquet(?)
            WHERE date < CAST(? AS TIMESTAMPTZ) GROUP BY user_id ORDER BY user_id
            """, [str(cache_path), CUTOFF.isoformat()]).fetchnumpy()
        users, days = source["user_id"], source["active_days"]
        # For a cache of unique UTC dates, each user contributes active_days minus one gaps.
        baseline = {"users": len(users), "completed_gaps": int(days.sum()) - len(users)}
        if baseline != {"users": EXPECTED_USERS, "completed_gaps": EXPECTED_GAPS}:
            raise ValueError(f"No-exclusion baseline differs: {baseline}")
        events = np.zeros(len(users), dtype=np.int64)
        peaks = np.zeros(len(users), dtype=np.int64)

        def accept(ids, totals, maxima):
            indices = aligned_indices(users, ids)
            events[indices] += totals
            peaks[indices] = np.maximum(peaks[indices], maxima)

        for month, path, prior in zip(MONTHS, paths, previous["files"], strict=True):
            print(f"Starting {path.name}", flush=True)
            entry = scan_month(path, month, scratch / month, db, accept, progress=True)
            if prior["name"] != entry["name"]:
                raise ValueError("Previous scan has different file names")
            entry["previous_raw_rows"] = prior["rows"]
            entry["raw_row_difference"] = entry["raw_rows"] - prior["rows"]
            diagnostics["files"].append(entry)
            write_progress()
            print(json.dumps(entry), flush=True)

        validate_population(days, events)
        if int(events.sum()) != sum(d["included_rows"] for d in diagnostics["files"]):
            raise ValueError("Total event count differs from sum of file diagnostics")
        if sha256_file(cache_path) != cached_expected:
            raise ValueError("The activity-day cache changed during the scan")

        ratios = events / days
        report, masks = score_scenarios(ratios, peaks, events)
        report.update({
            "status": "first_stop_pending_user_confirmation", "public_base": head,
            "period_utc_inclusive": ["2019-10-01", "2020-03-26"], "cutoff_exclusive": CUTOFF.isoformat(),
            "event_share_unit": "2019-10-01 ~ 2020-03-26 UTC pre-reference raw event rows",
            "quantile_method": "linear", "threshold_operator": ">",
            "scenario_definitions": {"S1": "A-p99", "S2": "A-p99.9", "S3": "B-p99.9", "S4": "S2 union S3"},
            "no_exclusion_baseline": baseline, "files": diagnostics["files"],
            "activity_days": {"eligible": activity_days_profile(days), "S3": activity_days_profile(days[masks["S3"]])},
            "scratch": {"directory": str(scratch.relative_to(config.PROJECT_ROOT)),
                        "monthly_partitions": PARTITIONS, "chunk_rows": CHUNK_ROWS,
                        "partial_spool_limit_bytes": SPOOL_LIMIT, "duckdb_spill_limit_gib": SPILL_LIMIT_GIB,
                        "disk_reserve_bytes": DISK_RESERVE,
                        "largest_month_partial_spool_bytes": max(d["spool_bytes"] for d in diagnostics["files"])},
            "limitations_for_later": [
                "Ground-truth bot labels are unavailable.",
                "Minute maxima cannot separate automation, duplicate recording or collection effects. Frequency can affect maxima.",
                "2,291,827 users first observed in the outcome window were outside the prescan eligibility scope.",
                "Identical full-event rows were not checked. Only user_id and event_time were read.",
                "No exclusion impact metrics were calculated at this stop.",
            ],
            "elapsed_seconds": round(time.monotonic() - started, 2),
        })
        pq.write_table(pa.table({"user_id": users, "pre_event_rows": events, "pre_active_days": days,
                                 "A": ratios, "B": peaks}), RUN_DIR / "user_scores.parquet", compression="zstd")
        union = masks["S1"] | masks["S3"]
        pq.write_table(pa.table({"user_id": users[union], **{k: v[union] for k, v in masks.items()}}),
                       RUN_DIR / "scenario_membership.parquet", compression="zstd")
        private_meta = {p.name: sha256_file(p) for p in
                        [RUN_DIR / "user_scores.parquet", RUN_DIR / "scenario_membership.parquet"]}
        write_json_new(RUN_DIR / "membership_manifest.json",
                       {"report": str(SUMMARY.relative_to(config.PROJECT_ROOT)), "files_sha256": private_meta,
                        "activity_day_cache_sha256": cached_expected, "module_sha256": sha256_file(Path(__file__)),
                        "state": "awaiting_user_confirmation"})
        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        write_json_new(SUMMARY, report)
        diagnostics["status"] = "first_stop_pending_user_confirmation"
        print(str(SUMMARY.relative_to(config.PROJECT_ROOT)), flush=True)
    except Exception as exc:
        diagnostics["status"] = "stopped_without_candidates"
        diagnostics["error"] = str(exc)
        if isinstance(exc, ScanRejected):
            diagnostics["rejected_file"] = exc.diagnostic
        raise
    finally:
        db.close()
        write_progress()
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
