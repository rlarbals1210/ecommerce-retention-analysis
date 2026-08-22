"""DuckDB SQL로 월별 코호트 잔존율을 재현한다."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser(description="REES46 월별 코호트 잔존율 재현")
    parser.add_argument("--glob", required=True, help="월별 CSV glob. 예: /data/2019-*.csv")
    parser.add_argument("--out", default="data/processed/cohort_retention.csv")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sql = (project_root / "sql" / "cohort_retention.sql").read_text(encoding="utf-8")
    event_files = [
        path for path in sorted(glob.glob(args.glob))
        if not Path(path).name.startswith("._")
    ]
    if not event_files:
        raise FileNotFoundError(f"No CSV files matched: {args.glob}")
    result = duckdb.connect().execute(sql, {"event_files": event_files}).df()

    output = project_root / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"OK {len(result):,} rows -> {output}")


if __name__ == "__main__":
    main()
