from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(folder: Path, exclude: Path | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(folder.glob("*.csv")):
        if path.name in {"main_results.csv", "demo_results.csv"}:
            continue
        if exclude is not None and path.resolve() == exclude.resolve():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows.extend(dict(row) for row in reader)
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs")
    parser.add_argument("--output", default="outputs/main_results.csv")
    args = parser.parse_args()
    write_rows(Path(args.output), read_rows(Path(args.input), exclude=Path(args.output)))


if __name__ == "__main__":
    main()
