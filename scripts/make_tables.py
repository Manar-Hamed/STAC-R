from __future__ import annotations

import argparse
import csv
from pathlib import Path
from importlib.resources import files


def escape_latex(value):
    escapes = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(escapes.get(char, char) for char in str(value))


def csv_to_latex(path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return ""
    cols = list(rows[0].keys())
    lines = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\toprule"]
    lines.append(" & ".join(escape_latex(c) for c in cols) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(escape_latex(row[c]) for c in cols) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(files("paper_data")))
    parser.add_argument("--output", default="outputs/tables")
    args = parser.parse_args()
    source = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for csv_path in sorted(source.glob("*.csv")):
        (out / f"{csv_path.stem}.tex").write_text(csv_to_latex(csv_path), encoding="utf-8")


if __name__ == "__main__":
    main()
