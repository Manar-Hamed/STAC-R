from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from paper_data.build._plot_common import clean_axes, output_path, parser, read_csv


def build(data_dir: str, output_dir: str, suffix: str) -> None:
    df = read_csv(data_dir, "composition.csv")
    df = df[~df["method"].str.contains("marginal", case=False, na=False)].copy()
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    ax.bar(x, df["peak_gib"], color="#52796f")
    ax.set_ylabel("Peak memory (GiB)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=25, ha="right", fontsize=8)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(
        output_path(output_dir, "composition_peak_memory", suffix), bbox_inches="tight", dpi=300
    )


def main() -> None:
    args = parser("Build the composition memory figure.").parse_args()
    build(args.data_dir, args.output_dir, args.format)


if __name__ == "__main__":
    main()
