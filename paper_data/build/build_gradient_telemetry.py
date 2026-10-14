from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from paper_data.build._plot_common import clean_axes, output_path, parser, read_csv


def build(data_dir: str, output_dir: str, suffix: str) -> None:
    df = read_csv(data_dir, "gradient_telemetry.csv")
    labels = [f"{row.family} {row.row}" for row in df.itertuples()]
    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(x - width / 2, df["mean_cosine"], width, label="Mean", color="#255f85")
    ax.bar(x + width / 2, df["worst_cosine"], width, label="Worst", color="#c46a4a")
    ax.set_ylim(0.6, 1.01)
    ax.set_ylabel("Gradient cosine")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.legend(frameon=False, fontsize=8)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path(output_dir, "gradient_telemetry", suffix), bbox_inches="tight", dpi=300)


def main() -> None:
    args = parser("Build the gradient-cosine telemetry figure.").parse_args()
    build(args.data_dir, args.output_dir, args.format)


if __name__ == "__main__":
    main()
