from __future__ import annotations

import matplotlib.pyplot as plt

from paper_data.build._plot_common import clean_axes, output_path, parser, read_csv


def build(data_dir: str, output_dir: str, suffix: str) -> None:
    df = read_csv(data_dir, "rate_distortion.csv")
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    markers = ["o", "s", "^", "D"]
    for idx, (method, group) in enumerate(df.groupby("method")):
        group = group.sort_values("bytes_per_entry")
        ax.plot(
            group["bytes_per_entry"],
            group["distortion"],
            marker=markers[idx % len(markers)],
            linewidth=1.5,
            label=method,
        )
    ax.set_xlabel("Stored bytes per activation entry")
    ax.set_ylabel("Distortion proxy")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path(output_dir, "rate_distortion", suffix), bbox_inches="tight", dpi=300)


def main() -> None:
    args = parser("Build the empirical rate-distortion frontier.").parse_args()
    build(args.data_dir, args.output_dir, args.format)


if __name__ == "__main__":
    main()
