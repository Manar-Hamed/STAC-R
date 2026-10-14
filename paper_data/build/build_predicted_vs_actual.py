from __future__ import annotations

import matplotlib.pyplot as plt

from paper_data.build._plot_common import clean_axes, output_path, parser, read_csv


def build(data_dir: str, output_dir: str, suffix: str) -> None:
    df = read_csv(data_dir, "predicted_vs_actual.csv")
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    ax.scatter(df["ssp_pct"], df["measured_saving_pct"], s=48, color="#1f6f8b")
    limit = max(df["ssp_pct"].max(), df["measured_saving_pct"].max()) * 1.12
    ax.plot([0, limit], [0, limit], color="#555555", linewidth=1.0, linestyle="--")
    for _, row in df.iterrows():
        ax.annotate(
            row["row"],
            (row["ssp_pct"], row["measured_saving_pct"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("SSP predicted saving ceiling (%)")
    ax.set_ylabel("Measured peak-memory saving (%)")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(
        output_path(output_dir, "predicted_vs_actual", suffix), bbox_inches="tight", dpi=300
    )


def main() -> None:
    args = parser("Build the SSP predicted-versus-measured figure.").parse_args()
    build(args.data_dir, args.output_dir, args.format)


if __name__ == "__main__":
    main()
