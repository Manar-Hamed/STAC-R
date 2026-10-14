from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from paper_data.build._plot_common import clean_axes, output_path, parser, read_csv


def build(data_dir: str, output_dir: str, suffix: str) -> None:
    df = read_csv(data_dir, "saved_surface_predictor.csv")
    df = df[pd.to_numeric(df["ssp_pct"], errors="coerce").notna()].copy()
    df["ssp_pct"] = df["ssp_pct"].astype(float)
    df = df.sort_values("ssp_pct")
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.barh(df["row"], df["ssp_pct"], color="#547aa5")
    ax.set_xlabel("SSP (%)")
    ax.set_ylabel("")
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(
        output_path(output_dir, "saved_surface_predictor", suffix), bbox_inches="tight", dpi=300
    )


def main() -> None:
    args = parser("Build the saved-surface predictor figure.").parse_args()
    build(args.data_dir, args.output_dir, args.format)


if __name__ == "__main__":
    main()
