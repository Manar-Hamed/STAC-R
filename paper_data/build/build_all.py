from __future__ import annotations

from paper_data.build._plot_common import parser
from paper_data.build.build_composition import build as build_composition
from paper_data.build.build_gradient_telemetry import build as build_gradient_telemetry
from paper_data.build.build_predicted_vs_actual import build as build_predicted_vs_actual
from paper_data.build.build_rate_distortion import build as build_rate_distortion
from paper_data.build.build_saved_surface_predictor import build as build_saved_surface_predictor


def main() -> None:
    args = parser("Build all paper-data figures.").parse_args()
    build_predicted_vs_actual(args.data_dir, args.output_dir, args.format)
    build_rate_distortion(args.data_dir, args.output_dir, args.format)
    build_saved_surface_predictor(args.data_dir, args.output_dir, args.format)
    build_gradient_telemetry(args.data_dir, args.output_dir, args.format)
    build_composition(args.data_dir, args.output_dir, args.format)


if __name__ == "__main__":
    main()
