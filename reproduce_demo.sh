#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
for method in baseline stacr turbofast; do
  python -m stacr train --config synthetic_stacr --device cpu --set "method=$method" --output "outputs/demo/$method.csv" "$@"
done
