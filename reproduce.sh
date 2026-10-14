#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
args=()
if [[ -n "${STACR_STEPS:-}" ]]; then args+=(--steps "$STACR_STEPS"); fi
for method in baseline checkpoint full turbo; do
  python -m stacr reproduce run "gpt2-medium-quality-$method" "${args[@]}" "$@"
done
