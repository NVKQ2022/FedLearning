#!/usr/bin/env bash
# ==============================================================================
# Helper script to trigger thesis_report/build.sh from repository root
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT_DIR/thesis_report/build.sh" "$@"
