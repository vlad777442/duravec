#!/usr/bin/env bash
# Build a disk-resident DiskANN (Vamana) index.
#
# Usage:
#   build_diskann_index.sh <build_disk_index_bin> <data_type> <dist_fn> <data_path> \
#       <index_path_prefix> <R> <L> <B> <M> [QD] [num_threads]
#
# QD (quantized dimension / PQ bytes per vector on disk) is optional; omit or pass 0
# to let DiskANN auto-derive it from B, matching the published big-ann-benchmarks
# baseline behavior for SIFT10M/Deep10M. Pass an explicit value (e.g. 32) to pin the
# on-disk PQ byte budget regardless of B, needed for MS MARCO to match the same
# per-vector byte budget as the other corpora.
set -euo pipefail

BUILD_BIN="$1"
DATA_TYPE="$2"
DIST_FN="$3"
DATA_PATH="$4"
INDEX_PREFIX="$5"
R="$6"
L="$7"
B="$8"
M="$9"
QD="${10:-0}"
NUM_THREADS="${11:-$(nproc)}"

mkdir -p "$(dirname "$INDEX_PREFIX")"

ARGS=(--data_type "$DATA_TYPE" --dist_fn "$DIST_FN" \
      --data_path "$DATA_PATH" --index_path_prefix "$INDEX_PREFIX" \
      -R "$R" -L "$L" -B "$B" -M "$M" -T "$NUM_THREADS")

if [[ "$QD" != "0" ]]; then
  ARGS+=(--QD "$QD")
fi

echo "Running: $BUILD_BIN ${ARGS[*]}"
time "$BUILD_BIN" "${ARGS[@]}"
