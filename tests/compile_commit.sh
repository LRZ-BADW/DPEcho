#!/bin/bash
# compile_commit.sh - Compile DPEcho binary for a specific commit
# Usage: ./compile_commit.sh [commit_hash]
#   commit_hash: Git commit hash (default: current HEAD)

# Get commit hash from argument or default to current HEAD
COMMIT_HASH="${1:-$(git log --oneline | head -n 1 | awk '{print $1}')}"
COMMIT_DIR="commit/${COMMIT_HASH}"

echo "=== Compiling DPEcho for commit: ${COMMIT_HASH} ==="
echo "=== Output directory: tests/${COMMIT_DIR} ==="

# Clean and create commit directory
cd "$(dirname "$0")"
rm -rf "$COMMIT_DIR"
mkdir -p "$COMMIT_DIR"
cd "$COMMIT_DIR"

# Configure with CMake
echo "=== Configuring CMake ==="
cmake \
  -DCMAKE_CXX_COMPILER=mpiicpx \
  -DSYCL_DEVICE=DEF \
  -DCMAKE_BUILD_TYPE=Release  \
  ../../..

# Build the binary
echo "=== Building DPEcho ==="
make -j "$(nproc)"

echo "=== Build complete ==="
echo "Binary location: tests/${COMMIT_DIR}/"
