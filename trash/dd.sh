#!/bin/bash
# compile_commit.sh - Compile DPEcho binary in a test/tmp directory
# Usage: ./compile_commit.sh

echo "=== Compiling DPEcho in test/tmp directory ==="
mkdir -p "tmp"
cd "tmp"

# Configure with CMake using the correct path (3 levels up from tests/tmp to DPEcho)
echo "=== Configuring CMake ==="
cmake \
  -DCMAKE_CXX_COMPILER=mpiicpx \
  -DSYCL_DEVICE=DEF \
  -DCMAKE_BUILD_TYPE=Release  \
  ../..

# Build the binary
echo "=== Building DPEcho ==="
make -j "$(nproc)"

echo "=== Build complete ==="
echo "Binary location: tests/tmp/dpecho"
