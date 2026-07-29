#!/bin/bash
# run_commit.sh - Run DPEcho tests and generate .dt and .perf files (no verification, no commit)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

./compile_commit.sh

# Find all test parameter files
echo "=== Setting tests up ==="
FILTER="${1:-*}"
NTESTS=$(ls $SCRIPT_DIR/par/*$FILTER*.par 2>/dev/null | wc -l)
echo "Found $NTESTS tests matching '$FILTER'"

cd tmp

# Run tests and generate dt/perf files
for i in $(ls $SCRIPT_DIR/par/*$FILTER*.par); do
    BNAME=$(basename "$i" .par)
    echo -n "- Running test: $BNAME   "
    # Run the test
    ./dpecho "$i"  1> ${BNAME}.out 2>${BNAME}.perf
    # Extract dt values
    awk '/Problem::dtUpdate/{for(j=1;j<=NF;j++) if($j=="dt") print $(j+1)}' "${BNAME}.out" > "${BNAME}.dt"
    echo "DONE"
done

echo "All tests completed!"

# Move all .dt and .perf files to commit directory based on current HEAD
HASH=$(cd "$SCRIPT_DIR/.." && git rev-parse --short HEAD)
COMMIT_DIR="../commit/$HASH"
mkdir -p "$COMMIT_DIR"
mv *.dt *.perf "$COMMIT_DIR/" 2>/dev/null || true

# Clean up run folders and .out files from tmp dir
rm -rf 20??-* *.out 2>/dev/null || true

echo "Test results stored in $COMMIT_DIR"