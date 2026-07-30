#!/bin/bash
# test_commit.sh - Run DPEcho tests and compare dt files with reference

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

./compile_commit.sh

# Find all test parameter files
echo "=== Setting tests up ==="
FILTER="${1:-*}"
NTESTS=$(ls $SCRIPT_DIR/par/*$FILTER*.par 2>/dev/null | wc -l)
echo "Found $NTESTS tests matching '$FILTER'"

cd tmp
# Check if reference directory exists, if not skip tests without reference
REF_DIR="../commit/e369a57"
if [ ! -d "$REF_DIR" ]; then
    echo "Reference directory $REF_DIR does not exist"
    exit 1
fi

# Run tests and compare dt files
for i in $(ls $SCRIPT_DIR/par/*$FILTER*.par); do
    BNAME=$(basename "$i" .par)
    echo -n "- Running test: $BNAME   "
    # Compare with reference dt file
    REF_DT_FILE="$REF_DIR/${BNAME}.dt"
    if [ -f "$REF_DT_FILE" ]; then
      # Run the test
      ./dpecho "$i"  1> ${BNAME}.out 2>${BNAME}.perf
      # Extract dt values
      awk '/Problem::dtUpdate/{for(j=1;j<=NF;j++) if($j=="dt") print $(j+1)}' "${BNAME}.out" > "${BNAME}.dt"
      if diff -q "${BNAME}.dt" "$REF_DT_FILE" >/dev/null; then
        REF_PERF_FILE="$REF_DIR/${BNAME}.perf"
        if [ -f "$REF_PERF_FILE" ]; then
            CURR_PERF=$(awk '{sum+=$5} END{print sum}' "${BNAME}.perf")
            REF_PERF=$(awk '{sum+=$5} END{print sum}' "$REF_PERF_FILE")
            if [ -n "$CURR_PERF" ] && [ -n "$REF_PERF" ] && [ "$REF_PERF" != "0" ]; then
                SPEEDUP=$(awk -v c="$CURR_PERF" -v r="$REF_PERF" 'BEGIN {printf "%.1f", (c-r)/r*100}')
                echo "PASSED (${SPEEDUP}%)"
            else
                echo "PASSED"
            fi
        else
            echo "PASSED"
        fi
      else
        echo "FAILED - dt values differ:"
        sdiff "${BNAME}.dt" "$REF_DT_FILE"
        exit 1
       fi
    else
        echo "SKIPPED - no reference dt file"
    fi
done

echo "All tests completed!"

while true; do
    read -p "Do you wish to commit? " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
#        [Yy]* ) git commit -a || exit 1 ; break;;

echo "=== Committing test results ==="
msg=$(cd "$SCRIPT_DIR/.." && git log -1 --pretty=%B)
name=$(echo "$msg" | awk '{for(i=1;i<=4;i++) printf "%s",$i}' | tr -cd '[:alnum:]')
COMMIT_DIR="../commit/$name"
mkdir -p "$COMMIT_DIR"
# Move all .dt and .perf files to commit directory
mv *.dt *.perf "$COMMIT_DIR/" 2>/dev/null || true

# Clean up run folders and .out files from tmp dir
rm -rf 20??-* *.out 2>/dev/null || true
echo "Test results stored in $COMMIT_DIR"

# Commit the new test results
cd "$SCRIPT_DIR/.."
git add "tests/commit/$NEXT_COMMIT"

git add "$COMMIT_DIR/*"
git commit --amend --no-edit

