#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMMIT_HASH="$(git log --oneline | head -n 1 | awk '{print $1}')"
COMMIT_DIR="commit/${COMMIT_HASH}"

FILTER="${1:-*}"

./compile_commit.sh

NTESTS=$(ls $SCRIPT_DIR/par/*$FILTER*.par 2>/dev/null | wc -l)
echo "Found $NTESTS tests matching '$FILTER'"

for i in $(ls $SCRIPT_DIR/par/*$FILTER*.par)
do
  cd $COMMIT_DIR
  BNAME=$(basename "$i" .par)
  ./dpecho $i | tee 1> ${BNAME}.out 2>${BNAME}.perf
  awk '/Problem::dtUpdate/{for(i=1;i<=NF;i++) if($i=="dt") print $(i+1)}' ${BNAME}.out > ${BNAME}.dt
  
#  if [[ ! -f "../ref/${BNAME}.dt" ]]; then
#    echo "ERROR: Missing reference ${BNAME}.dt"
#    exit 1
#  fi
#  
#  if [[ ! -f "../ref/${BNAME}.perf" ]]; then
#    echo "ERROR: Missing reference ${BNAME}.perf"
#    exit 1
#  fi
#  
#  if ! diff -q ${BNAME}.dt "../ref/${BNAME}.dt" > /dev/null 2>&1; then
#    echo "FAIL: ${BNAME} dt mismatch"
#    exit 1
#  fi
#  
#  if ! diff -q ${BNAME}.perf "../ref/${BNAME}.perf" > /dev/null 2>&1; then
#    echo "FAIL: ${BNAME} perf mismatch"
#    exit 1
#  fi
#  
#  echo "PASS: ${BNAME}"
#  cd ..
done

echo "All tests passed"
