#!/bin/bash
SEC=${1:-0.2}
MSEC=$(echo "1000 * $SEC" | bc -l)

perf stat -a -e power/energy-pkg/ --log-fd 1 -Sr 0  sleep $SEC | awk '/Jou/ {gsub(/,/, ".", $0); sum += $1+0; print sum}'
xpu-smi dump -m 8 --ims $MSEC --file /dev/stdout 2>/dev/null   | awk '{s += $3+0} (NR+2) % 4 == 0 {print s; s = 0}'
nvidia-smi -lms $MSEC --query-gpu=power.draw --format=csv,nounits,noheader | awk -v t=$SEC '{sum+=$1+0; print sum/t}'
