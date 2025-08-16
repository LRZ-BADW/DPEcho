#!/bin/bash
SEC=${1:-0.2}
MSEC=$(echo "1000 * $SEC" | bc -l)

perf stat -a -e power/energy-pkg/ --log-fd 1 -Sr 0  sleep $SEC | awk '/Jou/ {gsub(/,/, ".", $0); sum += $1+0; print sum}'

xpu-smi dump -m 8 --ims $MSEC --file /dev/stdout 2>/dev/null   | awk '{s += $3+0} (NR+2) % 4 == 0 {print s; s = 0}'

nvidia-smi -lms $MSEC --query-gpu=power.draw --format=csv,nounits,noheader | awk -v t=$SEC '{sum+=$1+0; print sum/t}'

# AMD is not great in general as it doesn't have a daemon mode. 
# AMD w rocm-smi, a bit faster: latency is about 0.5 sec
Z=$(rocm-smi --showenergycounter --csv | awk -F"," '{sum+=$3+0}END{print sum/1E6}'); 
while true; do rocm-smi --showenergycounter --csv | awk -F"," -v z=$Z '{sum+=($3+0)}END{print sum/1E6-z+5000}' ; done

# AMD w amd-smi, a bit simpler but slower: about 0.7 sec
Z=$(amd-smi metric -E --csv | awk -F"," '{sum+=$2+0}END{print sum}'); 
while true; do amd-smi metric -E --csv | awk -F"," -v z=$Z '{sum+=($2+0)}END{print sum-z+4000}' ; done
