#!/bin/bash
# WARNING: Do not forget to load the appropriate modules in your job script!
#          e.g. rocm, cuda, likwid, ear, ...
DT=0.75
GPUe=0.0
CPUe=0.0

START=$(date +%s.%N)

while true ; do
  ## GPU alternatives for deltaEnergy (DNRG)
#  GPUe=$(perf stat -e power/energy-gpu/   --log-fd 1 sleep $DT | awk -v dt=$DT '/Joules/     {print ($1+0)/dt }'&)
#  GPUe=$(nvidia-smi dmon -c 1 | awk '        {sum+=($2+0)} END{print sum}' &)
#  GPUe=$(rocm-smi        -P   | awk '/Power/ {sum+=($8+0)} END{print sum}' &)
#  GPUe=$(xpu-smi dump -m 1 -n 1 | awk '        {sum+=($3+0)} END{print sum}' &) # To be tested

  # CPU alternatives for deltaEnergy (DNRG)
#  CPUe=$(perf stat -e power/energy-cores/ --log-fd 1 sleep $DT | awk -v dt=$DT '/Joules/     {print ($1+0)/dt }'&)

# If all else fails, try the pkg energy
  CPUe=$(perf stat -a -e power/energy-pkg/ --log-fd 1 sleep $DT | awk '{print $1+0}' | grep '\.' | paste -sd " " | awk '{print $1/$2}')
#  CPUe=$(likwid-powermeter -M 0 -c 0,1 -s 0.1s 2>/dev/null | grep -A 2 PKG | awk '/Power/ {sum+=$3+0}END{print sum}' &)
#  CPUe=$(perf stat -e power/energy-pkg/   --log-fd 1 sleep $DT | awk -v dt=$DT '/Joules/     {print ($1+0)/dt }'&)
#  CPUe=$(likwid-perfctr -O -g ENERGY sleep $DT  2>/dev/null     | awk -F, '/Power \[W\] STAT/ {print ($2+0)    }'&)
#  CPUe=$(econtrol --power=$(hostname)                | awk    '                   {print ($5+0)    }' & ) ;   sleep $DT ;

  wait
  END=$(date +%s.%N)
  TIME=$(echo $END $START | awk '{print $1-$2}' )
  DNRG=$(echo $TIME $GPUe $CPUe | awk '{print $1*($2+$3)}')

  echo $DNRG
#  echo $CPUe, $GPUe,  $TIME #debug
  START=$(date +%s.%N)
done
