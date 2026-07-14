#!/bin/bash
git stash
git checkout
CC=${1:-$(git log --oneline | head -n 1 | awk '{print $1}')}

echo $CC
CN=commit_$CC

rm -rf $CN
mkdir -p $CN

cd $CN

cmake -DCMAKE_CXX_COMPILER=mpiicpx -DSYCL_DEVICE=CPU -DCMAKE_BUILD_TYPE=Release ..
make -j 11

./dpecho* ../../bisect/grmhd_periodic_detect.par | tee 1> out.out 2> perf.out

awk '/Problem::dtUpdate/{for(i=1;i<=NF;i++) if($i=="dt") print $(i+1)}' out.out > dts.txt

