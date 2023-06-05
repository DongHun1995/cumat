#!/bin/bash

: ${NODES:=1}
  salloc -N $NODES --partition shpc --exclusive --gres=gpu:4   \
  mpirun --bind-to none -mca btl ^openib -npernode 1         \
  numactl --physcpubind 0-31                                 \
  ./main $@

#/usr/local/cuda/bin/nsys profile --gpu-metrics-device all --cudabacktrace=all --trace=mpi,cuda,nvtx --mpi-impl=openmpi ./main $@
  








#: ${NODES:=4}

#salloc -N $NODES --partition shpc --exclusive --gres=gpu:4   \
#  mpirun --bind-to none -mca btl ^openib -npernode 1         \
#  numactl --physcpubind 0-31                                 \
#  ./main $@
