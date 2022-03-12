#!/bin/bash

default_kokkos_branch=3.3.00

top_level=`pwd`

subview_bench() {
  kokkos_branch=${1:-${default_kokkos_branch}}
  echo "============================================================="
  echo "Running ${FUNCNAME[0]} with kokkos branch ${kokkos_branch}"
  echo "============================================================="
  cd $top_level
  cd kokkos; git checkout ${kokkos_branch}; cd ..
  mkdir -p _build/serial_kokkos_${kokkos_branch}
  cd _build/serial_kokkos_${kokkos_branch}
  cmake -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ENABLE_SERIAL=ON ../..
  make -j 6
  ./subview 1000 | tee ../../subview_result_${kokkos_branch}
  cd $top_level
}

subview_bench 3.3.00
subview_bench 3.4.00
subview_bench 3.5.00
subview_bench master
subview_bench develop
