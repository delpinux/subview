#!/bin/bash

default_kokkos_branch=3.3.00

top_level=`pwd`

subview_bench() {
  kokkos_branch=${1:-${default_kokkos_branch}}
  nb_value_per_row=${2:-5}
  echo "============================================================="
  echo "Running ${FUNCNAME[0]} with kokkos branch ${kokkos_branch}"
  echo "============================================================="
  cd $top_level
  cd kokkos; git checkout ${kokkos_branch}; cd ..
  mkdir -p _build/serial_kokkos_${kokkos_branch}
  cd _build/serial_kokkos_${kokkos_branch}
  cmake -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ENABLE_SERIAL=ON ../..
  make -j 6
  ./subview 1000 ${nb_value_per_row} | tee ../../subview_result_${kokkos_branch}_${nb_value_per_row}
  cd $top_level
}

subview_bench 3.3.00
subview_bench 3.3.00 50

subview_bench 3.4.00
subview_bench 3.4.00 50

subview_bench 3.5.00
subview_bench 3.5.00 50

subview_bench master
subview_bench master 50

subview_bench develop
subview_bench develop 50
