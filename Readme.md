Tests various strategies for subviews
=============================================

# Installation

Clone the git repository adding the `--recursive` option

# Test

One defines a simple kind a CSR storage. The data are stored in a
large 1d view (`entries`) or an hand written array (`shared_array`)
which uses a naive reference counting mechanism.

This test computes the sum of the values of each row using various approaches
- `Kokkos/Direct` does ne create a view but browse simply the row stored in a
- `Kokkos::subView` which use Kokkos' provided sub-view mechanisme
- `Kokkos::View` which builds a Kokkos' view from `entries` (recommended in Kokkos manual)
- `Kokkos|RawView` an hand written **unsafe** view (the pointer may be invalid if the `Kokkos::View` is destroyed before the view
- `SharedArrayView` an hand written view for the simple `SharedArray` (shares the reference counter)
- `SharedArray|RawView` the same **unsafe** view used with `SharedArray`

# Compilation

- One requires a C++-17 compiler.
- Binary **cannot** be built in sources
- Default build type is `Release`

# Use

One must provide an integer parameter which denotes the number of times each test is run. A value of 1000 seems quite reasonnable.
