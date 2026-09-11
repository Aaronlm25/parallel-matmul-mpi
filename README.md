# Parallel Matrix Multiplication with MPI

A small, self-contained project that parallelizes dense matrix
multiplication (`C = A * B`) with MPI in C, measures real speedup and
parallel efficiency against a sequential baseline, and verifies
correctness automatically.

It exists to back up, with runnable code and a documented benchmark,
the parallel/distributed-computing work referenced in my CV — not to
be a huge framework, just a clean, correct, well-tested base I can
keep extending.

## Overview

Given two `N x N` matrices `A` and `B`, the program computes
`C = A * B`. There is a sequential reference implementation
(`matmul_serial.c`) and a parallel implementation (`matmul_mpi.c`)
that partitions the work across MPI processes.

**Parallelization strategy — row-wise decomposition:**

1. Rank 0 generates `A` and `B`.
2. `B` is broadcast in full to every rank (`MPI_Bcast`) — every rank
   needs the entire `B` matrix, no matter which rows of `A` it owns.
3. The rows of `A` are split across ranks with `MPI_Scatterv`, using
   explicit counts/displacements so **`N` does not need to be evenly
   divisible by the number of processes**: the first `N % P` ranks get
   one extra row, everyone else gets `N / P`. This avoids the common
   beginner bug of assuming a perfectly divisible workload, and avoids
   over-allocating memory on any single rank.
4. Each rank computes its own block of rows of `C` locally, using a
   cache-friendlier `i-k-j` loop order instead of the naive `i-j-k`.
5. The partial results are collected back on rank 0 with
   `MPI_Gatherv`.

## Complexity analysis

Let `N` be the matrix dimension and `P` the number of processes.

| | Serial | Parallel (per rank) |
|---|---|---|
| Computation | `O(N^3)` | `O(N^3 / P)` |
| Communication | — | `O(N^2)` (dominated by the `B` broadcast) + `O(N^2 / P)` (scatter/gather of `A`/`C` rows) |

Ignoring lower-order terms, the expected running time of the parallel
version is roughly:

```
T(N, P) ≈ N^3 / P  +  c · N^2
```

where the first term is useful work split across processes and the
second is (approximately) fixed communication overhead that does
**not** shrink with `P`. That single fact explains two things you
should expect to see in any benchmark of this program:

- **Speedup grows sublinearly with `P`.** As `P` increases, the
  communication term stays roughly constant while the compute term
  keeps shrinking, so communication becomes a larger fraction of the
  total time — this is Amdahl's Law in practice, not a bug.
- **Efficiency improves with larger `N`.** For a fixed `P`, a bigger
  `N` means more computation per byte communicated, so parallel
  efficiency (`speedup / P`) goes up. Benchmarking only one matrix
  size gives a misleading picture — see below.

## Repository structure

```
mpi-matmul/
├── src/
│   ├── matmul_serial.c   # sequential baseline
│   └── matmul_mpi.c      # MPI parallel version
├── scripts/
│   ├── benchmark.sh      # runs serial + MPI across sizes/process counts
│   └── plot_results.py   # turns results/benchmark.csv into a speedup chart
├── results/              # benchmark.csv + speedup.png land here (gitignored except this note)
├── Makefile
└── README.md
```

## Requirements

- `gcc` (or any C11 compiler)
- An MPI implementation with `mpicc` / `mpirun` — developed and tested
  against **OpenMPI 4.1**; MPICH should also work but is untested.
- Python 3 with `matplotlib` — only needed for `scripts/plot_results.py`.

On Debian/Ubuntu:

```bash
sudo apt-get install libopenmpi-dev openmpi-bin
pip install matplotlib
```

## Build

```bash
make            # builds bin/matmul_serial and bin/matmul_mpi
make clean      # removes bin/
```

## Usage

Serial baseline:

```bash
./bin/matmul_serial <N> [seed]
```

Parallel version:

```bash
mpirun -n <P> ./bin/matmul_mpi <N> [seed] [--verify]
```

`--verify` makes rank 0 also compute the product serially and compare
it against the gathered parallel result (max absolute difference),
printing `PASS`/`FAIL`. It's meant for correctness checks during
development on small `N`, not for benchmark runs (it roughly doubles
rank 0's work).

Example:

```bash
$ mpirun -n 4 ./bin/matmul_mpi 512 --verify
RESULT,mpi,4,512,0.041233
N=512  processes=4  time=0.041233s
checksum=130998421.870000
verify: max_abs_diff=0.000e+00 -> PASS
```

## Correctness

Correctness was checked, not assumed:

- The serial and parallel programs use the **same seeded random
  generator**, so for `P=1` the outputs must match the serial run
  exactly — verified via the printed checksum.
- `--verify` was run for multiple `N` **not** divisible by the process
  count (e.g. `N=201` with `P=4`) to make sure the uneven row
  distribution (`MPI_Scatterv`/`MPI_Gatherv` with explicit
  counts/displacements) is correct and not just the easy, evenly
  divisible case.
- All these checks reported `max_abs_diff=0.000e+00 -> PASS`.

## Reproducing the benchmark

```bash
./scripts/benchmark.sh "256 512 1024" "1 2 4 8" 3
python3 scripts/plot_results.py results/benchmark.csv
```

This writes `results/benchmark.csv` and `results/speedup.png`, and
prints a speedup/efficiency table to the terminal.

**Important — run this on a machine with real physical cores.** If you
run it on a machine/VM with fewer cores than the largest process count
you test, MPI has to oversubscribe a single core, and the timings will
show flat or *negative* scaling (more processes, same or worse time)
purely from context-switching overhead — that's an artifact of the
hardware, not of the parallelization strategy. Match the process
counts you test to the physical cores available (e.g. don't test
`P=8` on a 4-core laptop).

## Results

*(Fill this in after running `scripts/benchmark.sh` on your own
machine — see note above. Report your CPU model and core count
alongside the numbers, since speedup is meaningless without that
context.)*

Hardware: `<CPU model, number of physical cores>`

| N | P | Time (s) | Speedup | Efficiency |
|---|---|----------|---------|------------|
| | 1 | | 1.00 | 100% |
| | 2 | | | |
| | 4 | | | |
| | 8 | | | |

`results/speedup.png` (generated by `plot_results.py`) plots this
same data against the ideal linear-speedup line.

## Possible improvements

Things I'd tackle next, roughly in order of how much I'd learn from
each:

- **Overlap communication and computation** with non-blocking calls
  (`MPI_Ibcast`, `MPI_Iscatter`) instead of the current blocking
  version — should reduce the `O(N^2)` overhead's real-world impact.
- **2D block decomposition** (e.g. Cannon's algorithm or SUMMA)
  instead of row-wise 1D decomposition — better communication-to-computation
  ratio at large `P`, and the standard next step in most parallel
  numerical linear algebra courses.
- **Hybrid MPI + OpenMP**: one MPI rank per node/socket, OpenMP threads
  for the local `matmul_rows` loop, to test process vs. thread scaling
  separately.
- **Compare against a tuned BLAS `dgemm`** to see how far the naive
  loop is from a properly cache-blocked/vectorized implementation.
- **CI**: a GitHub Actions workflow that builds the project and runs
  the `--verify` check on every push, so correctness regressions are
  caught automatically.
- Extend to non-square / rectangular matrices.
