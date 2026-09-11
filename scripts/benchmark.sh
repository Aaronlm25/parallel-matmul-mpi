#!/usr/bin/env bash
# benchmark.sh
# ------------
# Runs matmul_serial once per matrix size (as the baseline) and
# matmul_mpi with several process counts, taking the minimum time
# over a few repeats to reduce noise from OS/scheduler jitter.
#
# Usage:
#   ./scripts/benchmark.sh [sizes] [proc_counts] [repeats]
#
# Examples:
#   ./scripts/benchmark.sh                       # defaults
#   ./scripts/benchmark.sh "512 1024" "1 2 4 8" 5
#
# Output: results/benchmark.csv with columns: n,processes,time_seconds
#
# NOTE: run this on a machine with at least as many physical cores as
# the largest process count you test (e.g. 8), otherwise MPI has to
# oversubscribe a single core and the timings will NOT reflect real
# parallel speedup -- see README.md, section "Reproducing the benchmark".

set -euo pipefail

SIZES="${1:-256 512 1024}"
PROCS="${2:-1 2 4 8}"
REPEATS="${3:-3}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results
OUT="results/benchmark.csv"
echo "n,kind,processes,time_seconds" > "$OUT"

if [[ ! -x bin/matmul_serial || ! -x bin/matmul_mpi ]]; then
    echo "Binaries not found, building..."
    make
fi

min_of() {
    # Prints the minimum of the numbers passed as arguments.
    printf '%s\n' "$@" | sort -g | head -n1
}

for N in $SIZES; do
    echo "== N=$N =="

    # --- Serial baseline (processes = 1, no MPI runtime involved) ---
    times=()
    for ((i = 1; i <= REPEATS; i++)); do
        t=$(./bin/matmul_serial "$N" | grep '^RESULT' | cut -d',' -f5)
        times+=("$t")
    done
    best=$(min_of "${times[@]}")
    echo "serial: ${times[*]} -> best=${best}s"
    echo "$N,serial,1,$best" >> "$OUT"

    # --- MPI, scaling process count ---
    for P in $PROCS; do
        times=()
        for ((i = 1; i <= REPEATS; i++)); do
            t=$(mpirun --allow-run-as-root --oversubscribe -n "$P" ./bin/matmul_mpi "$N" \
                | grep '^RESULT' | cut -d',' -f5)
            times+=("$t")
        done
        best=$(min_of "${times[@]}")
        echo "mpi P=$P: ${times[*]} -> best=${best}s"
        echo "$N,mpi,$P,$best" >> "$OUT"
    done
done

echo
echo "Done. Raw results written to $OUT"
echo "Next: python3 scripts/plot_results.py $OUT"
