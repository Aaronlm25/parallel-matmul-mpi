#!/usr/bin/env python3
"""
plot_results.py
----------------
Reads the CSV produced by scripts/benchmark.sh (columns: n,kind,processes,
time_seconds), computes speedup and parallel efficiency relative to the
serial baseline for each matrix size N, prints a summary table, and saves
a speedup-vs-processes chart to results/speedup.png.

Usage:
    python3 scripts/plot_results.py results/benchmark.csv
"""
import sys
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["n"] = int(row["n"])
            row["processes"] = int(row["processes"])
            row["time_seconds"] = float(row["time_seconds"])
            rows.append(row)
    return rows


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <benchmark.csv>")
        sys.exit(1)

    rows = load(sys.argv[1])

    by_n = defaultdict(list)
    for r in rows:
        by_n[r["n"]].append(r)

    fig, ax = plt.subplots(figsize=(7, 5))

    print(f"{'N':>6} {'P':>4} {'time (s)':>10} {'speedup':>9} {'efficiency':>11}")
    print("-" * 46)

    for n in sorted(by_n):
        entries = by_n[n]
        serial = next((r for r in entries if r["kind"] == "serial"), None)
        if serial is None:
            print(f"Skipping N={n}: no serial baseline found in CSV")
            continue
        t_serial = serial["time_seconds"]

        mpi_entries = sorted((r for r in entries if r["kind"] == "mpi"),
                              key=lambda r: r["processes"])
        procs, speedups, effs = [], [], []
        for r in mpi_entries:
            speedup = t_serial / r["time_seconds"]
            efficiency = speedup / r["processes"]
            procs.append(r["processes"])
            speedups.append(speedup)
            effs.append(efficiency)
            print(f"{n:>6} {r['processes']:>4} {r['time_seconds']:>10.4f} "
                  f"{speedup:>9.2f} {efficiency:>10.1%}")

        if procs:
            ax.plot(procs, speedups, marker="o", label=f"N={n}")

    if by_n:
        max_p = max(r["processes"] for rows_ in by_n.values() for r in rows_ if r["kind"] == "mpi")
        ax.plot([1, max_p], [1, max_p], linestyle="--", color="gray", label="ideal (linear)")

    ax.set_xlabel("Number of MPI processes")
    ax.set_ylabel("Speedup (T_serial / T_parallel)")
    ax.set_title("Speedup vs. number of processes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = "results/speedup.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {out_path}")


if __name__ == "__main__":
    main()
