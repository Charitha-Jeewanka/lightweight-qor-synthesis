#!/usr/bin/env python3
"""
QoR data generation script for "Lightweight Learning-Based QoR Prediction
for Logic Synthesis Optimization Sequences" (Section 5.2 of the proposal).

For each circuit in the EPFL benchmark suite (arithmetic + random/control,
excluding MtM), sample random fixed-length sequences from the ABC command
vocabulary, run them through ABC, and record area (AND-node count), delay
(logic level count), and runtime.

Also runs the two reference recipes (resyn2, resyn2rs) on every circuit.

Usage:
    python3 generate_data.py --abc ~/ABC_EPFL/abc/abc \
        --benchmarks ~/ABC_EPFL/benchmarks \
        --out data/qor_dataset.csv \
        --seq-len 10 \
        --n-seqs 500

Design notes:
- Writes to CSV incrementally (flush after every row) so an overnight run
  is crash-safe / resumable: on restart, already-completed (circuit,
  sequence) rows are skipped by checking a run-id already present in the
  output file.
- Each ABC invocation is a fresh subprocess (`abc -c "..."`), which keeps
  ABC's internal state clean per run and mirrors how the tool is normally
  used.
- Command vocabulary matches Section 5.2: balance, rewrite, rewrite -z,
  refactor, refactor -z, resub, resub -z, dc2.
"""

import argparse
import csv
import hashlib
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Command vocabulary (Section 5.2 candidate set)
# ---------------------------------------------------------------------------
VOCAB = [
    "balance",
    "rewrite",
    "rewrite -z",
    "refactor",
    "refactor -z",
    "resub",
    "resub -z",
    "dc2",
]

REFERENCE_RECIPES = {
    "resyn2": "balance; rewrite; rewrite -z; balance; rewrite -z; balance",
    "resyn2rs": (
        "balance; rewrite; refactor; balance; rewrite; rewrite -z; "
        "balance; refactor -z; rewrite -z; balance"
    ),
}

# regex to parse ABC's print_stats output, e.g.:
# adder : i/o = 256/129  lat = 0  and = 1020  lev = 255
STATS_RE = re.compile(
    r"i/o\s*=\s*(\d+)\s*/\s*(\d+)\s+lat\s*=\s*(\d+)\s+and\s*=\s*(\d+)\s+lev\s*=\s*(\d+)"
)


def discover_circuits(benchmarks_dir: Path):
    """Return list of (category, circuit_name, aig_path) for arithmetic and
    random_control subfolders, skipping MtM (not present as .aig by default,
    but skip explicitly if a directory named e.g. 'mtm' exists)."""
    circuits = []
    for category in ("arithmetic", "random_control"):
        cat_dir = benchmarks_dir / category
        if not cat_dir.is_dir():
            print(f"WARNING: expected folder not found: {cat_dir}", file=sys.stderr)
            continue
        for aig_file in sorted(cat_dir.glob("*.aig")):
            circuits.append((category, aig_file.stem, aig_file))
    return circuits


def run_id_for(circuit_name: str, seq_str: str) -> str:
    """Stable short hash identifying a (circuit, sequence) pair, used for
    resumability."""
    h = hashlib.sha1(f"{circuit_name}|{seq_str}".encode()).hexdigest()[:12]
    return h


def sample_sequence(seq_len: int, rng: random.Random):
    return [rng.choice(VOCAB) for _ in range(seq_len)]


def run_abc(abc_path: str, aig_path: Path, seq_cmds: str, timeout_s: int = 120):
    """Run ABC: read circuit, print_stats (initial), apply sequence,
    print_stats (final). Returns (result_dict_or_None, failure_reason_or_None).
    On success, failure_reason is None. On failure, result is None and
    failure_reason is a short string identifying what went wrong."""
    script = f"read {aig_path}; print_stats; {seq_cmds}; print_stats"
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [abc_path, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return None, f"timeout>{timeout_s}s"
    elapsed = time.perf_counter() - start

    if proc.returncode != 0:
        stderr_snip = (proc.stderr or "").strip().replace("\n", " | ")[:200]
        return None, f"returncode={proc.returncode}: {stderr_snip}"

    matches = STATS_RE.findall(proc.stdout)
    if len(matches) < 2:
        stdout_snip = (proc.stdout or "").strip().replace("\n", " | ")[:200]
        stderr_snip = (proc.stderr or "").strip().replace("\n", " | ")[:200]
        reason = f"stats_parse_fail (stdout: {stdout_snip}"
        if stderr_snip:
            reason += f" | stderr: {stderr_snip}"
        reason += ")"
        return None, reason

    (in0, out0, lat0, and0, lev0) = matches[0]
    (in1, out1, lat1, and1, lev1) = matches[-1]  # last match = final stats

    result = {
        "init_and": int(and0),
        "init_lev": int(lev0),
        "final_and": int(and1),
        "final_lev": int(lev1),
        "runtime_s": round(elapsed, 4),
    }
    return result, None


def load_completed_ids(out_path: Path):
    completed = set()
    if out_path.exists():
        with open(out_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "run_id" in row:
                    completed.add(row["run_id"])
    return completed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--abc", required=True, help="Path to abc executable")
    ap.add_argument("--benchmarks", required=True, help="Path to benchmarks/ dir (contains arithmetic/, random_control/)")
    ap.add_argument("--out", default="data/qor_dataset.csv", help="Output CSV path")
    ap.add_argument("--seq-len", type=int, default=10, help="Sequence length L")
    ap.add_argument("--n-seqs", type=int, default=500, help="Random sequences per circuit")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--timeout", type=int, default=120, help="Per-run ABC timeout (s)")
    ap.add_argument("--circuits", nargs="*", default=None,
                     help="Optional subset of circuit names to run (default: all)")
    ap.add_argument("--exclude-circuits", nargs="*", default=None,
                     help="Optional circuit names to skip (applied after --circuits)")
    args = ap.parse_args()

    abc_path = os.path.expanduser(args.abc)
    benchmarks_dir = Path(os.path.expanduser(args.benchmarks))
    out_path = Path(os.path.expanduser(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not Path(abc_path).exists():
        sys.exit(f"ERROR: abc executable not found at {abc_path}")

    circuits = discover_circuits(benchmarks_dir)
    if args.circuits:
        wanted = set(args.circuits)
        circuits = [c for c in circuits if c[1] in wanted]
    if args.exclude_circuits:
        excluded = set(args.exclude_circuits)
        circuits = [c for c in circuits if c[1] not in excluded]
    if not circuits:
        sys.exit("ERROR: no circuits found. Check --benchmarks path.")

    print(f"Discovered {len(circuits)} circuits:")
    for cat, name, _ in circuits:
        print(f"  [{cat}] {name}")

    completed_ids = load_completed_ids(out_path)
    print(f"Resuming: {len(completed_ids)} runs already completed in {out_path}")

    run_start = time.perf_counter()

    fieldnames = [
        "run_id", "category", "circuit", "seq_type", "seq_len", "seq_str",
        "init_and", "init_lev", "final_and", "final_lev", "runtime_s",
    ]
    write_header = not out_path.exists() or out_path.stat().st_size == 0

    fail_path = out_path.with_name(out_path.stem + "_failures.csv")
    fail_fieldnames = ["run_id", "category", "circuit", "seq_type", "seq_len", "seq_str", "reason"]
    fail_write_header = not fail_path.exists() or fail_path.stat().st_size == 0
    failed_ids = load_completed_ids(fail_path)  # reuse loader; checks 'run_id' column

    rng = random.Random(args.seed)

    with open(out_path, "a", newline="") as f, open(fail_path, "a", newline="") as ff:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        fail_writer = csv.DictWriter(ff, fieldnames=fail_fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()
        if fail_write_header:
            fail_writer.writeheader()
            ff.flush()

        for cat, name, aig_path in circuits:
            print(f"\n=== Circuit: {name} ({cat}) ===")

            # 1. Reference recipes (resyn2, resyn2rs) — always included
            for recipe_name, recipe_cmds in REFERENCE_RECIPES.items():
                rid = run_id_for(name, f"REF:{recipe_name}")
                if rid in completed_ids:
                    continue
                result, reason = run_abc(abc_path, aig_path, recipe_cmds, args.timeout)
                if result is None:
                    print(f"  [FAIL] {recipe_name}: {reason}")
                    if rid not in failed_ids:
                        fail_writer.writerow({
                            "run_id": rid, "category": cat, "circuit": name,
                            "seq_type": "reference", "seq_len": recipe_cmds.count(";") + 1,
                            "seq_str": recipe_cmds, "reason": reason,
                        })
                        ff.flush()
                        failed_ids.add(rid)
                    continue
                writer.writerow({
                    "run_id": rid, "category": cat, "circuit": name,
                    "seq_type": "reference", "seq_len": recipe_cmds.count(";") + 1,
                    "seq_str": recipe_cmds,
                    **{k: v for k, v in result.items()},
                })
                f.flush()
                completed_ids.add(rid)
                print(f"  [OK] {recipe_name}: and {result['init_and']}->{result['final_and']}, "
                      f"lev {result['init_lev']}->{result['final_lev']}")

            # 2. Random sequences
            n_done = 0
            n_failed = 0
            for i in range(args.n_seqs):
                seq = sample_sequence(args.seq_len, rng)
                seq_str = "; ".join(seq)
                rid = run_id_for(name, seq_str)
                if rid in completed_ids:
                    n_done += 1
                    continue
                if rid in failed_ids:
                    continue  # already logged as a failure previously, don't retry/recount

                result, reason = run_abc(abc_path, aig_path, seq_str, args.timeout)
                if result is None:
                    fail_writer.writerow({
                        "run_id": rid, "category": cat, "circuit": name,
                        "seq_type": "random", "seq_len": args.seq_len,
                        "seq_str": seq_str, "reason": reason,
                    })
                    ff.flush()
                    failed_ids.add(rid)
                    n_failed += 1
                    continue  # don't count toward n_seqs

                writer.writerow({
                    "run_id": rid, "category": cat, "circuit": name,
                    "seq_type": "random", "seq_len": args.seq_len,
                    "seq_str": seq_str,
                    **{k: v for k, v in result.items()},
                })
                f.flush()
                completed_ids.add(rid)
                n_done += 1

                if n_done % 50 == 0:
                    print(f"  ... {n_done}/{args.n_seqs} sequences done ({n_failed} failed)")

            print(f"  Finished {name}: {n_done}/{args.n_seqs} random sequences ({n_failed} failed, logged to {fail_path.name})")

    total_elapsed = time.perf_counter() - run_start
    mins, secs = divmod(total_elapsed, 60)
    hrs, mins = divmod(mins, 60)
    print(f"\nDone. Dataset written to {out_path}")
    print(f"Failures (if any) logged to {fail_path}")
    print(f"Total wall-clock time this session: {int(hrs)}h {int(mins)}m {secs:.1f}s")


if __name__ == "__main__":
    main()
