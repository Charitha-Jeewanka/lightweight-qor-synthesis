#!/usr/bin/env python3
"""
Validation script for QoR dataset CSVs produced by generate_data.py.

Checks:
  1. Row counts per circuit (vs. expected n-seqs + 2 reference rows)
  2. Duplicate run_id detection
  3. Failures file summary (row count, reasons breakdown)
  4. init_and / init_lev consistency per circuit (should be constant)
  5. Sanity ranges: final_and <= init_and? (flagged, not failed, since some
     sequences can legitimately increase area), runtime outliers, missing/NaN values

Usage:
    python3 validate_dataset.py --data data/qor_dataset_L10.csv --expected-n-seqs 2000
    python3 validate_dataset.py --data data/qor_dataset_L10.csv --expected-n-seqs 2000 \
        --circuit-overrides hyp=100
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def parse_overrides(override_strs):
    overrides = {}
    for item in override_strs or []:
        name, val = item.split("=")
        overrides[name.strip()] = int(val)
    return overrides


def load_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Path to main dataset CSV")
    ap.add_argument("--failures", default=None,
                     help="Path to failures CSV (default: inferred as <data>_failures.csv)")
    ap.add_argument("--expected-n-seqs", type=int, default=2000,
                     help="Expected number of random sequences per circuit")
    ap.add_argument("--circuit-overrides", nargs="*", default=None,
                     help="Per-circuit expected n-seqs overrides, e.g. hyp=100")
    ap.add_argument("--expected-circuits", type=int, default=20,
                     help="Expected total number of distinct circuits")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"ERROR: data file not found: {data_path}")

    if args.failures:
        fail_path = Path(args.failures)
    else:
        fail_path = data_path.with_name(data_path.stem + "_failures.csv")

    overrides = parse_overrides(args.circuit_overrides)

    rows = load_csv(data_path)
    print(f"Loaded {len(rows)} rows from {data_path}\n")

    issues = []  # collected problems, printed as a summary at the end

    # ------------------------------------------------------------------
    # 1. Duplicate run_id check
    # ------------------------------------------------------------------
    print("=== 1. Duplicate run_id check ===")
    run_id_counts = defaultdict(int)
    for r in rows:
        run_id_counts[r["run_id"]] += 1
    dupes = {rid: c for rid, c in run_id_counts.items() if c > 1}
    if dupes:
        print(f"  FOUND {len(dupes)} duplicated run_id(s). Examples: {list(dupes.items())[:5]}")
        issues.append(f"{len(dupes)} duplicate run_id(s) found")
    else:
        print("  OK: no duplicate run_id values.")
    print()

    # ------------------------------------------------------------------
    # 2. Row counts per circuit
    # ------------------------------------------------------------------
    print("=== 2. Row counts per circuit ===")
    by_circuit = defaultdict(list)
    for r in rows:
        by_circuit[r["circuit"]].append(r)

    if len(by_circuit) != args.expected_circuits:
        print(f"  WARNING: expected {args.expected_circuits} distinct circuits, found {len(by_circuit)}.")
        issues.append(f"expected {args.expected_circuits} circuits, found {len(by_circuit)}")

    header = f"  {'circuit':<14}{'total':>8}{'reference':>12}{'random':>10}{'expected':>10}{'status':>10}"
    print(header)
    for name in sorted(by_circuit.keys()):
        crows = by_circuit[name]
        n_ref = sum(1 for r in crows if r["seq_type"] == "reference")
        n_rand = sum(1 for r in crows if r["seq_type"] == "random")
        expected = overrides.get(name, args.expected_n_seqs)
        status = "OK" if n_rand == expected and n_ref == 2 else "CHECK"
        if status == "CHECK":
            issues.append(
                f"circuit '{name}': got {n_rand} random rows (expected {expected}), "
                f"{n_ref} reference rows (expected 2)"
            )
        print(f"  {name:<14}{len(crows):>8}{n_ref:>12}{n_rand:>10}{expected:>10}{status:>10}")
    print()

    # ------------------------------------------------------------------
    # 3. Failures file summary
    # ------------------------------------------------------------------
    print("=== 3. Failures file summary ===")
    if fail_path.exists():
        fail_rows = load_csv(fail_path)
        print(f"  {len(fail_rows)} failure row(s) in {fail_path}")
        if fail_rows:
            reason_counts = defaultdict(int)
            for fr in fail_rows:
                # bucket by reason prefix (before first colon/paren) for a compact summary
                reason = fr.get("reason", "unknown")
                bucket = reason.split(":")[0].split("(")[0].strip()
                reason_counts[bucket] += 1
            for reason, c in sorted(reason_counts.items(), key=lambda x: -x[1]):
                print(f"    {reason:<30}{c:>6}")
            issues.append(f"{len(fail_rows)} failure(s) logged in {fail_path.name}")
    else:
        print(f"  No failures file found at {fail_path} (assuming zero failures).")
    print()

    # ------------------------------------------------------------------
    # 4. init_and / init_lev consistency per circuit
    # ------------------------------------------------------------------
    print("=== 4. init_and / init_lev consistency per circuit ===")
    inconsistent = []
    for name, crows in sorted(by_circuit.items()):
        and_vals = set(r["init_and"] for r in crows)
        lev_vals = set(r["init_lev"] for r in crows)
        if len(and_vals) > 1 or len(lev_vals) > 1:
            inconsistent.append((name, and_vals, lev_vals))
    if inconsistent:
        for name, and_vals, lev_vals in inconsistent:
            print(f"  INCONSISTENT '{name}': init_and values={and_vals}, init_lev values={lev_vals}")
            issues.append(f"circuit '{name}' has inconsistent init_and/init_lev across rows")
    else:
        print("  OK: init_and and init_lev are constant within every circuit.")
    print()

    # ------------------------------------------------------------------
    # 5. Sanity ranges: missing values, final_and > init_and outliers, runtime outliers
    # ------------------------------------------------------------------
    print("=== 5. Sanity checks ===")
    numeric_fields = ["init_and", "init_lev", "final_and", "final_lev", "runtime_s"]
    missing_count = 0
    increased_area_count = 0
    increased_lev_count = 0
    runtimes = []

    for r in rows:
        for f in numeric_fields:
            val = r.get(f, "")
            if val is None or val == "":
                missing_count += 1
                break
        try:
            init_and = int(r["init_and"])
            final_and = int(r["final_and"])
            init_lev = int(r["init_lev"])
            final_lev = int(r["final_lev"])
            runtime = float(r["runtime_s"])
            runtimes.append(runtime)
            if final_and > init_and:
                increased_area_count += 1
            if final_lev > init_lev:
                increased_lev_count += 1
        except (ValueError, KeyError):
            continue

    print(f"  Rows with missing numeric fields: {missing_count}")
    if missing_count:
        issues.append(f"{missing_count} row(s) with missing numeric fields")

    print(f"  Sequences where final_and > init_and (area increased): {increased_area_count} "
          f"({100*increased_area_count/max(len(rows),1):.1f}% of rows) -- expected to be nonzero, flagged not failed")
    print(f"  Sequences where final_lev > init_lev (depth increased): {increased_lev_count} "
          f"({100*increased_lev_count/max(len(rows),1):.1f}% of rows) -- expected to be nonzero, flagged not failed")

    if runtimes:
        runtimes.sort()
        n = len(runtimes)
        p50 = runtimes[n // 2]
        p95 = runtimes[int(n * 0.95)]
        p99 = runtimes[int(n * 0.99)]
        print(f"  Runtime (s): min={runtimes[0]:.3f}  p50={p50:.3f}  p95={p95:.3f}  "
              f"p99={p99:.3f}  max={runtimes[-1]:.3f}")
        # flag extreme outliers: runtime > 10x the p99, just as a heads-up
        extreme = [rt for rt in runtimes if rt > 10 * p99]
        if extreme:
            print(f"  NOTE: {len(extreme)} row(s) with runtime > 10x p99 ({p99:.3f}s) -- worth a manual look.")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=== SUMMARY ===")
    if issues:
        print(f"  {len(issues)} issue(s) flagged for review:")
        for i in issues:
            print(f"   - {i}")
        sys.exit(1)
    else:
        print("  All checks passed cleanly. Dataset looks good.")
        sys.exit(0)


if __name__ == "__main__":
    main()
