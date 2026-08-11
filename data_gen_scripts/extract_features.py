#!/usr/bin/env python3
"""
Circuit feature extraction script for "Lightweight Learning-Based QoR
Prediction for Logic Synthesis Optimization Sequences" (Section 5.3 of the
proposal: "Engineered circuit features").

Unlike generate_data.py, this runs ONCE PER CIRCUIT (not per sequence),
since these are static structural properties of the circuit itself before
any optimization sequence is applied. Expect this to finish in well under
a minute for all 20 circuits combined.

Extracted features (one row per circuit):
    - pi, po               : primary input / output counts
    - and_count             : AIG node count (== init_and in the QoR dataset)
    - level_count           : logic depth (== init_lev in the QoR dataset)
    - latches               : sequential elements (should be 0 for combinational)
    - fanout_avg            : average fanout per node
    - fanout_max            : maximum fanout of any single node
    - edges                 : total number of edges (fanin connections)

Design notes:
- Uses `print_stats` (for i/o/lat/and/lev) and `print_fanio` (for fanout
  distribution) — both standard ABC commands.
- ABC's exact text output can vary slightly by build/version. This script
  is defensive: it prints the raw ABC stdout for the FIRST circuit it
  processes so you can visually confirm the regexes are matching correctly
  before trusting the rest of the batch. If a field fails to parse, it is
  left blank (empty string) in the CSV rather than silently guessing.

Usage:
    python3 extract_features.py --abc ~/ABC_EPFL/abc/abc \
        --benchmarks ~/ABC_EPFL/benchmarks \
        --out data/circuit_features.csv
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

# Same regex as generate_data.py's STATS_RE, reused for print_stats parsing.
# Example line: "adder : i/o = 256/129  lat = 0  and = 1020  lev = 255"
STATS_RE = re.compile(
    r"i/o\s*=\s*(\d+)\s*/\s*(\d+)\s+lat\s*=\s*(\d+)\s+and\s*=\s*(\d+)\s+lev\s*=\s*(\d+)"
)

# Actual ABC print_fanio output format (confirmed from live run):
#   "Fanins: Max = 2. Ave = 2.00.  Fanouts: Max = 3. Ave =  1.62."
FANIN_MAX_RE = re.compile(r"Fanins:\s*Max\s*=\s*(\d+)")
FANIN_AVE_RE = re.compile(r"Fanins:\s*Max\s*=\s*\d+\.\s*Ave\s*=\s*(\d+\.\d+)\.")
FANOUT_MAX_RE = re.compile(r"Fanouts:\s*Max\s*=\s*(\d+)")
FANOUT_AVE_RE = re.compile(r"Fanouts:\s*Max\s*=\s*\d+\.\s*Ave\s*=\s*(\d+\.\d+)\.")


def discover_circuits(benchmarks_dir: Path):
    circuits = []
    for category in ("arithmetic", "random_control"):
        cat_dir = benchmarks_dir / category
        if not cat_dir.is_dir():
            print(f"WARNING: expected folder not found: {cat_dir}", file=sys.stderr)
            continue
        for aig_file in sorted(cat_dir.glob("*.aig")):
            circuits.append((category, aig_file.stem, aig_file))
    return circuits


def run_abc_features(abc_path: str, aig_path: Path, show_raw: bool = False):
    """Run ABC: read circuit, print_stats, print_fanio. Returns a dict of
    extracted features (values may be None if a field could not be parsed)."""
    script = f"read {aig_path}; print_stats; print_fanio"
    proc = subprocess.run(
        [abc_path, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if show_raw:
        print("----- RAW ABC OUTPUT (first circuit, for regex sanity check) -----")
        print(proc.stdout)
        if proc.stderr.strip():
            print("----- STDERR -----")
            print(proc.stderr)
        print("--------------------------------------------------------------")

    if proc.returncode != 0:
        return None, f"returncode={proc.returncode}: {(proc.stderr or '').strip()[:200]}"

    stdout = proc.stdout

    stats_match = STATS_RE.search(stdout)
    if not stats_match:
        return None, f"print_stats parse failed. Raw stdout: {stdout[:300]}"

    pi, po, lat, and_count, lev = stats_match.groups()

    maxfanin_m = FANIN_MAX_RE.search(stdout)
    avefanin_m = FANIN_AVE_RE.search(stdout)
    maxfanout_m = FANOUT_MAX_RE.search(stdout)
    avefanout_m = FANOUT_AVE_RE.search(stdout)

    fanin_max = maxfanin_m.group(1) if maxfanin_m else ""
    fanin_avg = avefanin_m.group(1) if avefanin_m else ""
    fanout_max = maxfanout_m.group(1) if maxfanout_m else ""
    fanout_avg = avefanout_m.group(1) if avefanout_m else ""
    # edges: total fanout connections == total fanin connections in a DAG;
    # approximate as and_count * fanin_avg (each AND node has 2 fanins, so
    # this should come out close to 2 * and_count for AIGs)
    edges = round(int(and_count) * float(fanin_avg), 2) if fanin_avg else ""

    result = {
        "pi": int(pi),
        "po": int(po),
        "latches": int(lat),
        "and_count": int(and_count),
        "level_count": int(lev),
        "fanin_max": fanin_max,
        "fanin_avg": fanin_avg,
        "fanout_avg": fanout_avg,
        "fanout_max": fanout_max,
        "edges": edges,
    }
    return result, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--abc", required=True, help="Path to abc executable")
    ap.add_argument("--benchmarks", required=True, help="Path to benchmarks/ dir")
    ap.add_argument("--out", default="data/circuit_features.csv", help="Output CSV path")
    ap.add_argument("--circuits", nargs="*", default=None,
                     help="Optional subset of circuit names to run (default: all)")
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
    if not circuits:
        sys.exit("ERROR: no circuits found. Check --benchmarks path.")

    print(f"Discovered {len(circuits)} circuits. Extracting features...\n")

    fieldnames = [
        "category", "circuit", "pi", "po", "latches", "and_count",
        "level_count", "fanin_max", "fanin_avg", "fanout_avg", "fanout_max", "edges",
    ]

    rows = []
    for idx, (cat, name, aig_path) in enumerate(circuits):
        show_raw = (idx == 0)  # print raw ABC output for the first circuit only
        result, reason = run_abc_features(abc_path, aig_path, show_raw=show_raw)
        if result is None:
            print(f"  [FAIL] {name}: {reason}")
            continue
        row = {"category": cat, "circuit": name, **result}
        rows.append(row)
        print(f"  [OK] {name}: pi={result['pi']} po={result['po']} "
              f"and={result['and_count']} lev={result['level_count']} "
              f"fanout_max={result['fanout_max']} fanout_avg={result['fanout_avg']}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Extracted features for {len(rows)}/{len(circuits)} circuits.")
    print(f"Written to {out_path}")
    if len(rows) < len(circuits):
        print("NOTE: some circuits failed extraction — check the [FAIL] lines above.")


if __name__ == "__main__":
    main()
