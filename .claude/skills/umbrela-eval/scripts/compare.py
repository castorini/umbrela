#!/usr/bin/env python3
"""Compare two umbrela modified qrel files side-by-side.

Usage:
    python3 compare.py --qrel dl19-passage --run-a <modified_qrel_a> --run-b <modified_qrel_b>

Reports per-query agreement, label distribution comparison, and Cohen's kappa.
"""

import argparse
import sys
from collections import Counter


def load_qrel(path: str) -> dict[str, dict[str, int]]:
    """Load TREC qrel file into dict[qid][docid] = label."""
    qrel: dict[str, dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docid, label = parts[0], parts[1], parts[2], int(parts[3])
            qrel.setdefault(qid, {})[docid] = label
    return qrel


def cohens_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Compute Cohen's kappa for two lists of labels."""
    if len(labels_a) != len(labels_b) or len(labels_a) == 0:
        return 0.0
    n = len(labels_a)
    all_labels = sorted(set(labels_a) | set(labels_b))
    # Observed agreement
    agree = sum(1 for a, b in zip(labels_a, labels_b, strict=False) if a == b)
    po = agree / n
    # Expected agreement
    pe = 0.0
    for label in all_labels:
        count_a = sum(1 for a in labels_a if a == label)
        count_b = sum(1 for b in labels_b if b == label)
        pe += (count_a / n) * (count_b / n)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two modified qrel files")
    parser.add_argument("--qrel", required=True, help="Qrel name (for display)")
    parser.add_argument("--run-a", required=True, help="First modified qrel file")
    parser.add_argument("--run-b", required=True, help="Second modified qrel file")
    args = parser.parse_args()

    a = load_qrel(args.run_a)
    b = load_qrel(args.run_b)

    # Find common pairs
    common_labels_a = []
    common_labels_b = []
    for qid in sorted(set(a.keys()) & set(b.keys())):
        for docid in sorted(set(a.get(qid, {}).keys()) & set(b.get(qid, {}).keys())):
            common_labels_a.append(a[qid][docid])
            common_labels_b.append(b[qid][docid])

    if not common_labels_a:
        print("No common qid-docid pairs found.", file=sys.stderr)
        sys.exit(1)

    # Agreement
    n = len(common_labels_a)
    agree = sum(
        1 for la, lb in zip(common_labels_a, common_labels_b, strict=False) if la == lb
    )
    kappa = cohens_kappa(common_labels_a, common_labels_b)

    print(f"Qrel: {args.qrel}")
    print(f"Run A: {args.run_a}")
    print(f"Run B: {args.run_b}")
    print(f"Common pairs: {n}")
    print(f"Agreement: {agree}/{n} ({agree / n * 100:.1f}%)")
    print(f"Cohen's kappa: {kappa:.3f}")

    # Label distributions
    dist_a = Counter(common_labels_a)
    dist_b = Counter(common_labels_b)
    print("\nLabel distribution (common pairs):")
    print(f"  {'Label':<6} {'Run A':>8} {'Run B':>8}")
    for label in sorted(set(dist_a.keys()) | set(dist_b.keys())):
        print(f"  {label:<6} {dist_a.get(label, 0):>8} {dist_b.get(label, 0):>8}")

    # Per-label agreement
    print("\nPer-label agreement:")
    for label in sorted(set(common_labels_a) | set(common_labels_b)):
        both = sum(
            1
            for la, lb in zip(common_labels_a, common_labels_b, strict=False)
            if la == label and lb == label
        )
        in_a = sum(1 for la in common_labels_a if la == label)
        in_b = sum(1 for lb in common_labels_b if lb == label)
        prec = both / in_b if in_b > 0 else 0
        rec = both / in_a if in_a > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  Label {label}: precision={prec:.3f}, recall={rec:.3f}, F1={f1:.3f}")


if __name__ == "__main__":
    main()
