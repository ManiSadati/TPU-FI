#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
# import matplotlib.pyplot as plt

# Filename format:
# diff_<obsExec>-<injLayer>-<img>-<fi_type>-<iter>.npy
FNAME_RE = re.compile(
    r"^diff_(?P<obs>\d+)-(?P<inj>\d+)-(?P<img>\d+)-(?P<fi>[A-Za-z\-]+)-(?P<it>\d+)\.npy$"
)


def diff_metric(arr: np.ndarray, metric: str) -> float:
    if metric == "sum":
        return float(np.abs(arr).sum())
    if metric == "num_diff":
        return float(np.count_nonzero(arr))
    if metric == "mean":
        return float(np.abs(arr).mean())
    raise ValueError(f"Unknown metric: {metric}")


def mean_and_err(values: List[float], err: str) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    v = np.asarray(values, dtype=np.float64)
    m = float(v.mean())
    if len(v) == 1:
        return m, 0.0
    sd = float(v.std(ddof=1))
    if err == "std":
        return m, sd
    if err == "sem":
        return m, sd / np.sqrt(len(v))
    raise ValueError(f"Unknown err: {err}")


def load_block2_obs_exec_from_mapping(mapping_path: Path, heads_per_block: int = 16) -> Tuple[List[int], Dict[int, int]]:
    """
    Uses your static JSON produced from CSV:
      - fc_by_block_head_exec: [6][16][3] execution-layer indices
      - block2_fc_mid_exec_layers: [16] execution-layer indices (fc_mid)
    Returns:
      block2_obs_exec_by_head: length=16, obs exec layer per head
      obs_exec_to_head: {obs_exec_layer -> head_index}
    """
    data = json.loads(mapping_path.read_text())

    # Prefer explicit list if present
    if "block2_fc_mid_exec_layers" in data:
        mids = data["block2_fc_mid_exec_layers"]
        if len(mids) != heads_per_block:
            raise RuntimeError(
                f"Expected block2_fc_mid_exec_layers length {heads_per_block}, got {len(mids)}"
            )
        block2_obs_exec_by_head = mids
    else:
        # Fallback: derive from fc_by_block_head_exec
        fc_by = data["fc_by_block_head_exec"]
        block2_obs_exec_by_head = [fc_by[2][h][2] for h in range(heads_per_block)]

    if any(x == -1 for x in block2_obs_exec_by_head):
        bad = [i for i, x in enumerate(block2_obs_exec_by_head) if x == -1]
        raise RuntimeError(f"Missing block2 fc_mid exec layers for heads: {bad}")

    obs_exec_to_head = {obs: h for h, obs in enumerate(block2_obs_exec_by_head)}
    return block2_obs_exec_by_head, obs_exec_to_head


def main():
    ap = argparse.ArgumentParser(description="Plot diff_results for block2 observation points using exec-layer mapping")
    ap.add_argument("--diff_dir", default="diff_results", help="Directory containing diff_*.npy files")
    ap.add_argument("--mapping_json", default="head_fc_mapping_3fc_exec.json", help="Static mapping JSON (exec-layer)")
    ap.add_argument("--metric", choices=["sum", "num_diff", "mean"], default="sum")
    ap.add_argument("--err", choices=["std", "sem"], default="std")
    ap.add_argument("--fi_type", default=None, help="Optional filter (e.g., cpu, single, small-box, medium-box)")
    ap.add_argument("--topk", type=int, default=8, help="Number of top observer heads to plot (default 8)")
    ap.add_argument("--out_dir", default="attention_results_exec_top8", help="Output directory")
    args = ap.parse_args()

    diff_dir = Path(args.diff_dir)
    mapping_path = Path(args.mapping_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    block2_obs_exec_by_head, obs_exec_to_head = load_block2_obs_exec_from_mapping(mapping_path, heads_per_block=16)

    # Collect:
    # 1) obs_exec -> list(metric) across all files
    obs_values: Dict[int, List[float]] = defaultdict(list)

    # 2) per image: img -> obs_exec -> list(metric)
    img_obs_values: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    matched = 0
    used = 0

    for fname in os.listdir(diff_dir):
        m = FNAME_RE.match(fname)
        if not m:
            continue
        matched += 1

        obs_exec = int(m.group("obs"))
        img = int(m.group("img"))
        fi = m.group("fi")

        if args.fi_type is not None and fi != args.fi_type:
            continue

        # Only keep block2 observation points (exec-layer ids)
        if obs_exec not in obs_exec_to_head:
            continue

        arr = np.load(diff_dir / fname)
        val = diff_metric(arr, args.metric)

        obs_values[obs_exec].append(val)
        img_obs_values[img][obs_exec].append(val)
        used += 1

    if used == 0:
        raise RuntimeError(f"No usable diff files found. matched={matched}, used(block2)={used}")

    # print(f"INFO: matched pattern files = {matched}, used block2 obs files = {used}")
    # print("INFO: block2 observation exec-layer ids by head:")
    # for h, obs in enumerate(block2_obs_exec_by_head):
    #     print(f"  head {h:2d}: obs_exec={obs}")

    # Compute mean/err per head (all files)
    head_stats = []  # (head, mean, err, obs_exec, n)
    for h in range(16):
        obs_exec = block2_obs_exec_by_head[h]
        vals = obs_values.get(obs_exec, [])
        mval, errv = mean_and_err(vals, args.err)
        head_stats.append((h, mval, errv, obs_exec, len(vals)))

    # Select top-k by mean
    head_stats_sorted = sorted(head_stats, key=lambda x: x[1], reverse=True)
    topk = min(args.topk, 16)
    top = head_stats_sorted[:topk]

    # For plotting: x=0..topk-1 (rank), label indicates original head id
    ranks = list(range(topk))
    means = [t[1] for t in top]
    errs = [t[2] for t in top]
    labels = [f"h{t[0]}" for t in top]  # original head id

    # -------------------------
    # Plot 1: avg metric (top8 heads)
    # -------------------------
    # plt.figure()
    # plt.bar(ranks, means, yerr=errs)
    # plt.xlabel("Top observer head rank (0..7)")
    # plt.ylabel(f"{args.metric} of diff (avg across files)")
    # plt.title(f"Top {topk} observer heads (block2 fc_mid) with {args.err} error bars")
    # plt.xticks(ranks, labels)  # show which original head each rank is
    # plt.tight_layout()
    # out1 = out_dir / f"top{topk}_avg_{args.metric}_{args.err}.png"
    # plt.savefig(out1)
    # plt.close()
    # print(f"Saved plot: {out1}")

    # -------------------------
    # Plot 2: per-image winner score (restricted to top8 heads)
    # winner = max avg diff among only the top heads, per image
    # -------------------------
    top_heads = [t[0] for t in top]
    top_obs_exec = [block2_obs_exec_by_head[h] for h in top_heads]

    scores = [0] * topk
    img_ids = sorted(img_obs_values.keys())

    for img in img_ids:
        per_rank_mean = []
        for obs_exec in top_obs_exec:
            vals = img_obs_values[img].get(obs_exec, [])
            per_rank_mean.append(float(np.mean(vals)) if vals else float("-inf"))
        winner_rank = int(np.argmax(per_rank_mean))
        if per_rank_mean[winner_rank] != float("-inf"):
            scores[winner_rank] += 1

    # plt.figure()
    # plt.bar(ranks, scores)
    # plt.xlabel("Top observer head rank (0..7)")
    # plt.ylabel("Winner score (#images where head had max avg diff)")
    # plt.title(f"Top {topk} observer heads: per-image winner score")
    # plt.xticks(ranks, labels)
    # plt.tight_layout()
    # out2 = out_dir / f"top{topk}_winner_score.png"
    # plt.savefig(out2)
    # plt.close()
    # print(f"Saved plot: {out2}")

    # -------------------------
    # Write a readable mapping table
    # -------------------------
    summary = out_dir / "top_heads_summary.txt"
    with open(summary, "w") as f:
        f.write(f"Top {topk} observer heads (sorted by mean {args.metric}):\n")
        f.write("rank  head  obs_exec_layer  mean  err  n_samples\n")
        for r, (h, mval, errv, obs_exec, n) in enumerate(top):
            f.write(f"{r:>4}  {h:>4}  {obs_exec:>13}  {mval:.6g}  {errv:.6g}  {n}\n")
        f.write("\nMax Diff heads by rank:\n")
        for r, sc in enumerate(scores):
            f.write(f"rank {r} ({labels[r]}): {sc}\n")


    print(f"Top {topk} observer heads (sorted by mean {args.metric}):\n")
    print("rank  head  obs_exec_layer  mean  err  n_samples\n")
    for r, (h, mval, errv, obs_exec, n) in enumerate(top):
        print(f"{r:>4}  {h:>4}  {obs_exec:>13}  {mval:.6g}  {errv:.6g}  {n}\n")
    print("\nMax Diff heads by rank:\n")
    for r, sc in enumerate(scores):
        print(f"rank {r} ({labels[r]}): {sc}\n")


    print(f"Wrote summary: {summary}")


if __name__ == "__main__":
    main()