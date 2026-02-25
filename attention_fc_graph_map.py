#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tensorflow as tf

# --- Search boundaries (to avoid wandering into other heads/blocks too much) ---
STOP_FORWARD = {"ADD", "SOFTMAX"}
STOP_BACKWARD = {"SOFTMAX"}


def op_name(ops, idx: int) -> str:
    return ops[idx].get("op_name", "")


def build_maps(interpreter):
    """Build tensor producer/consumer maps from TFLite ops."""
    ops = interpreter._get_ops_details()

    producer: Dict[int, int] = {}                # tensor_idx -> producing op_idx
    consumers: Dict[int, List[int]] = defaultdict(list)  # tensor_idx -> list of consuming op_idx

    for op in ops:
        oi = op["index"]
        for t in op.get("outputs", []):
            if t is None or t < 0:
                continue
            producer[t] = oi
        for t in op.get("inputs", []):
            if t is None or t < 0:
                continue
            consumers[t].append(oi)

    return ops, producer, consumers


def nearest_backward_fc(
    ops,
    producer: Dict[int, int],
    start_op: int,
    max_hops: int = 30
) -> Optional[int]:
    """Backwards BFS to the first FULLY_CONNECTED op."""
    seen = {start_op}
    q = deque([(start_op, 0)])

    while q:
        cur, d = q.popleft()
        if d >= max_hops:
            continue

        for t in ops[cur].get("inputs", []):
            if t is None or t < 0:
                continue
            p = producer.get(t)
            if p is None or p in seen:
                continue

            pname = op_name(ops, p)
            if pname == "FULLY_CONNECTED":
                return p
            if pname in STOP_BACKWARD:
                continue

            seen.add(p)
            q.append((p, d + 1))

    return None


def bounded_forward_first_fc(
    ops,
    consumers: Dict[int, List[int]],
    start_op: int,
    max_hops: int = 30
) -> Optional[int]:
    """Forward BFS to the first FULLY_CONNECTED op, not crossing ADD/SOFTMAX."""
    seen = {start_op}
    q = deque([(start_op, 0)])

    while q:
        cur, d = q.popleft()
        if d >= max_hops:
            continue

        if cur != start_op and op_name(ops, cur) in STOP_FORWARD:
            continue

        for t in ops[cur].get("outputs", []):
            if t is None or t < 0:
                continue
            for c in consumers.get(t, []):
                if c in seen:
                    continue

                cname = op_name(ops, c)
                if cname == "FULLY_CONNECTED":
                    return c

                if cname in STOP_FORWARD:
                    seen.add(c)
                    continue

                seen.add(c)
                q.append((c, d + 1))

    return None


@dataclass
class AttentionFCMap:
    # [n_blocks][heads_per_block][3] = [map(fc_logits-1), map(fc_logits), map(fc_mid)]
    fc_by_block_head: List[List[List[int]]]
    # [0..N-1]
    all_fc_new_ids_sorted: List[int]
    # map: new_fc_id -> (block, head, pos_in_3)
    fc_newid_to_location: Dict[int, Tuple[int, int, int]]
    # fc_mid new ids for each head in block 2 (length heads_per_block)
    block2_fc_mids: List[int]
    # metadata
    heads_per_block: int
    n_blocks: int
    ignored_lm_softmax_op: Optional[int]


def build_attention_fc_map_from_tflite(
    tflite_path: str,
    heads_per_block: int = 16,
    expected_blocks: Optional[int] = 6,
    max_hops: int = 30,
    ignore_last_softmax_as_lm: bool = True,
) -> AttentionFCMap:
    """
    Builds the [block][head][3] FC-new-id mapping directly from the TFLite graph.
    """
    interpreter = tf.lite.Interpreter(
        model_path=tflite_path,
        experimental_preserve_all_tensors=True
    )
    interpreter.allocate_tensors()

    ops, producer, consumers = build_maps(interpreter)

    # ---- All FC ops -> sorted -> new FC ids 0..N-1 ----
    all_fc_ops_sorted = sorted(
        [op["index"] for op in ops if op_name(ops, op["index"]) == "FULLY_CONNECTED"]
    )
    fc_new_id = {op_idx: new_id for new_id, op_idx in enumerate(all_fc_ops_sorted)}
    all_fc_new_ids_sorted = list(range(len(all_fc_ops_sorted)))

    # ---- All softmax ops -> sorted -> head ids ----
    all_softmax_ops_sorted = sorted(
        [op["index"] for op in ops if op_name(ops, op["index"]) == "SOFTMAX"]
    )

    ignored_lm = None
    if ignore_last_softmax_as_lm and all_softmax_ops_sorted:
        ignored_lm = all_softmax_ops_sorted[-1]
        all_softmax_ops_sorted = all_softmax_ops_sorted[:-1]

    # Infer number of blocks
    if len(all_softmax_ops_sorted) % heads_per_block != 0:
        # still proceed, but blocks won't be perfect
        n_blocks = len(all_softmax_ops_sorted) // heads_per_block
        if expected_blocks is not None:
            # prefer expected if provided
            n_blocks = expected_blocks
    else:
        n_blocks = len(all_softmax_ops_sorted) // heads_per_block

    if expected_blocks is not None:
        n_blocks = expected_blocks

    # Initialize [blocks][heads][3] with -1
    fc_by_block_head: List[List[List[int]]] = [
        [[-1, -1, -1] for _ in range(heads_per_block)]
        for _ in range(n_blocks)
    ]

    # Fill by scanning heads in softmax order
    for hid, sm_op in enumerate(all_softmax_ops_sorted):
        block = hid // heads_per_block
        head = hid % heads_per_block
        if block < 0 or block >= n_blocks:
            # ignore extra heads outside expected blocks
            continue

        fc_logits = nearest_backward_fc(ops, producer, sm_op, max_hops=max_hops)
        fc_mid = bounded_forward_first_fc(ops, consumers, sm_op, max_hops=max_hops)
        if fc_logits is None or fc_mid is None:
            continue

        # final rule: 3 candidates only
        cand_ops = [fc_logits - 1, fc_logits, fc_mid]
        # Keep only true FC ops and map to new ids
        cand_new_ids = []
        for op_idx in cand_ops:
            if op_idx in fc_new_id:
                cand_new_ids.append(fc_new_id[op_idx])
            else:
                cand_new_ids.append(-1)

        fc_by_block_head[block][head] = cand_new_ids

    # Build reverse lookup: new_fc_id -> where it appears
    fc_newid_to_location: Dict[int, Tuple[int, int, int]] = {}
    for b in range(n_blocks):
        for h in range(heads_per_block):
            for k in range(3):
                nid = fc_by_block_head[b][h][k]
                if nid == -1:
                    continue
                fc_newid_to_location[nid] = (b, h, k)

    # block 2 fc_mids = index 2 in the 3-list
    block2 = 2
    block2_fc_mids = []
    if block2 < n_blocks:
        block2_fc_mids = [fc_by_block_head[block2][h][2] for h in range(heads_per_block)]
        block2_fc_mids = [x for x in block2_fc_mids if x != -1]

    return AttentionFCMap(
        fc_by_block_head=fc_by_block_head,
        all_fc_new_ids_sorted=all_fc_new_ids_sorted,
        fc_newid_to_location=fc_newid_to_location,
        block2_fc_mids=block2_fc_mids,
        heads_per_block=heads_per_block,
        n_blocks=n_blocks,
        ignored_lm_softmax_op=ignored_lm,
    )


def get_logged_layers_attention(
    fc_ind: int,
    attn_map: AttentionFCMap,
    log_if_block: int = 0,
    return_block: int = 2,
) -> List[int]:
    """
    fc_ind is a *new FC id* (what you use in FI).
    If fc_ind belongs to block `log_if_block`, return fc_mid new-ids of block `return_block`.
    Else return [].
    """
    loc = attn_map.fc_newid_to_location.get(fc_ind)
    if loc is None:
        return []
    block, _head, _pos = loc
    if block != log_if_block:
        return []

    # Return fc_mids of return_block
    if return_block == 2:
        return list(attn_map.block2_fc_mids)

    # Generic: compute from map
    if return_block < 0 or return_block >= attn_map.n_blocks:
        return []
    mids = [attn_map.fc_by_block_head[return_block][h][2] for h in range(attn_map.heads_per_block)]
    return [x for x in mids if x != -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tflite", help="Path to .tflite file")
    ap.add_argument("--heads_per_block", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--max_hops", type=int, default=30)
    ap.add_argument("--print_block", type=int, default=0)
    args = ap.parse_args()

    attn_map = build_attention_fc_map_from_tflite(
        args.tflite,
        heads_per_block=args.heads_per_block,
        expected_blocks=args.blocks,
        max_hops=args.max_hops,
        ignore_last_softmax_as_lm=True,
    )

    print(f"INFO: ignored_lm_softmax_op = {attn_map.ignored_lm_softmax_op}")
    print(f"INFO: n_blocks={attn_map.n_blocks}, heads_per_block={attn_map.heads_per_block}")
    print(f"INFO: total FC new-ids = {len(attn_map.all_fc_new_ids_sorted)}")
    print(f"INFO: block2 fc_mids (count={len(attn_map.block2_fc_mids)}): {attn_map.block2_fc_mids}")

    b = args.print_block
    print(f"\n=== BLOCK {b} fc_by_block_head (each head -> [fc_logits-1, fc_logits, fc_mid] as NEW ids) ===")
    for h in range(attn_map.heads_per_block):
        print(f"  head {h:2d}: {attn_map.fc_by_block_head[b][h]}")

    # Demo get_logged_layers_attention
    # Pick the first FC id in block0, head0, pos0 (if exists)
    demo = attn_map.fc_by_block_head[0][0][0]
    if demo != -1:
        print(f"\nDemo: get_logged_layers_attention(fc_ind={demo}) -> {get_logged_layers_attention(demo, attn_map)}")
    else:
        print("\nDemo skipped: block0/head0/pos0 is -1")

import json

class AttentionMapping:
    def __init__(self, json_path):
        with open(json_path) as f:
            self.data = json.load(f)

        # [6][16][3]
        self.fc_by_block_head_exec = self.data["fc_by_block_head_exec"]

        # length 16
        self.block2_fc_mid_exec_layers = self.data["block2_fc_mid_exec_layers"]

        # Build fast lookup: exec_layer -> block
        self.exec_layer_to_block = {}

        for b in range(len(self.fc_by_block_head_exec)):
            for h in range(len(self.fc_by_block_head_exec[b])):
                for layer in self.fc_by_block_head_exec[b][h]:
                    if layer != -1:
                        self.exec_layer_to_block[layer] = b

    def get_logged_layers_attention(self, fi_layer_exec):
        """
        If injection is in block 0 -> return fc_mid layers of block 2
        Else -> []
        """
        block = self.exec_layer_to_block.get(fi_layer_exec, None)
        if block == 0:
            return list(self.block2_fc_mid_exec_layers)
        return []

if __name__ == "__main__":
    main()