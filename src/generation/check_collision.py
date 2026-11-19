#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collision checker for semantic IDs saved by generate_semantic_ids.py (.pt format).

Update requested:
- Do NOT use merged_index.tsv anymore.
- semantic_ids_all.pt already contains:
    domain_id: LongTensor [N]  (domain numeric id)
    item_id:   LongTensor [N]  (original numeric id, e.g. 577)
    semantic_id: LongTensor [N, L]
- Resolve titles by:
    domain_id -> domain name -> DOMAIN_ITEM_CSVS[domain_name] -> lookup by original item_id.
- No argparse; hardcode paths in main().

Assumptions:
- item csv is 3 columns (no header): item_id, zh, en
- item_id column contains strings like "v577" (or any string containing digits)
"""

from collections import Counter
from typing import Dict, Tuple, List, Optional

import torch
import pandas as pd

ITEM_ID_NUM_REGEX = r"(\d+)"


# -------------------------
# IO
# -------------------------
def load_semantic_pt(pt_path: str):
    obj = torch.load(pt_path, map_location="cpu")
    for k in ["domain_id", "item_id", "semantic_id"]:
        if k not in obj:
            raise ValueError(f"Missing key '{k}' in {pt_path}. Keys: {list(obj.keys())}")

    domain_id = obj["domain_id"]
    item_id = obj["item_id"]
    semantic_id = obj["semantic_id"]
    if isinstance(obj["domain_id"], int):
        n = obj["item_id"].shape[0]
        domain_id = torch.full(
            (n,),
            fill_value=obj["domain_id"],
            dtype=torch.long,
            device=obj["item_id"].device,  # 保持和 item_id 在同一个 device
        )



    if not torch.is_tensor(domain_id) or domain_id.dtype != torch.long:
        raise ValueError("domain_id must be torch.LongTensor")
    if not torch.is_tensor(item_id) or item_id.dtype != torch.long:
        raise ValueError("item_id must be torch.LongTensor (original int ids)")
    if not torch.is_tensor(semantic_id) or semantic_id.dtype != torch.long or semantic_id.dim() != 2:
        raise ValueError("semantic_id must be torch.LongTensor of shape [N, L]")

    meta = {k: obj.get(k) for k in ["ckpt_path", "model_type", "num_layers", "config", "ckpt_meta"]}
    return domain_id, item_id, semantic_id, meta


def build_domain_tables(domain_item_csvs: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    domain_item_csvs: domain_name -> csv_path

    Returns:
      domain_tables[domain_name] = df with columns:
        - item_id (original string, e.g. "v577")
        - zh, en
        - item_id_num (int extracted from item_id)
    """
    tables: Dict[str, pd.DataFrame] = {}
    for domain, csv_path in domain_item_csvs.items():
        df = pd.read_csv(csv_path, header=None, names=["item_id", "zh", "en"])
        df["item_id_num"] = df["item_id"].astype(str).str.extract(ITEM_ID_NUM_REGEX).astype(float)
        df = df.dropna(subset=["item_id_num"]).copy()
        df["item_id_num"] = df["item_id_num"].astype(int)

        # Optional: speed up lookup by building an index
        df = df.drop_duplicates(subset=["item_id_num"], keep="first").reset_index(drop=True)

        tables[domain] = df
    return tables


# -------------------------
# Resolve & Display
# -------------------------
def _resolve_domain_id_to_name(config: Optional[dict], domain_item_csvs: Dict[str, str]) -> Dict[int, str]:
    """
    Best-effort mapping from numeric domain_id -> domain name.

    Priority:
    1) If config has data.domain_configs with (name, domain_id), use it.
    2) Else fallback to the order of DOMAIN_ITEM_CSVS keys (0..N-1).
       (This requires that your dataset used the same ordering.)

    Returns:
      id2name: Dict[int, str]
    """
    id2name: Dict[int, str] = {}

    # 1) From config if available
    if isinstance(config, dict):
        data = config.get("data", {})
        dcfgs = data.get("domain_configs", None)
        if isinstance(dcfgs, list) and len(dcfgs) > 0:
            # Some configs may not contain domain_id per entry; try anyway
            for i, dc in enumerate(dcfgs):
                if not isinstance(dc, dict):
                    continue
                name = dc.get("name", None)
                did = dc.get("domain_id", None)
                if name is None:
                    continue
                if did is None:
                    # fallback to list index if domain_id not provided in entry
                    did = i
                try:
                    did = int(did)
                except Exception:
                    continue
                id2name[did] = str(name)

    if id2name:
        return id2name

    # 2) Fallback to key order
    names = list(domain_item_csvs.keys())
    return {i: names[i] for i in range(len(names))}


def show_collision_group(
    sid: Tuple[int, ...],
    members: List[Tuple[int, int]],  # list of (domain_id, item_id)
    id2name: Dict[int, str],
    domain_tables: Dict[str, pd.DataFrame],
) -> None:
    print(f"\nSemantic ID {sid} hit {len(members)} items:")
    print("members (domain_id, item_id):", members)

    print("\nResolved titles:\n")
    for did, iid in members:
        domain_name = id2name.get(did, None)
        if domain_name is None:
            print(f"[WARN] Unknown domain_id={did}. item_id={iid}")
            continue

        tab = domain_tables.get(domain_name)
        if tab is None:
            print(f"[WARN] domain '{domain_name}' not in DOMAIN_ITEM_CSVS mapping. item_id={iid}")
            continue

        rec = tab[tab["item_id_num"] == int(iid)]
        if rec.empty:
            print(f"[MISS] {domain_name} item {iid} not found in csv (expected something like v{iid})")
            continue

        r0 = rec.iloc[0]
        print(f"domain={domain_name} (id={did})  item_id={r0['item_id']} (num={iid})")
        print(f"  zh: {r0['zh']}")
        print(f"  en: {r0['en']}")
        print("-" * 80)


# -------------------------
# Main (hardcoded)
# -------------------------
def main():
    pkl_path = "/home/hao/FRec/DisenID/codebook/ninerec_256/semantic_ids_all.pt"
    codebook_width = 256

    # domain name -> item csv
    DOMAIN_ITEM_CSVS: Dict[str, str] = {
        "QB": "/mnt/data_2/data_oleg/recsys/ninerec/Downstream_datasets/QB/QB_item.csv",
        "TN": "/mnt/data_2/data_oleg/recsys/ninerec/Downstream_datasets/TN/TN_item.csv",
        "Bili": "/mnt/data_2/data_oleg/recsys/ninerec/Source_datasets/Bili_500K/Bili_500K_item.csv"
    }

    domain_id, item_id, semantic_id, meta = load_semantic_pt(pkl_path)

    N = int(semantic_id.size(0))
    L = int(semantic_id.size(1))
    print("Loaded:", pkl_path)
    print(f"Total items: {N}")
    print(f"Num layers : {L}")
    print(f"Codebook width: {codebook_width}")
    if meta.get("model_type") is not None:
        print("Model type:", meta["model_type"])
    if meta.get("ckpt_path") is not None:
        print("Checkpoint:", meta["ckpt_path"])
    print()

    # 1) Per-layer coverage
    for layer in range(L):
        used = int(torch.unique(semantic_id[:, layer]).numel())
        coverage = used / float(codebook_width)
        print(f"Layer {layer}:")
        print(f"  Used clusters : {used} / {codebook_width}")
        print(f"  Coverage      : {coverage:.4f}")
        print()

    # 2) Collision metrics (exact match over full code)
    semantic_tuples = [tuple(row.tolist()) for row in semantic_id]
    counter = Counter(semantic_tuples)

    num_unique = len(counter)
    num_collided_items = sum(v for v in counter.values() if v > 1)
    num_collision_groups = sum(1 for v in counter.values() if v > 1)

    collision_item_rate = num_collided_items / N
    collision_group_rate = num_collision_groups / num_unique if num_unique > 0 else 0.0

    print("Collision analysis (full semantic ID):")
    print(f"  Unique semantic IDs     : {num_unique}")
    print(f"  Items in collisions     : {num_collided_items}")
    print(f"  Collision item rate     : {collision_item_rate:.4f}")
    print(f"  Collision group rate    : {collision_group_rate:.4f}")
    print()

    # Build domain tables & domain_id -> domain_name mapping
    domain_tables = build_domain_tables(DOMAIN_ITEM_CSVS)
    id2name = _resolve_domain_id_to_name(meta.get("config", None), DOMAIN_ITEM_CSVS)

    # Build sid -> list[(domain_id, item_id)]
    dom_list = domain_id.tolist()
    item_list = item_id.tolist()

    sid_to_members: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
    for sid, did, iid in zip(semantic_tuples, dom_list, item_list):
        sid_to_members.setdefault(sid, []).append((int(did), int(iid)))

    # Show top-k collision groups
    topk = 10
    print(f"Top-{topk} largest collision groups:")
    shown = 0
    for sid, cnt in counter.most_common(topk):
        if cnt > 1:
            print(" ", sid, "->", cnt)
            show_collision_group(
                sid,
                members=sid_to_members[sid],
                id2name=id2name,
                domain_tables=domain_tables,
            )
            shown += 1

    if shown == 0:
        print("  (No collisions in top-k; all semantic IDs are unique.)")


if __name__ == "__main__":
    main()