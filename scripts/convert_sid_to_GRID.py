import os
from collections import defaultdict

import torch

INPUT_PT_PATH = "/home/hao/FRec/DisenID/codebook/qwen_dsdtrqvae_bili500k_TN_QB/semantic_ids_all.pt"
OUTPUT_TENSOR_PATH = "/home/hao/FRec/DisenID/codebook/qwen_dsdtrqvae_bili500k_TN_QB/sid_dsdtrqvae.pt"


def build_dedup_suffix(semantic_id_sorted: torch.Tensor) -> torch.Tensor:
    """
    For each original semantic_id row [L], append one deduplication suffix:
    - 0 if this semantic_id is unique
    - 1..K if this semantic_id appears K times (collision group), assigned in sorted order

    Input:
        semantic_id_sorted: [N, L]
    Output:
        dedup_suffix: [N]
    """
    n = semantic_id_sorted.shape[0]

    # group row indices by semantic id tuple
    groups = defaultdict(list)
    for i in range(n):
        key = tuple(semantic_id_sorted[i].tolist())
        groups[key].append(i)

    dedup_suffix = torch.zeros(n, dtype=torch.long)

    collision_group_count = 0
    collision_item_count = 0
    max_collision_size = 1

    for _, indices in groups.items():
        group_size = len(indices)
        max_collision_size = max(max_collision_size, group_size)

        if group_size > 1:
            collision_group_count += 1
            collision_item_count += group_size

            # enumerate collision items as 1..K
            for rank, idx in enumerate(indices, start=1):
                dedup_suffix[idx] = rank
        else:
            # unique semantic id -> suffix 0
            dedup_suffix[indices[0]] = 1

    print(f"collision groups = {collision_group_count}")
    print(f"collision items = {collision_item_count}")
    print(f"max collision size = {max_collision_size}")

    return dedup_suffix


def main():
    obj = torch.load(INPUT_PT_PATH, map_location="cpu")

    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict, got {type(obj)}")

    if "semantic_id" not in obj or "item_id" not in obj:
        raise KeyError("Input file must contain keys: semantic_id, item_id")

    semantic_id = obj["semantic_id"]
    item_id = obj["item_id"]

    if not isinstance(semantic_id, torch.Tensor):
        semantic_id = torch.as_tensor(semantic_id)
    if not isinstance(item_id, torch.Tensor):
        item_id = torch.as_tensor(item_id)

    semantic_id = semantic_id.long().cpu()   # [N, L]
    item_id = item_id.long().cpu()           # [N]

    if semantic_id.ndim != 2:
        raise ValueError(f"semantic_id must be [N, L], got {tuple(semantic_id.shape)}")
    if item_id.ndim != 1:
        raise ValueError(f"item_id must be [N], got {tuple(item_id.shape)}")
    if semantic_id.shape[0] != item_id.shape[0]:
        raise ValueError("semantic_id and item_id length mismatch")

    n, l = semantic_id.shape
    print(f"N={n}, original L={l}")

    # sort by item_id
    order = torch.argsort(item_id)
    item_id_sorted = item_id[order]
    semantic_id_sorted = semantic_id[order]

    # check uniqueness
    unique_item_id = torch.unique(item_id_sorted)
    if unique_item_id.numel() != item_id_sorted.numel():
        raise ValueError("item_id contains duplicates")

    # add one deduplication layer
    dedup_suffix = build_dedup_suffix(semantic_id_sorted)          # [N]
    semantic_id_dedup = torch.cat(
        [semantic_id_sorted, dedup_suffix.unsqueeze(1)], dim=1
    )  # [N, L+1]

    new_l = semantic_id_dedup.shape[1]
    print(f"new L={new_l}")

    # check whether item ids are contiguous from 0
    expected = torch.arange(item_id_sorted.numel(), dtype=item_id_sorted.dtype)
    if not torch.equal(item_id_sorted, expected):
        print("item_id is not contiguous 0..N-1")
        print(f"min item_id = {item_id_sorted.min().item()}")
        print(f"max item_id = {item_id_sorted.max().item()}")
        print("Creating dense tensor indexed by raw item_id ...")

        num_items = int(item_id_sorted.max().item()) + 1
        dense = torch.full((num_items, new_l), -1, dtype=torch.long)
        dense[item_id_sorted] = semantic_id_dedup

        missing_rows = (dense[:, 0] == -1).sum().item()
        print(f"dense shape = {tuple(dense.shape)}, missing rows = {missing_rows}")

        output_tensor = dense
    else:
        output_tensor = semantic_id_dedup

    os.makedirs(os.path.dirname(OUTPUT_TENSOR_PATH), exist_ok=True)
    torch.save(output_tensor.t(), OUTPUT_TENSOR_PATH)  # downstream require [L, N] codebook

    print(f"Saved tensor to: {OUTPUT_TENSOR_PATH}")
    print(f"output shape = {tuple(output_tensor.shape)}")
    print("first 5 rows:")
    print(output_tensor[:5])


if __name__ == "__main__":
    main()