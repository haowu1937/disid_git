# generate_semantic_ids.py
# ------------------------------------------------------------
# Load a trained RQ-VAE checkpoint, run inference on all items,
# and save semantic IDs (code indices) to disk.
#
# This script is designed to match the training style in train_rqvae.py:
# - uses MultiDomainItemDataset
# - uses build_model(config, device)
# - supports baseline (forward(x)), rqkmeans (forward(x)),
#   and hierarchical/disentangle (forward(x, d))
#
# Output (default):
#   out_dir/
#     semantic_ids_all.pt                 # all items (across domains)
#     semantic_ids_domain_<id>.pt         # per-domain split
#     sid_{model_type}.pt                    # GRID format pt file with all items
# ------------------------------------------------------------

import os
import argparse
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

# Adjust these imports to match your repo layout
# (based on your train_rqvae.py snippet)
from src.data_loading.bakcup_multi_domain_dataset import MultiDomainItemDataset
from src.models.utils import build_model


def _load_checkpoint_to_model(ckpt_path: str, model: torch.nn.Module) -> Dict[str, Any]:
    """
    Loads checkpoint from ckpt_path and loads state_dict into model.

    We try to be robust to different checkpoint key styles.
    Returns the loaded checkpoint dict (so you can inspect config/epoch if needed).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Common patterns:
    # 1) {"model": state_dict, "optimizer": ..., "epoch": ..., "config": ...}
    # 2) {"model_state_dict": state_dict, ...}
    # 3) direct state_dict
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        else:
            # might already be a state_dict-like dict
            state = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint: {missing[:20]}{'...' if len(missing) > 20 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint: {unexpected[:20]}{'...' if len(unexpected) > 20 else ''}")

    return ckpt if isinstance(ckpt, dict) else {"state_dict": state}


def build_dedup_suffix(semantic_id_sorted: torch.Tensor) -> torch.Tensor:
    """
    For each original semantic_id row [L], append one deduplication suffix.

    Current behavior follows the previous convert_sid_to_GRID.py:
      - unique semantic id -> suffix 1
      - collision group -> suffix 1..K, assigned in sorted order

    Input:
        semantic_id_sorted: [N, L]
    Output:
        dedup_suffix: [N]
    """
    n = semantic_id_sorted.shape[0]

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
            # keep the same behavior as the old script
            dedup_suffix[indices[0]] = 1

    print(f"[GRID] collision groups = {collision_group_count}")
    print(f"[GRID] collision items = {collision_item_count}")
    print(f"[GRID] max collision size = {max_collision_size}")

    return dedup_suffix


def save_grid_format_sid(
    semantic_id: torch.Tensor,
    item_id: torch.Tensor,
    output_path: str,
) -> None:
    """
    Convert semantic_ids_all.pt content to GRID format.

    Input:
        semantic_id: [N, L]
        item_id: [N]

    Output file:
        output_path stores tensor with shape [L + 1, N]
        because downstream GRID requires [codebook_layers, num_items].
    """
    if not isinstance(semantic_id, torch.Tensor):
        semantic_id = torch.as_tensor(semantic_id)
    if not isinstance(item_id, torch.Tensor):
        item_id = torch.as_tensor(item_id)

    semantic_id = semantic_id.long().cpu()
    item_id = item_id.long().cpu()

    if semantic_id.ndim != 2:
        raise ValueError(f"semantic_id must be [N, L], got {tuple(semantic_id.shape)}")
    if item_id.ndim != 1:
        raise ValueError(f"item_id must be [N], got {tuple(item_id.shape)}")
    if semantic_id.shape[0] != item_id.shape[0]:
        raise ValueError("semantic_id and item_id length mismatch")

    n, l = semantic_id.shape
    print(f"[GRID] N={n}, original L={l}")

    # sort by item_id
    order = torch.argsort(item_id)
    item_id_sorted = item_id[order]
    semantic_id_sorted = semantic_id[order]

    # check uniqueness
    unique_item_id = torch.unique(item_id_sorted)
    if unique_item_id.numel() != item_id_sorted.numel():
        raise ValueError("item_id contains duplicates")

    # add one deduplication layer
    dedup_suffix = build_dedup_suffix(semantic_id_sorted)  # [N]
    semantic_id_dedup = torch.cat(
        [semantic_id_sorted, dedup_suffix.unsqueeze(1)], dim=1
    )  # [N, L + 1]

    new_l = semantic_id_dedup.shape[1]
    print(f"[GRID] new L={new_l}")

    # check whether item ids are contiguous from 0
    expected = torch.arange(item_id_sorted.numel(), dtype=item_id_sorted.dtype)
    if not torch.equal(item_id_sorted, expected):
        print("[GRID] item_id is not contiguous 0..N-1")
        print(f"[GRID] min item_id = {item_id_sorted.min().item()}")
        print(f"[GRID] max item_id = {item_id_sorted.max().item()}")
        print("[GRID] Creating dense tensor indexed by raw item_id ...")

        num_items = int(item_id_sorted.max().item()) + 1
        dense = torch.full((num_items, new_l), -1, dtype=torch.long)
        dense[item_id_sorted] = semantic_id_dedup

        missing_rows = (dense[:, 0] == -1).sum().item()
        print(f"[GRID] dense shape = {tuple(dense.shape)}, missing rows = {missing_rows}")

        output_tensor = dense
    else:
        output_tensor = semantic_id_dedup

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # downstream requires [L, N] codebook format
    output_tensor_t = output_tensor.t().contiguous()
    torch.save(output_tensor_t, output_path)

    print(f"[GRID] Saved tensor to: {output_path}")
    print(f"[GRID] saved tensor shape = {tuple(output_tensor_t.shape)}")
    print("[GRID] first 5 rows before transpose:")
    print(output_tensor[:5])


@torch.no_grad()
def generate_and_save(
    config: Dict[str, Any],
    ckpt_path: str,
    out_dir: str,
    batch_size: int,
    num_workers: int,
):
    os.makedirs(out_dir, exist_ok=True)

    seed = int(config["generation"].get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data (IMPORTANT: for generation we NEED item_id, so return_item_id=True)
    domain_configs = config["data"]["domain_configs"]
    expected_embed_dim = int(config["data"].get("expected_embed_dim", 768))

    dataset = MultiDomainItemDataset(
        domain_configs,
        expected_embed_dim=expected_embed_dim,
        return_item_id=True,  # <-- difference from training
    )

    pin_memory = bool(config["generation"].get("pin_memory", torch.cuda.is_available()))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # deterministic
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Model
    model = build_model(config, device)
    model.eval()

    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = _load_checkpoint_to_model(ckpt_path, model)

    # Inference loop
    # We'll collect tuples: (domain_id, item_id, indices[num_layers])
    rows: List[Tuple[int, int, torch.Tensor]] = []

    model_type = config["model"]["type"]
    for batch in dataloader:
        # Be robust to dataset return style:
        # - expected: (x, d, item_id)
        # - fallback: (x, d) (then no save)
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, d, item_id = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, d = batch
            item_id = None
        else:
            raise RuntimeError(f"Unexpected batch format, got type={type(batch)} len={len(batch) if hasattr(batch,'__len__') else 'NA'}")

        x = x.to(device, non_blocking=True).float()
        d = d.to(device, non_blocking=True).long()

        # RQVAE baseline and RQKMeans both use forward(x).
        # Disentangle / hierarchical variants use forward(x, d).
        if model_type in ["baseline", "rqkmeans"]:
            _, idx, _ = model(x)  # idx: [B, L]
        else:
            _, idx, _ = model(x, d)  # idx: [B, L]

        idx = idx.detach().cpu()

        if item_id is None:
            raise RuntimeError(
                "Your dataset did not return item_id. "
                "Please set MultiDomainItemDataset(return_item_id=True) and ensure it yields (x, d, item_id)."
            )

        # item_id can be tensor or list
        if torch.is_tensor(item_id):
            item_id_cpu = item_id.detach().cpu().long()
        else:
            item_id_cpu = torch.tensor(item_id, dtype=torch.long)

        d_cpu = d.detach().cpu().long()

        for i in range(idx.size(0)):
            rows.append((int(d_cpu[i].item()), int(item_id_cpu[i]), idx[i].clone()))

    if len(rows) == 0:
        raise RuntimeError("No items found from dataloader; nothing to save.")

    # Sort for deterministic output
    rows.sort(key=lambda t: (t[0], t[1]))

    # Save all in one file
    all_domain = torch.tensor([r[0] for r in rows], dtype=torch.long)
    all_item = torch.tensor([r[1] for r in rows], dtype=torch.long)
    all_idx = torch.stack([r[2] for r in rows], dim=0).long()  # [N, L]

    out_all = {
        "domain_id": all_domain,
        "item_id": all_item,
        "semantic_id": all_idx,  # code indices per layer
        "ckpt_path": ckpt_path,
        "model_type": model_type,
        "num_layers": int(all_idx.size(1)),
        "config": config,  # keep for traceability
        "ckpt_meta": {k: ckpt[k] for k in ckpt.keys() if k in ["epoch", "config"]},
    }
    all_path = os.path.join(out_dir, "semantic_ids_all.pt")
    torch.save(out_all, all_path)
    print(f"Saved: {all_path} (N={all_idx.size(0)}, L={all_idx.size(1)})")

    # Save GRID format sid file
    gen_cfg = config.get("generation", {})

    # Default GRID filename is based on config["model"]["type"], e.g.
    #   model.type = rqkmeans   -> sid_rqkmeans.pt
    #   model.type = baseline   -> sid_baseline.pt
    #   model.type = disentangle -> sid_disentangle.pt
    default_grid_output_name = f"sid_{model_type}.pt"

    # Optional override from config:
    # generation:
    #   grid_output_name: sid_custom_name.pt
    grid_output_name = str(gen_cfg.get("grid_output_name", default_grid_output_name))
    grid_output_path = os.path.join(out_dir, grid_output_name)

    save_grid_format_sid(
        semantic_id=all_idx,
        item_id=all_item,
        output_path=grid_output_path,
    )

    # Save per-domain split
    # Domain ids are assumed 0..num_domains-1
    unique_domains = sorted(set(int(r[0]) for r in rows))
    for dom in unique_domains:
        mask = (all_domain == dom)
        dom_item = all_item[mask]
        dom_idx = all_idx[mask]

        dom_out = {
            "domain_id": dom,
            "item_id": dom_item,
            "semantic_id": dom_idx,
            "ckpt_path": ckpt_path,
            "model_type": model_type,
            "num_layers": int(dom_idx.size(1)),
        }
        dom_path = os.path.join(out_dir, f"semantic_ids_domain_{dom}.pt")
        torch.save(dom_out, dom_path)
        print(f"Saved: {dom_path} (N={dom_idx.size(0)})")


def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="yaml config for generation")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Ensure num_domains is set (same logic as training)
    if "num_domains" not in config["data"]:
        config["data"]["num_domains"] = len(config["data"]["domain_configs"])

    # Read all generation params from config
    gen_cfg = config.get("generation", {})
    if "ckpt_path" not in gen_cfg:
        raise ValueError("Missing config['generation']['ckpt_path']")
    if "out_dir" not in gen_cfg:
        raise ValueError("Missing config['generation']['out_dir']")

    ckpt_path = str(gen_cfg["ckpt_path"])
    out_dir = str(gen_cfg["out_dir"])

    batch_size = int(gen_cfg.get("batch_size", config["generation"]["batch_size"]))
    num_workers = int(gen_cfg.get("num_workers", config["generation"].get("num_workers", 4)))

    generate_and_save(
        config=config,
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()