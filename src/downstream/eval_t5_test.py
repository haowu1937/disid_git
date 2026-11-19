import argparse
import os
import random
import time
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data_loading.semantic_dataset import (
    NextItemSemanticDataset,
    SeqDataset,
    collate_next_item_t5_batch,
)
from src.data_loading.semantic_eval_dataset import (
    NextItemEvalDataset,
    collate_next_item_eval_t5_batch,
)
from src.downstream.eval_utils import (
    beams_to_ranked_items,
    build_sid_mappings,
    constrained_beam_search_batch,
    ndcg_at_k,
    recall_at_k,
)
from src.downstream.semantic_t5 import (
    SemanticT5ForNextItem,
    build_t5_config_from_dict,
)

os.environ["TRANSFORMERS_NO_TF"] = "1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def build_base_dataset(cfg: Dict, split: str) -> SeqDataset:
    data_cfg = cfg["data"]
    split_dir = data_cfg[split]["tfrecord_dir"]

    dataset = SeqDataset(
        tfrecord_dir=split_dir,
        semantic_id_pt_path=data_cfg["semantic_id_pt_path"],
        tfrecord_pattern=data_cfg.get("tfrecord_pattern", "partition_*.tfrecord.gz"),
        skip_unknown_items=bool(data_cfg.get("skip_unknown_items", False)),
        min_seq_len=int(data_cfg.get("min_seq_len", 1)),
        max_seq_len=data_cfg.get("max_seq_len", None),
        drop_last_item_for_target=False,
    )
    return dataset


def build_eval_dataset(cfg: Dict, split: str) -> NextItemEvalDataset:
    data_cfg = cfg["data"]
    split_dir = data_cfg[split]["tfrecord_dir"]

    dataset = NextItemEvalDataset(
        tfrecord_dir=split_dir,
        semantic_id_pt_path=data_cfg["semantic_id_pt_path"],
        tfrecord_pattern=data_cfg.get("tfrecord_pattern", "partition_*.tfrecord.gz"),
        skip_unknown_items=bool(data_cfg.get("skip_unknown_items", False)),
        min_seq_len=int(data_cfg.get("min_seq_len", 2)),
        max_seq_len=data_cfg.get("max_seq_len", None),
    )
    return dataset


def build_loss_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pad_token_id: int,
) -> DataLoader:
    collate_fn = partial(
        collate_next_item_t5_batch,
        pad_token_id=pad_token_id,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )


def build_eval_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pad_token_id: int,
) -> DataLoader:
    collate_fn = partial(
        collate_next_item_eval_t5_batch,
        pad_token_id=pad_token_id,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )


@torch.no_grad()
def run_loss_epoch(
    model: SemanticT5ForNextItem,
    dataloader: DataLoader,
    device: torch.device,
    sid_width: int,
    log_every_steps: int,
    split_name: str,
) -> Tuple[float, Dict[str, float]]:
    model.eval()

    total_loss = 0.0
    total_count = 0

    data_wait_time = 0.0
    move_time = 0.0
    compute_time = 0.0

    epoch_start_time = time.perf_counter()
    prev_end_time = epoch_start_time
    num_steps = len(dataloader)

    for step_idx, batch in enumerate(dataloader, start=1):
        now = time.perf_counter()
        data_wait_time += now - prev_end_time

        move_start = time.perf_counter()
        batch = move_batch_to_device(batch, device)
        sync_if_needed(device)
        move_time += time.perf_counter() - move_start

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["target_semantic_ids"]

        compute_start = time.perf_counter()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sid_width=sid_width,
            labels=labels,
        )
        loss = outputs["loss"]

        sync_if_needed(device)
        compute_time += time.perf_counter() - compute_start

        batch_size = input_ids.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        prev_end_time = time.perf_counter()

        if step_idx % log_every_steps == 0:
            avg_so_far = total_loss / max(total_count, 1)
            print(
                f"[{split_name}] step={step_idx}/{num_steps} "
                f"loss={avg_so_far:.6f}"
            )

    epoch_total_time = time.perf_counter() - epoch_start_time
    timing = {
        "data_wait_time": data_wait_time,
        "move_time": move_time,
        "compute_time": compute_time,
        "epoch_total_time": epoch_total_time,
        "num_steps": float(num_steps),
    }

    return total_loss / max(total_count, 1), timing


@torch.no_grad()
def evaluate_ranking_metrics_batch(
    model: SemanticT5ForNextItem,
    dataloader: DataLoader,
    sid_to_items,
    trie,
    sid_width: int,
    beam_size: int,
    num_return_sequences: int,
    metrics_ks: List[int],
    device: torch.device,
    max_eval_batches: Optional[int] = None,
) -> Tuple[Dict[str, float], int, Dict[str, float]]:
    model.eval()

    max_k = max(metrics_ks)
    sums = {f"recall@{k}": 0.0 for k in metrics_ks}
    sums.update({f"ndcg@{k}": 0.0 for k in metrics_ks})

    num_samples = 0
    num_batches = 0

    data_wait_time = 0.0
    move_time = 0.0
    beam_search_time = 0.0
    ranking_time = 0.0

    eval_start_time = time.perf_counter()
    prev_end_time = eval_start_time

    for batch_idx, batch in enumerate(dataloader):
        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break

        now = time.perf_counter()
        data_wait_time += now - prev_end_time

        move_start = time.perf_counter()
        batch = move_batch_to_device(batch, device)
        sync_if_needed(device)
        move_time += time.perf_counter() - move_start

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target_item_ids = batch["target_item_ids"]
        batch_size = input_ids.size(0)
        num_batches += 1

        beam_start = time.perf_counter()
        batch_beams = constrained_beam_search_batch(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            sid_width=sid_width,
            trie=trie,
            beam_size=beam_size,
            num_return_sequences=num_return_sequences,
        )
        sync_if_needed(device)
        beam_search_time += time.perf_counter() - beam_start

        ranking_start = time.perf_counter()
        for i in range(batch_size):
            beams = batch_beams[i]
            target_item_id = int(target_item_ids[i].item())

            ranked_items = beams_to_ranked_items(
                beams=beams,
                sid_to_items=sid_to_items,
                max_items=max_k,
            )

            for k in metrics_ks:
                sums[f"recall@{k}"] += recall_at_k(ranked_items, target_item_id, k)
                sums[f"ndcg@{k}"] += ndcg_at_k(ranked_items, target_item_id, k)

            num_samples += 1

        ranking_time += time.perf_counter() - ranking_start
        prev_end_time = time.perf_counter()

    metrics = {}
    for k in metrics_ks:
        metrics[f"recall@{k}"] = sums[f"recall@{k}"] / max(num_samples, 1)
        metrics[f"ndcg@{k}"] = sums[f"ndcg@{k}"] / max(num_samples, 1)

    eval_total_time = time.perf_counter() - eval_start_time
    timing = {
        "data_wait_time": data_wait_time,
        "move_time": move_time,
        "beam_search_time": beam_search_time,
        "ranking_time": ranking_time,
        "eval_total_time": eval_total_time,
        "num_batches": float(num_batches),
        "num_samples": float(num_samples),
    }

    return metrics, num_samples, timing


def resolve_checkpoint_path(cfg: Dict, ckpt_arg: Optional[str]) -> str:
    if ckpt_arg is not None:
        return ckpt_arg

    ckpt_cfg = cfg["checkpoint"]
    best_ckpt_path = ckpt_cfg.get("best_ckpt_path", None)
    last_ckpt_path = ckpt_cfg.get("last_ckpt_path", None)

    if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
        return best_ckpt_path
    if last_ckpt_path is not None and os.path.exists(last_ckpt_path):
        return last_ckpt_path

    candidates = [p for p in [best_ckpt_path, last_ckpt_path] if p is not None]
    raise FileNotFoundError(
        "No checkpoint found. Checked: " + ", ".join(candidates)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional checkpoint path. If not set, prefer checkpoint.best_ckpt_path, then checkpoint.last_ckpt_path.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))

    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["evaluation"]

    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # -------------------------
    # Build datasets
    # -------------------------
    train_base = build_base_dataset(cfg, "train")
    test_base = build_base_dataset(cfg, "test")

    test_dataset_for_loss = NextItemSemanticDataset(test_base)
    test_eval_dataset = build_eval_dataset(cfg, "test")

    print("Train base summary:", train_base.summary())
    print("Test base summary :", test_base.summary())
    print(
        f"Test samples -> "
        f"test(last-item loss)={len(test_dataset_for_loss)}, "
        f"test(eval)={len(test_eval_dataset)}"
    )

    sid_width = train_base.sid_width
    pad_token_id = train_base.get_pad_token_id()
    bos_token_id = pad_token_id + 1
    total_vocab_size = train_base.sid_vocab_size + 2

    if test_base.sid_width != sid_width:
        raise ValueError("sid_width mismatch across train/test datasets.")
    if test_base.sid_vocab_size != train_base.sid_vocab_size:
        raise ValueError("sid_vocab_size mismatch across train/test datasets.")

    test_loss_loader = build_loss_dataloader(
        dataset=test_dataset_for_loss,
        batch_size=int(train_cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pad_token_id=pad_token_id,
    )

    test_eval_loader = build_eval_dataloader(
        dataset=test_eval_dataset,
        batch_size=int(eval_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 0)),
        pad_token_id=pad_token_id,
    )

    sid_to_items, trie, sid_width_map, sid_vocab_size = build_sid_mappings(
        cfg["data"]["semantic_id_pt_path"]
    )
    if sid_width_map != sid_width:
        raise ValueError(
            f"sid_width mismatch between codebook mapping and dataset: "
            f"{sid_width_map} vs {sid_width}"
        )

    print(
        f"Codebook stats -> sid_width={sid_width_map}, "
        f"sid_vocab_size={sid_vocab_size}, unique_sids={len(sid_to_items)}"
    )

    # -------------------------
    # Build model
    # -------------------------
    t5_config = build_t5_config_from_dict(
        model_cfg=model_cfg,
        total_vocab_size=total_vocab_size,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
    )

    model = SemanticT5ForNextItem(
        config=t5_config,
        sid_num_layers=sid_width,
        sid_vocab_size=total_vocab_size,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: total={num_params:,} trainable={num_trainable_params:,}")
    print(f"Special tokens: pad_token_id={pad_token_id}, bos_token_id={bos_token_id}")
    print(
        f"Semantic config: sid_width={sid_width}, "
        f"base_sid_vocab_size={train_base.sid_vocab_size}, "
        f"total_vocab_size={total_vocab_size}"
    )

    # -------------------------
    # Load checkpoint
    # -------------------------
    ckpt_path = resolve_checkpoint_path(cfg, args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from: {ckpt_path}")

    if "epoch" in ckpt:
        print(f"Checkpoint epoch: {ckpt['epoch']}")
    if "best_val_ndcg10" in ckpt:
        print(f"Checkpoint best_val_ndcg10: {ckpt['best_val_ndcg10']:.6f}")

    # -------------------------
    # Test
    # -------------------------
    log_every_steps = int(train_cfg.get("log_every_steps", 100))
    beam_size = int(eval_cfg.get("beam_size", 20))
    num_return_sequences = int(eval_cfg.get("num_return_sequences", beam_size))
    metrics_ks = [10]

    test_max_eval_batches = eval_cfg.get("test_max_eval_batches", None)
    if test_max_eval_batches is not None:
        test_max_eval_batches = int(test_max_eval_batches)

    test_loss, test_loss_timing = run_loss_epoch(
        model=model,
        dataloader=test_loss_loader,
        device=device,
        sid_width=sid_width,
        log_every_steps=log_every_steps,
        split_name="test-loss",
    )

    test_metrics, test_num_samples, test_eval_timing = evaluate_ranking_metrics_batch(
        model=model,
        dataloader=test_eval_loader,
        sid_to_items=sid_to_items,
        trie=trie,
        sid_width=sid_width,
        beam_size=beam_size,
        num_return_sequences=num_return_sequences,
        metrics_ks=metrics_ks,
        device=device,
        max_eval_batches=test_max_eval_batches,
    )

    print(
        f"[test] "
        f"loss(full)={test_loss:.6f} "
        f"recall@10(top-{test_num_samples})={test_metrics['recall@10']:.6f} "
        f"ndcg@10(top-{test_num_samples})={test_metrics['ndcg@10']:.6f}"
    )

    print(
        f"[test timing] "
        f"test-loss: total={test_loss_timing['epoch_total_time']:.2f}s "
        f"(data={test_loss_timing['data_wait_time']:.2f}s, "
        f"move={test_loss_timing['move_time']:.2f}s, "
        f"compute={test_loss_timing['compute_time']:.2f}s) | "
        f"test-eval: total={test_eval_timing['eval_total_time']:.2f}s "
        f"(data={test_eval_timing['data_wait_time']:.2f}s, "
        f"move={test_eval_timing['move_time']:.2f}s, "
        f"beam={test_eval_timing['beam_search_time']:.2f}s, "
        f"rank={test_eval_timing['ranking_time']:.2f}s)"
    )


if __name__ == "__main__":
    main()