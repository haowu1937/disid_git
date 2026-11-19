import argparse
import os
import random
import time
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data_loading.semantic_dataset import (
    NextItemSemanticDataset,
    SeqDataset,
    SlidingWindowNextItemSemanticDataset,
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


def ensure_dir(path: int | str) -> None:
    os.makedirs(path, exist_ok=True)


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


def build_train_dataloader(
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


def run_one_epoch(
    model: SemanticT5ForNextItem,
    dataloader: DataLoader,
    optimizer,
    device: torch.device,
    sid_width: int,
    train: bool,
    grad_clip_norm: float,
    log_every_steps: int,
    epoch_idx: int,
    split_name: str,
) -> Tuple[float, Dict[str, float]]:
    if train:
        model.train()
    else:
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

        if train:
            optimizer.zero_grad(set_to_none=True)

        compute_start = time.perf_counter()
        with torch.set_grad_enabled(train):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sid_width=sid_width,
                labels=labels,
            )
            loss = outputs["loss"]

            if train:
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        sync_if_needed(device)
        compute_time += time.perf_counter() - compute_start

        batch_size = input_ids.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        prev_end_time = time.perf_counter()

        if step_idx % log_every_steps == 0:
            avg_so_far = total_loss / max(total_count, 1)
            print(
                f"[{split_name}] epoch={epoch_idx} step={step_idx}/{num_steps} "
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


def save_checkpoint(
    save_path: str,
    model: SemanticT5ForNextItem,
    optimizer,
    epoch: int,
    best_val_ndcg10: float,
    best_epoch: int,
    no_improve_epochs: int,
    cfg: Dict,
    metadata: Dict,
) -> None:
    ensure_dir(os.path.dirname(save_path))
    torch.save(
        {
            "epoch": epoch,
            "best_val_ndcg10": best_val_ndcg10,
            "best_epoch": best_epoch,
            "no_improve_epochs": no_improve_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "metadata": metadata,
        },
        save_path,
    )


def load_resume_state(
    ckpt_path: str,
    model: SemanticT5ForNextItem,
    optimizer,
    device: torch.device,
    resume_mode: str,
) -> Tuple[int, float, int, int]:
    """
    Return:
        start_epoch, best_val_ndcg10, best_epoch, no_improve_epochs

    resume_mode:
        - strict_resume
        - finetune_resume
        - none
    """
    if resume_mode == "none":
        return 1, float("-inf"), -1, 0

    if not ckpt_path:
        raise ValueError(
            f"resume_mode={resume_mode} but checkpoint.resume_from is empty."
        )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded model weights from: {ckpt_path}")

    if resume_mode == "strict_resume":
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print(f"Loaded optimizer state from: {ckpt_path}")

        last_epoch = int(ckpt.get("epoch", 0))
        best_val_ndcg10 = float(ckpt.get("best_val_ndcg10", float("-inf")))
        best_epoch = int(ckpt.get("best_epoch", -1))
        no_improve_epochs = int(ckpt.get("no_improve_epochs", 0))
        start_epoch = last_epoch + 1

        print(
            f"Strict resume state: "
            f"last_epoch={last_epoch}, "
            f"start_epoch={start_epoch}, "
            f"best_val_ndcg10={best_val_ndcg10:.6f}, "
            f"best_epoch={best_epoch}, "
            f"no_improve_epochs={no_improve_epochs}"
        )
        return start_epoch, best_val_ndcg10, best_epoch, no_improve_epochs

    if resume_mode == "finetune_resume":
        print(
            "Finetune resume mode: only model weights are loaded. "
            "Optimizer state and early-stopping state are reset."
        )
        return 1, float("-inf"), -1, 0

    raise ValueError(
        f"Unsupported checkpoint.resume_mode={resume_mode}. "
        f"Expected one of: ['none', 'strict_resume', 'finetune_resume']"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to yaml config.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))

    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    optim_cfg = cfg["optimizer"]
    ckpt_cfg = cfg["checkpoint"]
    eval_cfg = cfg["evaluation"]

    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    train_base = build_base_dataset(cfg, "train")
    val_base = build_base_dataset(cfg, "val")
    test_base = build_base_dataset(cfg, "test")

    train_dataset = SlidingWindowNextItemSemanticDataset(
        base_dataset=train_base,
        min_history_len=int(train_cfg.get("min_history_len", 1)),
        max_history_len=train_cfg.get("max_history_len", None),
    )

    val_dataset_for_loss = NextItemSemanticDataset(val_base)
    test_dataset_for_loss = NextItemSemanticDataset(test_base)

    print("Train base summary:", train_base.summary())
    print("Val base summary  :", val_base.summary())
    print("Test base summary :", test_base.summary())
    print(
        f"Next-item samples -> "
        f"train(sliding-window)={len(train_dataset)}, "
        f"val(last-item)={len(val_dataset_for_loss)}, "
        f"test(last-item)={len(test_dataset_for_loss)}"
    )

    sid_width = train_base.sid_width
    pad_token_id = train_base.get_pad_token_id()
    bos_token_id = pad_token_id + 1
    total_vocab_size = train_base.sid_vocab_size + 2

    if val_base.sid_width != sid_width or test_base.sid_width != sid_width:
        raise ValueError("sid_width mismatch across train/val/test datasets.")
    if (
        val_base.sid_vocab_size != train_base.sid_vocab_size
        or test_base.sid_vocab_size != train_base.sid_vocab_size
    ):
        raise ValueError("sid_vocab_size mismatch across train/val/test datasets.")

    train_loader = build_train_dataloader(
        dataset=train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pad_token_id=pad_token_id,
    )
    val_loss_loader = build_train_dataloader(
        dataset=val_dataset_for_loss,
        batch_size=int(train_cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pad_token_id=pad_token_id,
    )
    test_loss_loader = build_train_dataloader(
        dataset=test_dataset_for_loss,
        batch_size=int(train_cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pad_token_id=pad_token_id,
    )

    val_eval_dataset = build_eval_dataset(cfg, "val")
    test_eval_dataset = build_eval_dataset(cfg, "test")

    val_eval_loader = build_eval_dataloader(
        dataset=val_eval_dataset,
        batch_size=int(eval_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 0)),
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
            f"sid_width mismatch between codebook mapping and train dataset: "
            f"{sid_width_map} vs {sid_width}"
        )

    print(f"Eval samples -> val={len(val_eval_dataset)}, test={len(test_eval_dataset)}")
    print(
        f"Codebook stats -> sid_width={sid_width_map}, "
        f"sid_vocab_size={sid_vocab_size}, unique_sids={len(sid_to_items)}"
    )

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

    optimizer = AdamW(
        model.parameters(),
        lr=float(optim_cfg["lr"]),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
        eps=float(optim_cfg.get("eps", 1e-8)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )

    output_dir = ckpt_cfg["output_dir"]
    ensure_dir(output_dir)

    best_ckpt_path = ckpt_cfg.get("best_ckpt_path", os.path.join(output_dir, "best.pt"))
    last_ckpt_path = ckpt_cfg.get("last_ckpt_path", os.path.join(output_dir, "last.pt"))

    num_epochs = int(train_cfg["num_epochs"])
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    log_every_steps = int(train_cfg.get("log_every_steps", 100))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 5))

    beam_size = int(eval_cfg.get("beam_size", 20))
    num_return_sequences = int(eval_cfg.get("num_return_sequences", beam_size))
    metrics_ks = [10]

    eval_every_n = int(eval_cfg.get("eval_every_n", 1))
    if eval_every_n <= 0:
        raise ValueError(f"evaluation.eval_every_n must be >= 1, got {eval_every_n}")

    val_max_eval_batches = eval_cfg.get("val_max_eval_batches", None)
    test_max_eval_batches = eval_cfg.get("test_max_eval_batches", None)
    if val_max_eval_batches is not None:
        val_max_eval_batches = int(val_max_eval_batches)
    if test_max_eval_batches is not None:
        test_max_eval_batches = int(test_max_eval_batches)

    resume_mode = ckpt_cfg.get("resume_mode", "none")
    resume_from = ckpt_cfg.get("resume_from", None)

    start_epoch, best_val_ndcg10, best_epoch, no_improve_epochs = load_resume_state(
        ckpt_path=resume_from,
        model=model,
        optimizer=optimizer,
        device=device,
        resume_mode=resume_mode,
    )

    print(
        f"Training state before loop: "
        f"start_epoch={start_epoch}, "
        f"best_val_ndcg10={best_val_ndcg10:.6f}, "
        f"best_epoch={best_epoch}, "
        f"no_improve_epochs={no_improve_epochs}"
    )

    if start_epoch > num_epochs:
        print(
            f"start_epoch={start_epoch} is greater than num_epochs={num_epochs}. "
            f"Skipping training loop and going directly to final test."
        )

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_wall_start = time.perf_counter()

        train_loss, train_timing = run_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            sid_width=sid_width,
            train=True,
            grad_clip_norm=grad_clip_norm,
            log_every_steps=log_every_steps,
            epoch_idx=epoch,
            split_name="train",
        )

        val_loss, val_loss_timing = run_one_epoch(
            model=model,
            dataloader=val_loss_loader,
            optimizer=optimizer,
            device=device,
            sid_width=sid_width,
            train=False,
            grad_clip_norm=grad_clip_norm,
            log_every_steps=log_every_steps,
            epoch_idx=epoch,
            split_name="val-loss",
        )

        do_eval = (epoch % eval_every_n == 0)

        val_metrics = None
        val_num_samples = 0
        val_eval_timing = None
        val_recall10 = None
        val_ndcg10 = None

        if do_eval:
            val_metrics, val_num_samples, val_eval_timing = evaluate_ranking_metrics_batch(
                model=model,
                dataloader=val_eval_loader,
                sid_to_items=sid_to_items,
                trie=trie,
                sid_width=sid_width,
                beam_size=beam_size,
                num_return_sequences=num_return_sequences,
                metrics_ks=metrics_ks,
                device=device,
                max_eval_batches=val_max_eval_batches,
            )

            val_recall10 = val_metrics["recall@10"]
            val_ndcg10 = val_metrics["ndcg@10"]

        epoch_wall_time = time.perf_counter() - epoch_wall_start

        if do_eval:
            print(
                f"[epoch {epoch}] "
                f"train_loss={train_loss:.6f} "
                f"val_loss(full)={val_loss:.6f} "
                f"val_recall@10(top-{val_num_samples})={val_recall10:.6f} "
                f"val_ndcg@10(top-{val_num_samples})={val_ndcg10:.6f}"
            )
            print(
                f"[epoch {epoch} timing] total={epoch_wall_time:.2f}s | "
                f"train: total={train_timing['epoch_total_time']:.2f}s "
                f"(data={train_timing['data_wait_time']:.2f}s, move={train_timing['move_time']:.2f}s, compute={train_timing['compute_time']:.2f}s) | "
                f"val-loss: total={val_loss_timing['epoch_total_time']:.2f}s "
                f"(data={val_loss_timing['data_wait_time']:.2f}s, move={val_loss_timing['move_time']:.2f}s, compute={val_loss_timing['compute_time']:.2f}s) | "
                f"val-eval: total={val_eval_timing['eval_total_time']:.2f}s "
                f"(data={val_eval_timing['data_wait_time']:.2f}s, move={val_eval_timing['move_time']:.2f}s, "
                f"beam={val_eval_timing['beam_search_time']:.2f}s, rank={val_eval_timing['ranking_time']:.2f}s)"
            )
        else:
            print(
                f"[epoch {epoch}] "
                f"train_loss={train_loss:.6f} "
                f"val_loss(full)={val_loss:.6f} "
                f"val_eval=skipped (eval_every_n={eval_every_n})"
            )
            print(
                f"[epoch {epoch} timing] total={epoch_wall_time:.2f}s | "
                f"train: total={train_timing['epoch_total_time']:.2f}s "
                f"(data={train_timing['data_wait_time']:.2f}s, move={train_timing['move_time']:.2f}s, compute={train_timing['compute_time']:.2f}s) | "
                f"val-loss: total={val_loss_timing['epoch_total_time']:.2f}s "
                f"(data={val_loss_timing['data_wait_time']:.2f}s, move={val_loss_timing['move_time']:.2f}s, compute={val_loss_timing['compute_time']:.2f}s)"
            )

        metadata = {
            "sid_width": sid_width,
            "pad_token_id": pad_token_id,
            "bos_token_id": bos_token_id,
            "base_sid_vocab_size": train_base.sid_vocab_size,
            "total_vocab_size": total_vocab_size,
            "resume_mode": resume_mode,
        }

        save_checkpoint(
            save_path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_ndcg10=best_val_ndcg10,
            best_epoch=best_epoch,
            no_improve_epochs=no_improve_epochs,
            cfg=cfg,
            metadata=metadata,
        )

        if do_eval:
            if val_ndcg10 > best_val_ndcg10:
                best_val_ndcg10 = val_ndcg10
                best_epoch = epoch
                no_improve_epochs = 0

                save_checkpoint(
                    save_path=best_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_ndcg10=best_val_ndcg10,
                    best_epoch=best_epoch,
                    no_improve_epochs=no_improve_epochs,
                    cfg=cfg,
                    metadata=metadata,
                )
                print(
                    f"Saved new best checkpoint to: {best_ckpt_path} "
                    f"(best val_ndcg@10={best_val_ndcg10:.6f}, eval_samples={val_num_samples})"
                )
            else:
                no_improve_epochs += 1
                print(
                    f"No improvement in val_ndcg@10 for {no_improve_epochs} evaluation(s). "
                    f"Best so far: {best_val_ndcg10:.6f} at epoch {best_epoch}"
                )

            if no_improve_epochs >= early_stopping_patience:
                print(
                    f"Early stopping triggered: "
                    f"no val_ndcg@10 improvement for {early_stopping_patience} consecutive evaluation(s)."
                )
                break

    print(f"Training finished. best_epoch={best_epoch}, best_val_ndcg@10={best_val_ndcg10:.6f}")

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint from: {best_ckpt_path}")

    test_loss, test_loss_timing = run_one_epoch(
        model=model,
        dataloader=test_loss_loader,
        optimizer=optimizer,
        device=device,
        sid_width=sid_width,
        train=False,
        grad_clip_norm=grad_clip_norm,
        log_every_steps=log_every_steps,
        epoch_idx=best_epoch if best_epoch > 0 else 0,
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
        f"(data={test_loss_timing['data_wait_time']:.2f}s, move={test_loss_timing['move_time']:.2f}s, compute={test_loss_timing['compute_time']:.2f}s) | "
        f"test-eval: total={test_eval_timing['eval_total_time']:.2f}s "
        f"(data={test_eval_timing['data_wait_time']:.2f}s, move={test_eval_timing['move_time']:.2f}s, "
        f"beam={test_eval_timing['beam_search_time']:.2f}s, rank={test_eval_timing['ranking_time']:.2f}s)"
    )


if __name__ == "__main__":
    main()