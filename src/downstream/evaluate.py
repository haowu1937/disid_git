import argparse
import os
from functools import partial
from typing import Dict, List

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_loading.semantic_eval_dataset import (
    AmazonNextItemEvalDataset,
    collate_next_item_eval_t5_batch,
)
from src.downstream.eval_utils import (
    beams_to_ranked_items,
    build_sid_mappings,
    constrained_beam_search_single,
    ndcg_at_k,
    recall_at_k,
)
from src.downstream.semantic_t5 import (
    SemanticT5ForNextItem,
    build_t5_config_from_dict,
)

# Optional but often useful in environments with broken TF installations.
os.environ["TRANSFORMERS_NO_TF"] = "1"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def build_model_from_checkpoint(ckpt: Dict, device: torch.device) -> SemanticT5ForNextItem:
    train_cfg = ckpt["config"]
    metadata = ckpt["metadata"]

    model_cfg = train_cfg["model"]
    total_vocab_size = int(metadata["total_vocab_size"])
    pad_token_id = int(metadata["pad_token_id"])
    bos_token_id = int(metadata["bos_token_id"])
    sid_width = int(metadata["sid_width"])

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
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def evaluate(
    model: SemanticT5ForNextItem,
    dataloader: DataLoader,
    sid_to_items,
    trie,
    sid_width: int,
    beam_size: int,
    num_return_sequences: int,
    metrics_ks: List[int],
    device: torch.device,
):
    max_k = max(metrics_ks)
    sums = {f"recall@{k}": 0.0 for k in metrics_ks}
    sums.update({f"ndcg@{k}": 0.0 for k in metrics_ks})

    num_samples = 0

    for batch in tqdm(dataloader, desc="Evaluating", ncols=100):
        batch = move_batch_to_device(batch, device)

        input_ids = batch["input_ids"]             # [B, S]
        attention_mask = batch["attention_mask"]   # [B, S]
        target_item_ids = batch["target_item_ids"] # [B]

        batch_size = input_ids.size(0)

        for i in range(batch_size):
            single_input_ids = input_ids[i:i + 1]
            single_attention_mask = attention_mask[i:i + 1]
            target_item_id = int(target_item_ids[i].item())

            beams = constrained_beam_search_single(
                model=model,
                input_ids=single_input_ids,
                attention_mask=single_attention_mask,
                sid_width=sid_width,
                trie=trie,
                beam_size=beam_size,
                num_return_sequences=num_return_sequences,
            )

            ranked_items = beams_to_ranked_items(
                beams=beams,
                sid_to_items=sid_to_items,
                max_items=max_k,
            )

            for k in metrics_ks:
                sums[f"recall@{k}"] += recall_at_k(ranked_items, target_item_id, k)
                sums[f"ndcg@{k}"] += ndcg_at_k(ranked_items, target_item_id, k)

            num_samples += 1

    metrics = {}
    for k in metrics_ks:
        metrics[f"recall@{k}"] = sums[f"recall@{k}"] / max(num_samples, 1)
        metrics[f"ndcg@{k}"] = sums[f"ndcg@{k}"] / max(num_samples, 1)

    return metrics, num_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to evaluation yaml config.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    data_cfg = cfg["data"]
    eval_cfg = cfg["evaluation"]
    ckpt_path = cfg["checkpoint"]["path"]

    # ------------------------------------------------------------
    # Load checkpoint and rebuild model
    # ------------------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model_from_checkpoint(ckpt, device=device)

    metadata = ckpt["metadata"]
    pad_token_id = int(metadata["pad_token_id"])
    sid_width_ckpt = int(metadata["sid_width"])

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Checkpoint metadata: sid_width={sid_width_ckpt}, pad_token_id={pad_token_id}")

    # ------------------------------------------------------------
    # Build evaluation dataset / dataloader
    # ------------------------------------------------------------
    test_dataset = AmazonNextItemEvalDataset(
        tfrecord_dir=data_cfg["test"]["tfrecord_dir"],
        semantic_id_pt_path=data_cfg["semantic_id_pt_path"],
        tfrecord_pattern=data_cfg.get("tfrecord_pattern", "partition_*.tfrecord.gz"),
        skip_unknown_items=bool(data_cfg.get("skip_unknown_items", False)),
        min_seq_len=int(data_cfg.get("min_seq_len", 2)),
        max_seq_len=data_cfg.get("max_seq_len", None),
    )

    if test_dataset.sid_width != sid_width_ckpt:
        raise ValueError(
            f"sid_width mismatch between checkpoint and test dataset: "
            f"{sid_width_ckpt} vs {test_dataset.sid_width}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(eval_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 0)),
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(
            collate_next_item_eval_t5_batch,
            pad_token_id=pad_token_id,
        ),
    )

    print(f"Test samples: {len(test_dataset)}")

    # ------------------------------------------------------------
    # Build SID->items mapping and trie for constrained decoding
    # ------------------------------------------------------------
    sid_to_items, trie, sid_width_map, sid_vocab_size = build_sid_mappings(
        data_cfg["semantic_id_pt_path"]
    )
    if sid_width_map != sid_width_ckpt:
        raise ValueError(
            f"sid_width mismatch between checkpoint and codebook mapping: "
            f"{sid_width_ckpt} vs {sid_width_map}"
        )

    print(f"Codebook stats: sid_width={sid_width_map}, sid_vocab_size={sid_vocab_size}, unique_sids={len(sid_to_items)}")

    # ------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------
    beam_size = int(eval_cfg["beam_size"])
    num_return_sequences = int(eval_cfg["num_return_sequences"])
    metrics_ks = [int(x) for x in eval_cfg["metrics_k"]]

    metrics, num_samples = evaluate(
        model=model,
        dataloader=test_loader,
        sid_to_items=sid_to_items,
        trie=trie,
        sid_width=sid_width_ckpt,
        beam_size=beam_size,
        num_return_sequences=num_return_sequences,
        metrics_ks=metrics_ks,
        device=device,
    )

    print(f"Evaluated samples: {num_samples}")
    print("Evaluation metrics:")
    for k in metrics_ks:
        print(
            f"  Recall@{k}: {metrics[f'recall@{k}']:.6f} | "
            f"NDCG@{k}: {metrics[f'ndcg@{k}']:.6f}"
        )

    output_path = eval_cfg.get("output_path", None)
    if output_path is not None:
        ensure_dir(os.path.dirname(output_path))
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "checkpoint": ckpt_path,
                    "num_samples": num_samples,
                    "metrics": metrics,
                    "beam_size": beam_size,
                    "num_return_sequences": num_return_sequences,
                    "metrics_k": metrics_ks,
                },
                f,
                sort_keys=False,
            )
        print(f"Saved evaluation results to: {output_path}")


if __name__ == "__main__":
    main()