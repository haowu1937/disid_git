import argparse
import csv
import json
import os
from typing import Dict, List, Tuple, Optional

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 改成你的实际 import 路径
from src.data_loading.newsfeed_item_dataset import NewsFeedItemDataset
from src.data_loading.multi_domain_dataset import MultiDomainItemDataset


def l2_normalize_inplace(x: np.ndarray) -> None:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    x /= norms


def collate_with_docid(batch):
    """
    batch element:
        (embed, domain_id, docid)
    returns:
        embeds: np.ndarray [B, D]
        docids: List[str]
    """
    embeds = np.stack([b[0] for b in batch], axis=0).astype(np.float32, copy=False)
    docids = [b[2] for b in batch]
    return embeds, docids


def load_embeddings_from_dataset(
    dataset: NewsFeedItemDataset,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
        embeddings: [N, D] float32
        docids: [N] object
    """
    n = len(dataset)
    d = dataset.expected_embed_dim

    # 优先直接取内存里的数据，最快
    if dataset.load_all_to_memory and dataset._in_memory_embeddings is not None:
        if dataset._in_memory_docids is None:
            raise ValueError("Dataset must be created with return_item_id=True")
        embeddings = dataset._in_memory_embeddings.astype(np.float32, copy=False)
        docids = dataset._in_memory_docids.astype(object, copy=False)
        return embeddings, docids

    # 否则走 DataLoader 顺序读
    embeddings = np.empty((n, d), dtype=np.float32)
    docids = np.empty(n, dtype=object)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_with_docid,
        drop_last=False,
    )

    pos = 0
    for embeds_batch, docids_batch in tqdm(loader, desc="Loading from DataLoader"):
        bsz = embeds_batch.shape[0]
        embeddings[pos:pos + bsz] = embeds_batch
        docids[pos:pos + bsz] = np.asarray(docids_batch, dtype=object)
        pos += bsz

    if pos != n:
        raise RuntimeError(f"Loaded rows mismatch: expected={n}, got={pos}")

    return embeddings, docids


def build_faiss_index(
    vecs: np.ndarray,
    nlist: int,
    nprobe: int,
    train_size: int,
    add_batch_size: int,
    seed: int,
):
    n, dim = vecs.shape

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    rng = np.random.default_rng(seed)
    train_size = min(train_size, n)

    if train_size < n:
        train_idx = rng.choice(n, size=train_size, replace=False)
        train_vecs = vecs[train_idx]
    else:
        train_vecs = vecs

    print(f"[FAISS] training with train_size={train_size}, nlist={nlist}")
    index.train(train_vecs)
    index.nprobe = nprobe

    print(f"[FAISS] adding {n} vectors")
    for start in tqdm(range(0, n, add_batch_size), desc="Adding vectors"):
        end = min(start + add_batch_size, n)
        index.add(vecs[start:end])

    return index


def compute_collision_rates(
    index,
    vecs: np.ndarray,
    docids: np.ndarray,
    thresholds: List[float],
    top_k: int,
    query_batch_size: int,
    max_examples_per_threshold: int,
) -> Tuple[Dict[float, float], Dict[float, List[dict]]]:
    n = vecs.shape[0]
    thresholds = sorted(thresholds, reverse=True)

    collision_counts = {thr: 0 for thr in thresholds}
    examples = {thr: [] for thr in thresholds}

    for start in tqdm(range(0, n, query_batch_size), desc="Searching ANN"):
        end = min(start + query_batch_size, n)
        q = vecs[start:end]

        sims, nbrs = index.search(q, top_k + 1)  # 一般会把自己也搜出来

        row_ids = np.arange(start, end)[:, None]
        valid = (nbrs != -1) & (nbrs != row_ids)

        for thr in thresholds:
            hit = valid & (sims >= thr)
            hit_any = hit.any(axis=1)
            collision_counts[thr] += int(hit_any.sum())

            if len(examples[thr]) < max_examples_per_threshold:
                rows = np.where(hit_any)[0]
                remain = max_examples_per_threshold - len(examples[thr])

                for r in rows[:remain]:
                    good = np.where(hit[r])[0]
                    best_idx = good[np.argmax(sims[r, good])]

                    global_i = start + r
                    global_j = int(nbrs[r, best_idx])

                    examples[thr].append(
                        {
                            "query_row": int(global_i),
                            "query_docid": str(docids[global_i]),
                            "neighbor_row": global_j,
                            "neighbor_docid": str(docids[global_j]),
                            "cosine": float(sims[r, best_idx]),
                        }
                    )

    rates = {thr: collision_counts[thr] / n for thr in thresholds}
    return rates, examples


def save_examples(examples: Dict[float, List[dict]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    for thr, rows in examples.items():
        out_path = os.path.join(out_dir, f"collision_examples_thr_{str(thr).replace('.', '_')}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "query_row",
                    "query_docid",
                    "neighbor_row",
                    "neighbor_docid",
                    "cosine",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("--expected_embed_dim", type=int, default=128)
    parser.add_argument("--load_all_to_memory", action="store_true")
    parser.add_argument("--loader_batch_size", type=int, default=100000)
    parser.add_argument("--loader_num_workers", type=int, default=8)

    # parquet shard args
    parser.add_argument("--parquets", nargs="+", required=True)

    # faiss args
    parser.add_argument("--nlist", type=int, default=16384)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--train_size", type=int, default=500000)
    parser.add_argument("--add_batch_size", type=int, default=200000)
    parser.add_argument("--query_batch_size", type=int, default=50000)
    parser.add_argument("--top_k", type=int, default=20)

    # collision args
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.999, 0.995, 0.99])
    parser.add_argument("--max_examples_per_threshold", type=int, default=200)

    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./collision_check_output")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    domain_configs = []
    for i, p in enumerate(args.parquets, start=1):
        domain_configs.append(
            {
                "name": f"news_part_{i}",
                "parquet_path": p,
            }
        )

    print("[1/4] Building dataset")
    dataset = NewsFeedItemDataset(
        domain_configs=domain_configs,
        expected_embed_dim=args.expected_embed_dim,
        return_item_id=True,
        load_all_to_memory=args.load_all_to_memory,
    )

    print("[2/4] Loading embeddings/docids from dataset")
    vecs, docids = load_embeddings_from_dataset(
        dataset=dataset,
        batch_size=args.loader_batch_size,
        num_workers=args.loader_num_workers,
    )
    print(f"Loaded embeddings shape: {vecs.shape}")
    print(f"Loaded docids shape: {docids.shape}")

    print("[3/4] L2 normalizing")
    l2_normalize_inplace(vecs)

    print("[4/4] Building FAISS index and computing collision rate")
    index = build_faiss_index(
        vecs=vecs,
        nlist=args.nlist,
        nprobe=args.nprobe,
        train_size=args.train_size,
        add_batch_size=args.add_batch_size,
        seed=args.seed,
    )

    rates, examples = compute_collision_rates(
        index=index,
        vecs=vecs,
        docids=docids,
        thresholds=args.thresholds,
        top_k=args.top_k,
        query_batch_size=args.query_batch_size,
        max_examples_per_threshold=args.max_examples_per_threshold,
    )

    save_examples(examples, args.out_dir)

    summary = {
        "num_items": int(vecs.shape[0]),
        "embed_dim": int(vecs.shape[1]),
        "thresholds": args.thresholds,
        "collision_rates": rates,
        "faiss_config": {
            "nlist": args.nlist,
            "nprobe": args.nprobe,
            "top_k": args.top_k,
            "train_size": args.train_size,
            "add_batch_size": args.add_batch_size,
            "query_batch_size": args.query_batch_size,
        },
        "dataset_config": {
            "expected_embed_dim": args.expected_embed_dim,
            "load_all_to_memory": args.load_all_to_memory,
            "loader_batch_size": args.loader_batch_size,
            "loader_num_workers": args.loader_num_workers,
        },
    }

    summary_path = os.path.join(args.out_dir, "collision_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Collision Summary ===")
    for thr in args.thresholds:
        print(f"threshold={thr:.6f} | item_collision_rate={rates[thr]:.6f}")

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved examples to: {args.out_dir}")


if __name__ == "__main__":
    main()