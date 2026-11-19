import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import faiss
import numpy as np
from tqdm import tqdm

from src.data_loading.multi_domain_dataset import MultiDomainItemDataset


def l2_normalize_inplace(x: np.ndarray) -> None:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    x /= norms


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


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
    item_ids: np.ndarray,
    domain_ids: np.ndarray,
    domain_names: List[str],
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

        sims, nbrs = index.search(q, top_k + 1)

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

                    query_domain_id = int(domain_ids[global_i])
                    neighbor_domain_id = int(domain_ids[global_j])

                    examples[thr].append(
                        {
                            "query_row": int(global_i),
                            "query_domain": domain_names[query_domain_id],
                            "query_item_id": str(item_ids[global_i]),
                            "neighbor_row": int(global_j),
                            "neighbor_domain": domain_names[neighbor_domain_id],
                            "neighbor_item_id": str(item_ids[global_j]),
                            "cosine": float(sims[r, best_idx]),
                            "is_cross_domain": int(query_domain_id != neighbor_domain_id),
                        }
                    )

    rates = {thr: collision_counts[thr] / n for thr in thresholds}
    return rates, examples


def save_examples(examples: Dict[float, List[dict]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    for thr, rows in examples.items():
        out_path = os.path.join(
            out_dir,
            f"collision_examples_thr_{str(thr).replace('.', '_')}.csv"
        )
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "query_row",
                    "query_domain",
                    "query_item_id",
                    "neighbor_row",
                    "neighbor_domain",
                    "neighbor_item_id",
                    "cosine",
                    "is_cross_domain",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("--expected_embed_dim", type=int, default=768)

    parser.add_argument(
        "--qb_embed_path",
        type=str,
        default="/mnt/data_2/data_oleg/recsys/ninerec/Downstream_datasets/QB/CLIP_emb/shard_00000_fused.npy",
    )
    parser.add_argument(
        "--qb_id_map_path",
        type=str,
        default="/mnt/data_2/data_oleg/recsys/ninerec/Downstream_datasets/QB/CLIP_emb/shard_00000_ids_incremental.npy",
    )

    parser.add_argument(
        "--tn_embed_path",
        type=str,
        default="/mnt/data_2/data_oleg/recsys/ninerec/Downstream_datasets/TN/CLIP_emb/shard_00000_fused.npy",
    )
    parser.add_argument(
        "--tn_id_map_path",
        type=str,
        default="/mnt/data_2/data_oleg/recsys/ninerec/Downstream_datasets/TN/CLIP_emb/shard_00000_ids_incremental.npy",
    )

    parser.add_argument(
        "--bili_embed_path",
        type=str,
        default="/mnt/data_2/data_hao/DisenID/data/Bili_10k/CLIP_emb/shard_00000_fused.npy",
    )
    parser.add_argument(
        "--bili_id_map_path",
        type=str,
        default="/mnt/data_2/data_hao/DisenID/data/Bili_10k/CLIP_emb/shard_00000_ids_incremental.npy",
    )

    parser.add_argument(
        "--beauty_embed_path",
        type=str,
        default="/mnt/data_2/data_hao/DisenID/data/Amazon/beauty/embedding.npy",
    )
    parser.add_argument(
        "--beauty_id_map_path",
        type=str,
        default="/mnt/data_2/data_hao/DisenID/data/Amazon/beauty/ids.npy",
    )


    # faiss args
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--add_batch_size", type=int, default=100000)
    parser.add_argument("--query_batch_size", type=int, default=10000)
    parser.add_argument("--top_k", type=int, default=50)

    # collision args
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.99, 0.97, 0.95, 0.9])
    parser.add_argument("--max_examples_per_threshold", type=int, default=200)

    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./collision_check_output")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    domain_configs = [
        {
            "name": "Beauty",
            "embed_path": args.beauty_embed_path,
            "id_map_path": args.beauty_id_map_path,
        },
    ]

    print("[1/4] Building dataset")
    dataset = MultiDomainItemDataset(
        domain_configs=domain_configs,
        expected_embed_dim=args.expected_embed_dim,
        return_item_id=True,
    )

    print("[2/4] Loading embeddings/item_ids/domain_ids from dataset")
    vecs = dataset.item_embeds.astype(np.float32, copy=False)
    domain_ids = dataset.domain_ids.astype(np.int64, copy=False)
    item_ids = np.asarray(dataset.item_global_ids, dtype=object)
    domain_names = dataset.domain_names
    domain_stats = {k: int(v) for k, v in dataset.get_domain_stats().items()}

    print(f"Loaded embeddings shape: {vecs.shape}")
    print(f"Loaded domain_ids shape: {domain_ids.shape}")
    print(f"Loaded item_ids shape: {item_ids.shape}")
    print(f"Domain stats: {domain_stats}")

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
        item_ids=item_ids,
        domain_ids=domain_ids,
        domain_names=domain_names,
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
        "domain_stats": domain_stats,
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
            "domain_configs": [
                {
                    "name": cfg["name"],
                    "embed_path": cfg["embed_path"],
                    "id_map_path": cfg["id_map_path"],
                }
                for cfg in domain_configs
            ],
        },
    }

    summary = to_jsonable(summary)

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