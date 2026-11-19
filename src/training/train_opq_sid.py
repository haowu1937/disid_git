# train_opq_sid.py

import argparse
import json
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data_loading.multi_domain_dataset import MultiDomainItemDataset
from src.data_loading.newsfeed_item_dataset import NewsFeedItemDataset
from src.models.opq_sid import OPQSID, compute_opq_sid_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_x_from_batch(batch: Any) -> torch.Tensor:
    """
    Current dataset format:
        batch = (x, d)

    Also supports:
        batch = x
        batch = {"x": x}
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 0:
            raise ValueError("Received empty batch.")
        return batch[0]

    if isinstance(batch, dict):
        if "x" not in batch:
            raise ValueError("Batch dict does not contain key 'x'.")
        return batch["x"]

    if torch.is_tensor(batch):
        return batch

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def build_dataset(config: Dict[str, Any]):
    data_configs = config["data"]

    domain_configs = data_configs["domain_configs"]
    dataset_name = data_configs.get("dataset_name", "public")
    expected_embed_dim = int(data_configs.get("expected_embed_dim", 768))

    if dataset_name == "newsfeed":
        dataset = NewsFeedItemDataset(
            domain_configs,
            expected_embed_dim=expected_embed_dim,
            return_item_id=False,
            load_all_to_memory=bool(data_configs.get("load_all_to_memory", False)),
        )

    elif dataset_name == "public":
        dataset = MultiDomainItemDataset(
            domain_configs,
            expected_embed_dim=expected_embed_dim,
            return_item_id=False,
        )

    else:
        raise ValueError(f"Unsupported data.dataset_name={dataset_name}")

    return dataset


@torch.no_grad()
def collect_embeddings(
    dataloader: DataLoader,
    max_batches: Optional[int] = None,
) -> np.ndarray:
    all_x = []

    n_batches = 0
    n_items = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = extract_x_from_batch(batch)
        x = x.detach().cpu().float().numpy().astype(np.float32, copy=False)

        all_x.append(x)

        n_batches += 1
        n_items += int(x.shape[0])

        if n_batches % 100 == 0:
            print(
                f"[Collect] batches={n_batches}, "
                f"items={n_items}, "
                f"last_batch_shape={x.shape}"
            )

    if len(all_x) == 0:
        raise ValueError("No embeddings were collected.")

    embeddings = np.concatenate(all_x, axis=0)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    print(f"[Collect] Finished. embeddings shape={embeddings.shape}")

    return embeddings


def build_train_mask(
    num_items: int,
    max_train_samples: Optional[int],
    seed: int,
) -> Optional[np.ndarray]:
    """
    If max_train_samples is None, OPQ is trained on all items.

    If max_train_samples is set, OPQ/PQ is trained on a random subset,
    but all items are still encoded.
    """
    if max_train_samples is None:
        return None

    max_train_samples = int(max_train_samples)

    if max_train_samples <= 0 or max_train_samples >= num_items:
        return None

    rng = np.random.default_rng(seed)
    selected = rng.choice(num_items, size=max_train_samples, replace=False)

    mask = np.zeros(num_items, dtype=bool)
    mask[selected] = True

    print(
        f"[OPQ] Train subset enabled: "
        f"{max_train_samples}/{num_items} items"
    )

    return mask


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def train(config: Dict[str, Any]) -> None:
    seed = int(config["training"].get("seed", 42))
    set_seed(seed)

    data_configs = config["data"]
    model_configs = config["model"]
    train_configs = config["training"]

    expected_embed_dim = int(data_configs.get("expected_embed_dim", model_configs["input_dim"]))

    batch_size = int(train_configs.get("batch_size", 256))
    num_workers = int(train_configs.get("num_workers", 0))
    pin_memory = bool(train_configs.get("pin_memory", torch.cuda.is_available()))

    save_dir = train_configs["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    print("[OPQ] Building dataset...")
    dataset = build_dataset(config)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    max_collect_batches = train_configs.get("max_collect_batches")
    if max_collect_batches is not None:
        max_collect_batches = int(max_collect_batches)
        if max_collect_batches <= 0:
            max_collect_batches = None

    print("[OPQ] Collecting item embeddings...")
    embeddings = collect_embeddings(
        dataloader=dataloader,
        max_batches=max_collect_batches,
    )

    if embeddings.shape[1] != expected_embed_dim:
        raise ValueError(
            f"Expected embedding dim={expected_embed_dim}, "
            f"got {embeddings.shape[1]}."
        )

    train_mask = build_train_mask(
        num_items=embeddings.shape[0],
        max_train_samples=train_configs.get("opq_train_max_samples"),
        seed=seed,
    )

    opq = OPQSID(
        input_dim=int(model_configs.get("input_dim", expected_embed_dim)),
        n_codebook=int(model_configs["n_codebook"]),
        codebook_size=int(model_configs.get("codebook_size", 256)),
        metric=str(model_configs.get("metric", "inner_product")),
        opq_use_gpu=bool(model_configs.get("opq_use_gpu", False)),
        opq_gpu_id=int(model_configs.get("opq_gpu_id", 0)),
        faiss_omp_num_threads=model_configs.get("faiss_omp_num_threads", 32),
        l2_normalize=bool(model_configs.get("l2_normalize", False)),
        opq_output_dim=model_configs.get("opq_output_dim"),
    )

    sem_ids = opq.fit_transform(
        embeddings=embeddings,
        train_mask=train_mask,
    )

    tokens = opq.sem_ids_to_tokens(sem_ids)

    sem_ids_path = os.path.join(save_dir, "opq_sem_ids.npy")
    tokens_path = os.path.join(save_dir, "opq_tokens.npy")
    index_path = os.path.join(save_dir, "opq.index")
    metadata_path = os.path.join(save_dir, "opq_metadata.json")
    metrics_path = os.path.join(save_dir, "opq_metrics.json")

    np.save(sem_ids_path, sem_ids)
    np.save(tokens_path, tokens)

    print(f"[OPQ] Saved raw semantic IDs to: {sem_ids_path}")
    print(f"[OPQ] Saved RPG-style offset tokens to: {tokens_path}")

    if bool(train_configs.get("save_faiss_index", True)):
        opq.save_index(index_path)
        print(f"[OPQ] Saved FAISS index to: {index_path}")

    opq.save_metadata(metadata_path)
    print(f"[OPQ] Saved metadata to: {metadata_path}")

    metrics = compute_opq_sid_metrics(
        sem_ids=sem_ids,
        codebook_size=opq.codebook_size,
    )

    save_json(metrics_path, metrics)
    print(f"[OPQ] Saved metrics to: {metrics_path}")

    per_layer_str = " | ".join(
        [
            f"D{m['digit']}:ppl={m['perplexity']:.1f},"
            f"H={m['entropy']:.3f},"
            f"util={m['utilization']:.2%}"
            for m in metrics["per_layer"]
        ]
    )

    print(
        "[OPQ] Final metrics | "
        f"num_items={metrics['num_items']} | "
        f"num_groups={metrics['num_groups']} | "
        f"collision_item_rate={metrics['collision_item_rate']:.6f} | "
        f"collision_group_rate={metrics['collision_group_rate']:.6f} | "
        f"{per_layer_str}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if "num_domains" not in config["data"]:
        config["data"]["num_domains"] = len(config["data"]["domain_configs"])

    train(config)