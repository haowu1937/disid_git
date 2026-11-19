import argparse
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data_loading.multi_domain_dataset import MultiDomainItemDataset
from src.data_loading.newsfeed_item_dataset import NewsFeedItemDataset
from src.models.rq_kmeans import RQKMeans
from src.models.utils import build_train_dataloader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_rqkmeans_checkpoint(
    path: str,
    model: torch.nn.Module,
    config: Dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        path,
    )


@torch.no_grad()
def _entropy_and_perplexity_from_counts(counts: torch.Tensor) -> tuple[float, float]:
    counts = counts.to(torch.float64)
    total = counts.sum()

    if total.item() <= 0:
        return 0.0, 1.0

    p = counts / total
    p = p.clamp_min(1e-12)

    entropy = float(-(p * p.log()).sum().item())
    perplexity = float(torch.exp(torch.tensor(entropy, dtype=torch.float64)).item())

    return entropy, perplexity


@torch.no_grad()
def compute_rqkmeans_metrics(
    model: RQKMeans,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()

    all_indices = []

    total_loss = 0.0
    total_items = 0

    for batch in dataloader:
        x = model._extract_x_from_batch(batch)
        x = x.to(device, non_blocking=True).float()

        _, indices, losses = model(x)

        all_indices.append(indices.detach().cpu())

        batch_size = int(x.size(0))
        total_loss += float(losses["total"].detach().item()) * batch_size
        total_items += batch_size

    if len(all_indices) == 0:
        return {
            "mean_quantization_mse": 0.0,
            "per_layer": [],
            "overall_entropy": 0.0,
            "overall_perplexity": 1.0,
            "collision_group_rate": 0.0,
            "collision_item_rate": 0.0,
            "num_items": 0,
            "num_groups": 0,
        }

    indices_mat = torch.cat(all_indices, dim=0)  # [N, L]
    n, num_layers = indices_mat.shape
    k = int(model.codebook_size)

    per_layer = []

    for layer in range(num_layers):
        idx = indices_mat[:, layer]
        counts = torch.bincount(idx, minlength=k)

        entropy, perplexity = _entropy_and_perplexity_from_counts(counts)
        utilization = float((counts > 0).sum().item() / max(1, k))

        per_layer.append(
            {
                "layer": layer,
                "entropy": entropy,
                "perplexity": perplexity,
                "utilization": utilization,
            }
        )

    symbol_ids = (
        torch.arange(num_layers).view(1, num_layers) * k + indices_mat
    ).reshape(-1)

    counts_overall = torch.bincount(symbol_ids, minlength=num_layers * k)
    overall_entropy, overall_perplexity = _entropy_and_perplexity_from_counts(counts_overall)

    unique_seqs, counts_seq = torch.unique(indices_mat, dim=0, return_counts=True)

    num_groups = int(unique_seqs.shape[0])
    num_collision_groups = int((counts_seq >= 2).sum().item())

    collision_group_rate = float(num_collision_groups / max(1, num_groups))

    num_collision_items = int(counts_seq[counts_seq >= 2].sum().item())
    collision_item_rate = float(num_collision_items / max(1, n))

    return {
        "mean_quantization_mse": total_loss / max(1, total_items),
        "per_layer": per_layer,
        "overall_entropy": overall_entropy,
        "overall_perplexity": overall_perplexity,
        "collision_group_rate": collision_group_rate,
        "collision_item_rate": collision_item_rate,
        "num_items": n,
        "num_groups": num_groups,
    }


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
        shuffle = False

    elif dataset_name == "public":
        dataset = MultiDomainItemDataset(
            domain_configs,
            expected_embed_dim=expected_embed_dim,
            return_item_id=False,
        )
        shuffle = True

    else:
        raise ValueError(f"Unsupported data.dataset_name={dataset_name}")

    return dataset, shuffle


def train(config: Dict[str, Any]) -> None:
    seed = int(config["training"].get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_configs = config["data"]
    expected_embed_dim = int(data_configs.get("expected_embed_dim", 768))

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 4))
    pin_memory = bool(config["training"].get("pin_memory", torch.cuda.is_available()))

    dataset, shuffle = build_dataset(config)

    # Main loader for RQ-KMeans fitting.
    dataloader, _ = build_train_dataloader(
        dataset=dataset,
        config=config,
        shuffle=shuffle,
    )

    # Non-shuffled loader for initialization and metrics.
    eval_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model_config = config["model"]

    model = RQKMeans(
        input_dim=int(model_config.get("input_dim", expected_embed_dim)),
        codebook_size=int(model_config["codebook_size"]),
        num_layers=int(model_config["num_layers"]),
        keep_empty_codes=bool(model_config.get("keep_empty_codes", True)),
    ).to(device)

    init_mode = str(config["training"].get("codebook_init", "kmeans")).lower()
    max_init_batches = config["training"].get("codebook_init_max_batches")

    if max_init_batches is not None:
        max_init_batches = int(max_init_batches)
        if max_init_batches <= 0:
            max_init_batches = None

    num_iters = int(config["training"].get("num_iters", 20))

    model.fit(
        dataloader=dataloader,
        device=device,
        num_iters=num_iters,
        init_mode=init_mode,
        max_init_batches=max_init_batches,
        seed=seed,
    )

    print("[RQKMeans] Computing final metrics...")

    metrics = compute_rqkmeans_metrics(
        model=model,
        dataloader=eval_loader,
        device=device,
    )

    per_layer_str = " | ".join(
        [
            f"L{m['layer']}:ppl={m['perplexity']:.1f},"
            f"H={m['entropy']:.3f},"
            f"util={m['utilization']:.2%}"
            for m in metrics["per_layer"]
        ]
    )

    print(
        "[RQKMeans] Final metrics | "
        f"mse={metrics['mean_quantization_mse']:.8f} | "
        f"overall:ppl={metrics['overall_perplexity']:.1f},"
        f"H={metrics['overall_entropy']:.3f} | "
        f"collision_item_rate={metrics['collision_item_rate']:.4f} | "
        f"collision_group_rate={metrics['collision_group_rate']:.4f} | "
        f"{per_layer_str}"
    )

    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    final_path = os.path.join(save_dir, "rqkmeans_final.pt")
    save_rqkmeans_checkpoint(final_path, model, config)

    print(f"Saved RQKMeans checkpoint to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if "num_domains" not in config["data"]:
        config["data"]["num_domains"] = len(config["data"]["domain_configs"])

    train(config)