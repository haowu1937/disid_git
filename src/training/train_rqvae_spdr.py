import argparse
import os
import random
from typing import Dict, Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from src.data_loading.multi_domain_dataset import MultiDomainItemDataset
from src.data_loading.newsfeed_item_dataset import NewsFeedItemDataset
from src.models.utils import build_model, build_train_dataloader

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make results more reproducible (slower but stable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                    config: Dict[str, Any]) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


@torch.no_grad()
def _entropy_and_perplexity_from_counts(counts: torch.Tensor) -> tuple[float, float]:
    """
    counts: [K] non-negative ints
    returns: (entropy, perplexity)
    """
    counts = counts.to(torch.float64)
    total = counts.sum()
    if total.item() <= 0:
        return 0.0, 1.0
    p = counts / total
    # avoid log(0)
    p = p.clamp_min(1e-12)
    entropy = float(-(p * p.log()).sum().item())
    perplexity = float(torch.exp(torch.tensor(entropy, dtype=torch.float64)).item())
    return entropy, perplexity


@torch.no_grad()
def compute_rqvae_epoch_metrics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_type: str,
) -> Dict[str, Any]:
    """
    Collect indices over the whole dataloader and compute:
      - per-layer entropy/perplexity
      - overall entropy/perplexity (treat (layer, token) as a joint symbol)
      - collision rates based on token sequences (row-wise indices)
          * collision_group_rate: fraction of unique token sequences that appear >=2 times
          * collision_item_rate : fraction of items that belong to a collided group
    """
    model_was_training = model.training
    model.eval()

    all_indices = []  # list of [B, L] Long
    for batch in dataloader:
        x, d = batch
        x = x.to(device, non_blocking=True).float()
        d = d.to(device, non_blocking=True).long()

        if model_type == "baseline":
            _, indices, _ = model(x)
        elif model_type == "disentangle":
            _, indices, _ = model(x, d)
        else:
            raise ValueError(f"Unsupported model.type={model_type}")

        all_indices.append(indices.detach().to("cpu", non_blocking=False))

    if model_was_training:
        model.train()

    if len(all_indices) == 0:
        return {
            "per_layer": [],
            "overall_entropy": 0.0,
            "overall_perplexity": 1.0,
            "collision_group_rate": 0.0,
            "collision_item_rate": 0.0,
            "num_items": 0,
        }

    indices_mat = torch.cat(all_indices, dim=0)  # [N, L]
    N, L = indices_mat.shape

    # ---- per-layer entropy/perplexity ----
    per_layer = []
    K = getattr(model, "codebook_size", None)

    for layer in range(L):
        idx = indices_mat[:, layer]  # [N]
        if K is None:
            # fallback: infer K from max token id in this layer
            K_layer = int(idx.max().item()) + 1 if N > 0 else 1
        else:
            K_layer = int(K)

        counts = torch.bincount(idx, minlength=K_layer)
        ent, ppl = _entropy_and_perplexity_from_counts(counts)
        per_layer.append(
            {
                "layer": layer,
                "entropy": ent,
                "perplexity": ppl,
                "utilization": float((counts > 0).sum().item() / max(1, K_layer)),
            }
        )

    # ---- overall entropy/perplexity ----
    # Treat (layer, token) as one symbol: symbol_id = layer*K + token
    if K is None:
        # infer a reasonable K across layers
        K_infer = int(indices_mat.max().item()) + 1 if N > 0 else 1
        K_use = max(1, K_infer)
    else:
        K_use = int(K)

    symbol_ids = (torch.arange(L).view(1, L) * K_use + indices_mat).reshape(-1)  # [N*L]
    counts_overall = torch.bincount(symbol_ids, minlength=L * K_use)
    overall_entropy, overall_perplexity = _entropy_and_perplexity_from_counts(counts_overall)

    # ---- collision rate on token sequences ----
    # Define a "group" by the full token sequence (indices row)
    # group_count: how many times each unique sequence appears
    # collision_group_rate: (#unique sequences with count>=2) / (#unique sequences)
    # collision_item_rate : (#items that are in collided sequences) / N
    unique_seqs, counts_seq = torch.unique(indices_mat, dim=0, return_counts=True)
    num_groups = int(unique_seqs.shape[0])
    num_collision_groups = int((counts_seq >= 2).sum().item())
    collision_group_rate = float(num_collision_groups / max(1, num_groups))

    num_collision_items = int(counts_seq[counts_seq >= 2].sum().item())
    collision_item_rate = float(num_collision_items / max(1, N))

    return {
        "per_layer": per_layer,
        "overall_entropy": overall_entropy,
        "overall_perplexity": overall_perplexity,
        "collision_group_rate": collision_group_rate,
        "collision_item_rate": collision_item_rate,
        "num_items": N,
        "num_groups": num_groups,
    }


def train(config: Dict[str, Any]) -> None:
    seed = int(config["training"].get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    data_configs = config["data"]
    domain_configs = data_configs["domain_configs"]
    dataset_name = data_configs.get("dataset_name", "public")
    expected_embed_dim = int(config["data"].get("expected_embed_dim", 768))
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 4))
    pin_memory = bool(config["training"].get("pin_memory", torch.cuda.is_available()))

    # Build dataset (return_item_id=False because we only need embeddings and domain ids)
    if dataset_name == "newsfeed":
        dataset = NewsFeedItemDataset(
        domain_configs,
        expected_embed_dim=expected_embed_dim,
        return_item_id=False,
        load_all_to_memory=bool(data_configs.get("load_all_to_memory", False))
    )
        shuffle = False # if shuffle will be super slow for random seek
    elif dataset_name == "public":
        dataset = MultiDomainItemDataset(
            domain_configs,
            expected_embed_dim=expected_embed_dim,
            return_item_id=False
        )
        shuffle = True
    else:
        raise ValueError(f"Unsupported dataset.type={dataset_name}.")

    dataloader, train_batch_sampler = build_train_dataloader(
        dataset=dataset,
        config=config,
        shuffle=shuffle,
    )

    # Model
    model = build_model(config, device)

    # Build a non-shuffled loader for all initialization procedures
    init_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Optional mean init for domain embedding (disentangle only)
    if config["model"]["type"] == "disentangle" and config["model"].get("domain_embed_init", "random") == "mean":
        print("Initializing domain embeddings with mean of per-domain items...")
        # The model must implement init_domain_embeddings_with_mean
        model.init_domain_embeddings_with_mean(init_loader, device)


    codebook_init_mode = str(config["training"].get("codebook_init", "random")).lower()
    codebook_init_max_batches = config["training"].get("codebook_init_max_batches")

    if codebook_init_max_batches is not None:
        codebook_init_max_batches = int(codebook_init_max_batches)
        if codebook_init_max_batches <= 0:
            codebook_init_max_batches = None

    if codebook_init_mode == "random":
        print("[Codebook Init] Skip KMeans init. Use random codebook initialization.")

    elif codebook_init_mode == "kmeans":

        print(
            f"[Codebook Init] Running KMeans codebook initialization before training "
            f"(max_init_batches={codebook_init_max_batches})."
        )
        model.init_codebooks_with_kmeans(
            dataloader=init_loader,
            device=device,
            max_init_batches=codebook_init_max_batches,
            seed=seed,
        )
        print("[Codebook Init] Computing utilization metrics immediately after KMeans init...")

        class _CappedLoader:
            def __init__(self, dl, max_batches):
                self.dl = dl
                self.max_batches = max_batches

            def __iter__(self):
                for i, b in enumerate(self.dl):
                    if i >= self.max_batches:
                        break
                    yield b

            def __len__(self):
                return min(len(self.dl), self.max_batches)

        init_metrics_loader = _CappedLoader(init_loader, int(codebook_init_max_batches))

        init_metrics = compute_rqvae_epoch_metrics(
            model=model,
            dataloader=init_metrics_loader,
            device=device,
            model_type=config["model"]["type"],
        )

        per_layer_str = " | ".join(
            [
                f"L{m['layer']}:ppl={m['perplexity']:.1f},H={m['entropy']:.3f},util={m['utilization']:.2%}"
                for m in init_metrics["per_layer"]
            ]
        )

        print(
            "[Codebook Init] Post-init metrics | "
            f"overall:ppl={init_metrics['overall_perplexity']:.1f},H={init_metrics['overall_entropy']:.3f} | "
            f"collision_item_rate={init_metrics['collision_item_rate']:.4f} | "
            f"collision_group_rate={init_metrics['collision_group_rate']:.4f} | "
            f"{per_layer_str}"
        )

    else:
        raise ValueError(
            f"Unsupported training.codebook_init={codebook_init_mode}. "
            f"Expected one of: random, kmeans."
        )

    # Optimizer
    lr = float(config["training"]["lr"])
    weight_decay = float(config["training"].get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train loop
    num_epochs = int(config["training"]["num_epochs"])
    save_dir = config["training"]["save_dir"]
    save_every = int(config["training"].get("save_every", 1))
    grad_clip = float(config["training"].get("grad_clip", 1.0))

    os.makedirs(save_dir, exist_ok=True)

    # Metrics config
    metrics_every = int(config["training"].get("metrics_every", 1))  # 每隔几个epoch算一次

    model.train()
    for epoch in range(1, num_epochs + 1):
        if train_batch_sampler is not None and hasattr(train_batch_sampler, "set_epoch"):
            train_batch_sampler.set_epoch(epoch)
        total_loss = 0.0
        total_recon = 0.0
        total_codebook = 0.0
        total_commit = 0.0
        total_spdr_align = 0.0
        total_spdr_separation = 0.0

        for batch in dataloader:
            x, d = batch
            x = x.to(device, non_blocking=True).float()
            d = d.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            if config["model"]["type"] == "baseline":
                _, _, losses = model(x)
            elif config["model"]["type"] == "disentangle":
                _, _, losses = model(x, d)
            else:
                raise ValueError(f"Unsupported model.type={config['model']['type']}")

            loss = losses["total"]
            if not torch.is_tensor(loss):
                raise RuntimeError(
                    "losses['total'] must be a torch.Tensor for backward(). "
                    "Make sure your model forward returns tensor losses (no .item())."
                )

            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total_loss += float(loss.detach().item())
            if "recon" in losses:
                total_recon += float(losses["recon"].detach().item())
            if "codebook" in losses:
                total_codebook += float(losses["codebook"].detach().item())
            if "commit" in losses:
                total_commit += float(losses["commit"].detach().item())
            if "spdr_align" in losses:
                total_spdr_align += float(losses["spdr_align"].detach().item())
            if "spdr_separation" in losses:
                total_spdr_separation += float(losses["spdr_separation"].detach().item())

        n_batches = max(1, len(dataloader))
        msg = (
            f"Epoch {epoch:03d} | "
            f"loss={total_loss / n_batches:.6f} | "
            f"recon={total_recon / n_batches:.6f} | "
            f"codebook={total_codebook / n_batches:.6f} | "
            f"commit={total_commit / n_batches:.6f} | "
            f"spdr_align={total_spdr_align / n_batches:.6f} | "
            f"spdr_separation={total_spdr_separation / n_batches:.6f}"
        )

        # ---- epoch-end metrics ----
        if metrics_every > 0 and (epoch % metrics_every == 0):
            metrics_base_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )

            metrics = compute_rqvae_epoch_metrics(
                model=model,
                dataloader=metrics_base_loader,
                device=device,
                model_type=config["model"]["type"],
            )

            per_layer_str = " | ".join(
                [
                    f"L{m['layer']}:ppl={m['perplexity']:.1f},H={m['entropy']:.3f},util={m['utilization']:.2%}"
                    for m in metrics["per_layer"]
                ]
            )

            msg += (
                f" | overall:ppl={metrics['overall_perplexity']:.1f},H={metrics['overall_entropy']:.3f}"
                f" | collision_item_rate={metrics['collision_item_rate']:.4f}"
                f" | collision_group_rate={metrics['collision_group_rate']:.4f}"
                f" | {per_layer_str}"
            )

        print(msg)

        # Save checkpoint
        if save_every > 0 and (epoch % save_every == 0):
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pt")
            save_checkpoint(ckpt_path, model, optimizer, epoch, config)

    # Save final
    final_path = os.path.join(save_dir, "final.pt")
    save_checkpoint(final_path, model, optimizer, num_epochs, config)
    print(f"Saved final checkpoint to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Ensure num_domains is set (can be inferred from domain_configs)
    if "num_domains" not in config["data"]:
        config["data"]["num_domains"] = len(config["data"]["domain_configs"])

    train(config)