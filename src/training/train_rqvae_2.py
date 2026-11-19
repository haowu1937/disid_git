import argparse
import os
import random
from typing import Dict, Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data_loading.bakcup_multi_domain_dataset import MultiDomainItemDataset
from src.models.rqvae import RQVAE
from src.models.hierarchical_rqvae import HierarchicalRQVAE


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


def build_model(config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    input_dim = int(config["model"]["input_dim"])
    hidden_dim = int(config["model"]["hidden_dim"])
    codebook_size = int(config["model"]["codebook_size"])
    num_layers = int(config["model"]["num_layers"])

    model_type = config["model"]["type"]
    if model_type == "baseline":
        model = RQVAE(input_dim, hidden_dim, codebook_size, num_layers)
    elif model_type == "hierarchical":
        num_shared = int(config["model"]["num_shared_layers"])
        num_domains = int(config["data"]["num_domains"])  # explicitly set
        domain_embed_init = config["model"].get("domain_embed_init", "random")
        model = HierarchicalRQVAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            codebook_size=codebook_size,
            num_layers=num_layers,
            num_shared_layers=num_shared,
            num_domains=num_domains,
            domain_embed_init=domain_embed_init,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.to(device)
    return model


def train(config: Dict[str, Any]) -> None:
    seed = int(config["training"].get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    domain_configs = config["data"]["domain_configs"]
    expected_embed_dim = int(config["data"].get("expected_embed_dim", 768))
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 4))
    pin_memory = bool(config["training"].get("pin_memory", torch.cuda.is_available()))

    # Build dataset (return_item_id=False because we only need embeddings and domain ids)
    dataset = MultiDomainItemDataset(
        domain_configs,
        expected_embed_dim=expected_embed_dim,
        return_item_id=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # Model
    model = build_model(config, device)

    # Optional mean init for domain embedding (hierarchical only)
    if config["model"]["type"] == "hierarchical" and config["model"].get("domain_embed_init", "random") == "mean":
        print("Initializing domain embeddings with mean of per-domain items...")
        # Use a non-shuffled dataloader to gather all embeddings for mean calculation
        init_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        # The model must implement init_domain_embeddings_with_mean
        model.init_domain_embeddings_with_mean(init_loader, device)

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

    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_codebook = 0.0

        for batch in dataloader:
            # MultiDomainItemDataset returns (embed, domain_id) when return_item_id=False
            x, d = batch
            x = x.to(device, non_blocking=True).float()
            d = d.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            if config["model"]["type"] == "baseline":
                # baseline ignores domain id
                _, _, losses = model(x)
            else:
                _, _, losses = model(x, d)

            # IMPORTANT: must be a Tensor with grad
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

        n_batches = max(1, len(dataloader))
        print(
            f"Epoch {epoch:03d} | "
            f"loss={total_loss / n_batches:.6f} | "
            f"recon={total_recon / n_batches:.6f} | "
            f"codebook={total_codebook / n_batches:.6f}"
        )

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