import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RQKMeans(nn.Module):
    """
    Residual Quantization KMeans.

    This model learns multiple codebooks using residual KMeans:

        residual_0 = x
        q_l = nearest_code(residual_l, codebook_l)
        residual_{l+1} = residual_l - q_l

    Compared with RQVAE:
      - no encoder
      - no decoder
      - no gradient training
      - codebooks are updated by assignment/update steps
      - forward returns the same style:
            quantized, indices, losses
    """

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        num_layers: int,
        keep_empty_codes: bool = True,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(input_dim)
        self.codebook_size = int(codebook_size)
        self.num_layers = int(num_layers)
        self.keep_empty_codes = bool(keep_empty_codes)

        self.codebooks = nn.ModuleList(
            [nn.Embedding(self.codebook_size, self.input_dim) for _ in range(self.num_layers)]
        )

        for emb in self.codebooks:
            nn.init.uniform_(emb.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)
            emb.weight.requires_grad_(False)

    @staticmethod
    def _extract_x_from_batch(batch: Any) -> torch.Tensor:
        """
        Support current dataset format:
            batch = (x, d)

        Also support:
            batch = x
            batch = {"x": x}
        """
        if isinstance(batch, (list, tuple)):
            if len(batch) == 0:
                raise ValueError("Received an empty batch.")
            return batch[0]

        if isinstance(batch, dict):
            if "x" not in batch:
                raise ValueError("Batch dict does not contain key 'x'.")
            return batch["x"]

        if torch.is_tensor(batch):
            return batch

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    @staticmethod
    def _nearest_code(x: torch.Tensor, codebook_weight: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D]
        codebook_weight: [K, D]

        return:
            idx: [B]
        """
        x2 = (x ** 2).sum(dim=1, keepdim=True)              # [B, 1]
        c2 = (codebook_weight ** 2).sum(dim=1).unsqueeze(0) # [1, K]
        dist = x2 + c2 - 2.0 * x @ codebook_weight.t()      # [B, K]
        return dist.argmin(dim=1)

    @staticmethod
    def _run_kmeans_with_fallback(
        x_cpu: torch.Tensor,
        codebook_size: int,
        seed: int,
        layer: int,
    ) -> torch.Tensor:
        """
        x_cpu: [N, D], CPU float32

        return:
            centers_cpu: [K, D], CPU float32
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError as e:
            raise ImportError(
                "RQKMeans init requires scikit-learn. "
                "Install it with: pip install scikit-learn"
            ) from e

        if x_cpu.ndim != 2:
            raise ValueError(f"x_cpu must be 2D, got shape={tuple(x_cpu.shape)}")

        n, d = x_cpu.shape

        if n <= 0:
            raise ValueError("Cannot run KMeans with zero samples.")

        if n == 1:
            print(
                f"[RQKMeans Init][Layer {layer}] only one sample is available; "
                f"repeat it to fill all {codebook_size} codes."
            )
            return x_cpu.repeat(codebook_size, 1)

        if n < codebook_size:
            print(
                f"[RQKMeans Init][Layer {layer}] num_samples={n} < codebook_size={codebook_size}; "
                f"use n_clusters={n} and pad remaining centers."
            )
            n_clusters = n
        else:
            n_clusters = codebook_size

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=10,
        )
        kmeans.fit(x_cpu.numpy())

        centers_cpu = torch.from_numpy(kmeans.cluster_centers_).float()

        if centers_cpu.size(0) < codebook_size:
            pad_count = codebook_size - centers_cpu.size(0)

            g = torch.Generator(device="cpu")
            g.manual_seed(seed + 1000 + layer)

            pad_idx = torch.randint(
                low=0,
                high=n,
                size=(pad_count,),
                generator=g,
            )

            extra = x_cpu.index_select(0, pad_idx)
            centers_cpu = torch.cat([centers_cpu, extra], dim=0)

        if centers_cpu.shape != (codebook_size, d):
            raise RuntimeError(
                f"Unexpected center shape: got {tuple(centers_cpu.shape)}, "
                f"expected ({codebook_size}, {d})"
            )

        return centers_cpu

    @torch.no_grad()
    def init_codebooks_with_kmeans(
        self,
        dataloader,
        device: torch.device,
        max_init_batches: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        """
        Layer-wise residual KMeans initialization.

        This is similar to your RQVAE init_codebooks_with_kmeans(),
        but it directly works on x instead of encoder(x).
        """
        self.eval()

        if max_init_batches is not None:
            max_init_batches = int(max_init_batches)
            if max_init_batches <= 0:
                max_init_batches = None

        print("[RQKMeans Init] Start initialization.")
        print(
            f"[RQKMeans Init] num_layers={self.num_layers}, "
            f"codebook_size={self.codebook_size}, "
            f"input_dim={self.input_dim}, "
            f"max_init_batches={max_init_batches}, "
            f"seed={seed}"
        )

        xs = []
        used_batches = 0
        used_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            if max_init_batches is not None and batch_idx >= max_init_batches:
                break

            x = self._extract_x_from_batch(batch)
            x = x.float().cpu()

            xs.append(x)
            used_batches += 1
            used_samples += int(x.size(0))

        if len(xs) == 0:
            raise ValueError("No samples were collected for RQKMeans initialization.")

        residual_cpu = torch.cat(xs, dim=0).contiguous()

        print(
            f"[RQKMeans Init] Collected samples: "
            f"batches={used_batches}, samples={used_samples}, "
            f"shape={tuple(residual_cpu.shape)}"
        )

        for layer in range(self.num_layers):
            print(f"[RQKMeans Init][Layer {layer}] running KMeans on residuals...")

            centers_cpu = self._run_kmeans_with_fallback(
                x_cpu=residual_cpu,
                codebook_size=self.codebook_size,
                seed=seed + layer,
                layer=layer,
            )

            self.codebooks[layer].weight.copy_(
                centers_cpu.to(
                    device=self.codebooks[layer].weight.device,
                    dtype=self.codebooks[layer].weight.dtype,
                )
            )

            idx = self._nearest_code(residual_cpu, centers_cpu)
            q = F.embedding(idx, centers_cpu)

            before = float(residual_cpu.norm(dim=1).mean().item())
            residual_cpu = residual_cpu - q
            after = float(residual_cpu.norm(dim=1).mean().item())

            used_codes = int(idx.unique().numel())

            print(
                f"[RQKMeans Init][Layer {layer}] done | "
                f"used_codes={used_codes}/{self.codebook_size} | "
                f"mean_residual_l2_before={before:.6f} | "
                f"mean_residual_l2_after={after:.6f}"
            )

        print("[RQKMeans Init] Finished.")

    @torch.no_grad()
    def fit_one_iteration(
        self,
        dataloader,
        device: torch.device,
    ) -> Dict[str, Any]:
        """
        One RQ-KMeans assignment/update iteration.

        Step 1:
            assign each sample to nearest code per residual layer

        Step 2:
            update each code as the mean residual assigned to it
        """
        self.eval()

        sums: List[torch.Tensor] = []
        counts: List[torch.Tensor] = []

        for _ in range(self.num_layers):
            sums.append(torch.zeros(self.codebook_size, self.input_dim, device=device))
            counts.append(torch.zeros(self.codebook_size, device=device))

        total_sq_error = 0.0
        total_items = 0
        total_dims = self.input_dim

        for batch in dataloader:
            x = self._extract_x_from_batch(batch)
            x = x.to(device, non_blocking=True).float()

            residual = x
            quantized = torch.zeros_like(x)

            for layer in range(self.num_layers):
                w = self.codebooks[layer].weight

                idx = self._nearest_code(residual, w)
                q = F.embedding(idx, w)

                sums[layer].index_add_(0, idx, residual)

                ones = torch.ones(idx.size(0), device=device, dtype=counts[layer].dtype)
                counts[layer].index_add_(0, idx, ones)

                quantized = quantized + q
                residual = residual - q

            total_sq_error += float(((x - quantized) ** 2).sum().item())
            total_items += int(x.size(0))

        # update centers
        per_layer_stats = []

        for layer in range(self.num_layers):
            w = self.codebooks[layer].weight
            cnt = counts[layer]
            used_mask = cnt > 0

            old_w = w.data.clone()
            new_w = old_w.clone()

            new_w[used_mask] = sums[layer][used_mask] / cnt[used_mask].unsqueeze(1)

            if not self.keep_empty_codes:
                # Simple fallback: randomly copy used centers to empty centers.
                # Usually keeping empty codes is more stable.
                empty_mask = ~used_mask
                num_empty = int(empty_mask.sum().item())
                used_idx = torch.nonzero(used_mask, as_tuple=False).view(-1)

                if num_empty > 0 and used_idx.numel() > 0:
                    rand_pos = torch.randint(
                        low=0,
                        high=used_idx.numel(),
                        size=(num_empty,),
                        device=device,
                    )
                    empty_idx = torch.nonzero(empty_mask, as_tuple=False).view(-1)
                    new_w[empty_idx] = new_w[used_idx[rand_pos]]

            w.copy_(new_w)

            used_codes = int(used_mask.sum().item())
            utilization = used_codes / max(1, self.codebook_size)

            shift = float((new_w - old_w).norm(dim=1).mean().item())

            per_layer_stats.append(
                {
                    "layer": layer,
                    "used_codes": used_codes,
                    "utilization": utilization,
                    "mean_center_shift": shift,
                }
            )

        mean_mse = total_sq_error / max(1, total_items * total_dims)
        mean_sq_l2 = total_sq_error / max(1, total_items)

        return {
            "num_items": total_items,
            "mean_mse": mean_mse,
            "mean_sq_l2": mean_sq_l2,
            "per_layer": per_layer_stats,
        }

    @torch.no_grad()
    def fit(
        self,
        dataloader,
        device: torch.device,
        num_iters: int = 20,
        init_mode: str = "kmeans",
        max_init_batches: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        """
        Full RQ-KMeans training procedure.
        """
        init_mode = str(init_mode).lower()

        if init_mode == "kmeans":
            self.init_codebooks_with_kmeans(
                dataloader=dataloader,
                device=device,
                max_init_batches=max_init_batches,
                seed=seed,
            )
        elif init_mode == "random":
            print("[RQKMeans] Use random initialized codebooks.")
        else:
            raise ValueError(f"Unsupported init_mode={init_mode}. Use random or kmeans.")

        for it in range(1, int(num_iters) + 1):
            stats = self.fit_one_iteration(dataloader=dataloader, device=device)

            layer_str = " | ".join(
                [
                    f"L{s['layer']}:used={s['used_codes']}/{self.codebook_size},"
                    f"util={s['utilization']:.2%},shift={s['mean_center_shift']:.6f}"
                    for s in stats["per_layer"]
                ]
            )

            print(
                f"[RQKMeans][Iter {it:03d}] "
                f"mse={stats['mean_mse']:.8f} | "
                f"sq_l2={stats['mean_sq_l2']:.6f} | "
                f"{layer_str}"
            )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: [B, input_dim]

        returns:
            quantized: [B, input_dim]
            indices: [B, num_layers]
            losses:
                quantization: MSE between x and quantized
                total: same as quantization
        """
        residual = x
        quantized = torch.zeros_like(x)
        indices = []

        for layer in range(self.num_layers):
            w = self.codebooks[layer].weight

            idx = self._nearest_code(residual, w)
            q = F.embedding(idx, w)

            indices.append(idx)

            quantized = quantized + q
            residual = residual - q

        quantization_loss = F.mse_loss(quantized, x)

        losses = {
            "quantization": quantization_loss,
            "recon": quantization_loss,
            "total": quantization_loss,
        }

        return quantized, torch.stack(indices, dim=1), losses