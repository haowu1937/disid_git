import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from src.models.mlp import MLP


class DisRQVAE_SV(nn.Module):
    """
    Hierarchical Residual Quantization VAE-style module for multi-domain recommendation.

    Design:
      - total num_layers residual quantization layers
      - all layers use shared codebooks / shared token vocabulary
      - first num_shared_layers layers quantize clean residuals
      - remaining layers quantize domain-conditioned residuals using the same shared codebooks
      - a learnable domain embedding is injected once before the first domain-conditioned layer
        as a conditioning signal for code selection only

    Forward returns:
      quantized: [B, hidden_dim]
      indices:   [B, num_layers] with ordering
                 [shared_1, ..., shared_N, conditioned_1, ..., conditioned_(L-N)]
      losses:    dict with keys {"recon", "codebook", "commit", "total"}
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        codebook_size: int,
        num_layers: int,
        num_shared_layers: int,
        num_domains: int,
        commitment_weight: float = 0.25,
        codebook_weight: float = 1.0,
        alpha = 0.2,
        normalize_codebook_loss: bool = True,
        encoder_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        activation: nn.Module = nn.ReLU,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        if num_shared_layers < 0 or num_shared_layers > num_layers:
            raise ValueError(
                f"num_shared_layers must be in [0, num_layers], got "
                f"num_shared_layers={num_shared_layers}, num_layers={num_layers}"
            )
        if num_domains <= 0:
            raise ValueError(f"num_domains must be positive, got {num_domains}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        self.num_shared_layers = num_shared_layers
        self.num_private_layers = num_layers - num_shared_layers
        self.num_domains = num_domains
        self.commitment_weight = float(commitment_weight)
        self.codebook_weight = float(codebook_weight)
        self.normalize_codebook_loss = bool(normalize_codebook_loss)
        self.debug_batch_count = 0
        self.debug_print_every = -1  # debug every n batches, set -1 to disable debug
        self.alpha = alpha
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [512, 256, 128]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [128, 256, 512]

        self.encoder_hidden_dims = list(encoder_hidden_dims)
        self.decoder_hidden_dims = list(decoder_hidden_dims)

        self.encoder = MLP(
            input_dim=input_dim,
            output_dim=hidden_dim,
            hidden_dim_list=list(self.encoder_hidden_dims),
            activation=activation,
            bias=bias,
            dropout=dropout,
        )

        # Keep the top-level attribute name `codebooks` consistent with baseline.
        # All layers use nn.Embedding, so all domains share the same token vocabulary.
        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, hidden_dim) for _ in range(num_layers)]
        )

        # Used only as a conditioning signal before the first domain-conditioned layer.
        self.domain_embedding = nn.Embedding(num_domains, hidden_dim)

        self.decoder = MLP(
            input_dim=hidden_dim,
            output_dim=input_dim,
            hidden_dim_list=list(self.decoder_hidden_dims),
            activation=activation,
            bias=bias,
            dropout=dropout,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Keep codebook init style aligned with baseline.
        for layer in range(self.num_layers):
            emb = self.codebooks[layer]
            nn.init.uniform_(emb.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

        # Small init for domain conditioning vectors as well.
        nn.init.uniform_(
            self.domain_embedding.weight,
            -1.0 / self.codebook_size,
            1.0 / self.codebook_size,
        )

    def _is_shared_layer(self, layer: int) -> bool:
        return layer < self.num_shared_layers

    @torch.no_grad()
    def init_domain_embeddings_with_mean(self, dataloader, device):
        """
        Initialize domain embeddings using the mean of encoder outputs per domain.
        Expects batches like:
          batch[0]: [B, input_dim]
          batch[1]: [B] domain_id int64
        """
        self.eval()

        all_hidden = []
        all_domain_ids = []

        for batch in dataloader:
            x = batch[0].to(device)
            domain = batch[1].to(device).long()

            h = self.encoder(x)  # [B, hidden_dim]
            all_hidden.append(h)
            all_domain_ids.append(domain)

        all_hidden = torch.cat(all_hidden, dim=0)         # [N, hidden_dim]
        all_domain_ids = torch.cat(all_domain_ids, dim=0) # [N]

        for d in range(self.num_domains):
            mask = (all_domain_ids == d)
            if mask.any():
                mean_h = all_hidden[mask].mean(dim=0)
                self.domain_embedding.weight.data[d].copy_(mean_h)

        self.train()

    @staticmethod
    def _nearest_code(residual: torch.Tensor, codebook_weight: torch.Tensor) -> torch.Tensor:
        """
        residual: [B, D]
        codebook_weight: [K, D]
        return idx: [B]
        """
        r2 = (residual ** 2).sum(dim=1, keepdim=True)          # [B, 1]
        c2 = (codebook_weight ** 2).sum(dim=1).unsqueeze(0)    # [1, K]
        dist = r2 + c2 - 2.0 * residual @ codebook_weight.t()  # [B, K]
        return dist.argmin(dim=1)

    @staticmethod
    def _extract_x_and_domain_from_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expected dataset format:
          batch = (x, domain_id)
        Also supports:
          batch = {"x": ..., "domain_id": ...}
          batch = {"x": ..., "domain": ...}
          batch = {"x": ..., "d": ...}
        """
        if isinstance(batch, (list, tuple)):
            if len(batch) < 2:
                raise ValueError(
                    "Expected batch to contain at least (x, domain_id), "
                    f"but got length={len(batch)}"
                )
            return batch[0], batch[1]

        if isinstance(batch, dict):
            if "x" not in batch:
                raise ValueError("Batch dict does not contain key 'x'.")
            if "domain_id" in batch:
                return batch["x"], batch["domain_id"]
            if "domain" in batch:
                return batch["x"], batch["domain"]
            if "d" in batch:
                return batch["x"], batch["d"]
            raise ValueError(
                "Batch dict does not contain domain key. "
                "Expected one of {'domain_id', 'domain', 'd'}."
            )

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _quantize_shared_vocab_layer(
        self,
        residual_for_selection: torch.Tensor,
        layer: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize using the layer's shared codebook.

        Args:
            residual_for_selection: [B, D]
            layer: absolute layer index in [0, num_layers)

        Returns:
            idx: [B]
            q:   [B, D]
        """
        emb = self.codebooks[layer]
        idx = self._nearest_code(residual_for_selection, emb.weight)
        q = emb(idx)
        return idx, q

    def forward(
        self,
        x: torch.Tensor,
        domain_id: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: [B, input_dim]
            domain_id: [B], integer tensor in [0, num_domains - 1]

        Also supports:
            forward(batch) where batch contains both x and domain_id, using
            _extract_x_and_domain_from_batch(batch)

        Returns:
            quantized: [B, hidden_dim]
            indices:   [B, num_layers]
            losses:    dict
        """
        if domain_id is None:
            x, domain_id = self._extract_x_and_domain_from_batch(x)

        if x.ndim != 2:
            raise ValueError(f"x must be [B, input_dim], got shape={tuple(x.shape)}")

        if domain_id.ndim != 1:
            raise ValueError(f"domain_id must be [B], got shape={tuple(domain_id.shape)}")

        if x.size(0) != domain_id.size(0):
            raise ValueError(
                f"Batch size mismatch: x has B={x.size(0)} but domain_id has B={domain_id.size(0)}"
            )

        domain_id = domain_id.to(device=x.device, dtype=torch.long, non_blocking=True)

        if domain_id.numel() > 0:
            min_domain = int(domain_id.min().item())
            max_domain = int(domain_id.max().item())
            if min_domain < 0 or max_domain >= self.num_domains:
                raise ValueError(
                    f"domain_id values must be in [0, {self.num_domains - 1}], "
                    f"but got min={min_domain}, max={max_domain}"
                )

        x_orig = x
        h = self.encoder(x)  # [B, hidden_dim]

        quantized = torch.zeros_like(h)
        residual = h

        indices = []
        codebook_loss = torch.zeros([], device=h.device, dtype=h.dtype)
        commit_loss = torch.zeros([], device=h.device, dtype=h.dtype)

        domain_cond = None
        if self.num_private_layers > 0:
            domain_cond = self.domain_embedding(domain_id)  # [B, hidden_dim]

        # Track a domain-conditioned residual stream for lower layers.
        cond_residual = None
        for layer in range(self.num_layers):
            if self._is_shared_layer(layer):
                idx, q = self._quantize_shared_vocab_layer(
                    residual_for_selection=residual,
                    layer=layer,
                )

                codebook_loss = codebook_loss + F.mse_loss(q, residual.detach())
                commit_loss = commit_loss + F.mse_loss(residual, q.detach())

                q_st = residual + (q - residual).detach()

                quantized = quantized + q_st
                residual = residual - q.detach()
                indices.append(idx)
                continue

            # Domain-conditioned layer with shared token vocabulary.
            target_residual = residual
            if layer == self.num_shared_layers:
                # Inject domain conditioning only once, before the first domain-conditioned layer.
                domain_cond_use = F.normalize(domain_cond, dim=-1)
                self.alpha = 0.02
                cond_residual = target_residual + self.alpha * domain_cond_use

                # ---- debug: scale comparison ----
                cond_shift = self.alpha * domain_cond_use
                if self.debug_print_every > 0 and self.debug_batch_count % self.debug_print_every == 0:
                    with torch.no_grad():
                        tr_norm = target_residual.norm(dim=1)  # [B]
                        shift_norm = cond_shift.norm(dim=1)  # [B]
                        ratio_shift_tr = shift_norm / tr_norm.clamp_min(1e-12)
                    print(
                        f"[debug] batch={self.debug_batch_count} | "
                        f"mean(||cond_shift||/||target_residual||)={ratio_shift_tr.mean().item():.6f}"
                    )

                self.debug_batch_count += 1
                # Selection ALWAYS uses the conditioned residual stream for all domain-conditioned layers.

            idx, q = self._quantize_shared_vocab_layer(
                residual_for_selection=cond_residual,
                layer=layer,
            )

            # Losses ALWAYS use the unconditioned target_residual (prevents target shift).
            codebook_loss = codebook_loss + F.mse_loss(q, target_residual.detach())
            commit_loss = commit_loss + F.mse_loss(target_residual, q.detach())

            # STE uses the cond_residual so domain_cond gets gradients from ALL domain-conditioned layers.
            q_st = cond_residual + (q - cond_residual).detach()

            quantized = quantized + q_st

            # Update BOTH residual streams.
            residual = target_residual - q.detach()  # Clean residual for the next loop.
            cond_residual = cond_residual - q.detach()  # Conditioned residual for the next loop.
            indices.append(idx)

        if self.normalize_codebook_loss and self.num_layers > 0:
            codebook_loss = codebook_loss / self.num_layers
            commit_loss = commit_loss / self.num_layers

        recon = self.decoder(quantized)
        recon_loss = F.mse_loss(recon, x_orig)

        total_loss = (
            recon_loss
            + self.codebook_weight * codebook_loss
            + self.commitment_weight * commit_loss
        )

        losses = {
            "recon": recon_loss,
            "codebook": codebook_loss,
            "commit": commit_loss,
            "total": total_loss,
        }

        return quantized, torch.stack(indices, dim=1), losses
