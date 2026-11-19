import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from src.models.mlp import MLP


class DisRQVAE(nn.Module):
    """
    Hierarchical Residual Quantization VAE-style module for multi-domain recommendation.

    Design:
      - total num_layers residual quantization layers
      - first num_shared_layers layers use shared codebooks
      - remaining layers use domain-private codebooks
      - a learnable domain embedding is injected once before the first private layer
        as a conditioning signal for code selection only

    Forward returns:
      quantized: [B, hidden_dim]
      indices:   [B, num_layers] with ordering
                 [shared_1, ..., shared_N, private_1, ..., private_(L-N)]
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
        # - shared layer:  self.codebooks[layer] is nn.Embedding
        # - private layer: self.codebooks[layer] is ModuleList[nn.Embedding], one per domain
        codebooks = []
        for layer in range(num_layers):
            if layer < num_shared_layers:
                codebooks.append(nn.Embedding(codebook_size, hidden_dim))
            else:
                codebooks.append(
                    nn.ModuleList(
                        [nn.Embedding(codebook_size, hidden_dim) for _ in range(num_domains)]
                    )
                )
        self.codebooks = nn.ModuleList(codebooks)

        # Used only as a conditioning signal before the first private layer.
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
            if self._is_shared_layer(layer):
                emb = self.codebooks[layer]
                nn.init.uniform_(emb.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)
            else:
                for emb in self.codebooks[layer]:
                    nn.init.uniform_(
                        emb.weight,
                        -1.0 / self.codebook_size,
                        1.0 / self.codebook_size,
                    )

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
          batch["embed"]: [B, input_dim]
          batch["domain_id"]: [B] (int64)
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

    def _quantize_private_layer(
        self,
        residual_for_selection: torch.Tensor,
        layer: int,
        domain_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a mixed-domain batch using per-domain private codebooks.

        Args:
            residual_for_selection: [B, D]
            layer: absolute layer index in [0, num_layers)
            domain_id: [B]

        Returns:
            idx: [B]
            q:   [B, D]
        """
        batch_size = residual_for_selection.size(0)
        idx = torch.empty(batch_size, device=residual_for_selection.device, dtype=torch.long)
        q = torch.zeros_like(residual_for_selection)

        for d in domain_id.unique(sorted=True).detach().cpu().tolist():
            mask = (domain_id == d)
            emb = self.codebooks[layer][d]
            idx_d = self._nearest_code(residual_for_selection[mask], emb.weight)
            q_d = emb(idx_d)
            idx[mask] = idx_d
            q[mask] = q_d

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

        for layer in range(self.num_layers):
            if self._is_shared_layer(layer):
                emb = self.codebooks[layer]
                w = emb.weight

                idx = self._nearest_code(residual, w)
                q = emb(idx)

                codebook_loss = codebook_loss + F.mse_loss(q, residual.detach())
                commit_loss = commit_loss + F.mse_loss(residual, q.detach())

                q_st = residual + (q - residual).detach()

                quantized = quantized + q_st
                residual = residual - q.detach()
                indices.append(idx)
                continue

            # Private layer
            target_residual = residual

            if layer == self.num_shared_layers:
                # Inject domain conditioning only once, before the first private layer.
                # Selection uses the conditioned residual, but losses and residual update
                # stay aligned to the original residual stream.
                domain_cond_use = F.normalize(domain_cond, dim=-1)
                alpha = 0.01
                residual_for_selection = target_residual + alpha * domain_cond_use

                # residual_for_selection = target_residual + domain_cond
                idx, q = self._quantize_private_layer(
                    residual_for_selection=residual_for_selection,
                    layer=layer,
                    domain_id=domain_id,
                )

                codebook_loss = codebook_loss + F.mse_loss(q, target_residual.detach())
                commit_loss = commit_loss + F.mse_loss(target_residual, q.detach())

                # Forward value is still q, but the surrogate gradient path includes
                # domain conditioning so the domain embedding can learn useful selection behavior.
                q_st = residual_for_selection + (q - residual_for_selection).detach()
            else:
                residual_for_selection = target_residual
                idx, q = self._quantize_private_layer(
                    residual_for_selection=residual_for_selection,
                    layer=layer,
                    domain_id=domain_id,
                )

                codebook_loss = codebook_loss + F.mse_loss(q, target_residual.detach())
                commit_loss = commit_loss + F.mse_loss(target_residual, q.detach())

                q_st = target_residual + (q - target_residual).detach()

            quantized = quantized + q_st
            residual = target_residual - q.detach()
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