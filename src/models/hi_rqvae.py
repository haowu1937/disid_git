import torch
import torch.nn as nn
import torch.nn.functional as F

from .rqvae import RQVAE


class HierarchicalRQVAE(RQVAE):
    """
    Hierarchical RQ-VAE with:
      - first N layers shared across domains
      - remaining layers domain-conditioned via a learnable domain embedding e_domain

    Key design choice (to match your written hypothesis):
      - domain embedding affects *code assignment* (i.e., which centroid is chosen),
        by shifting residual -> residual + e_domain
      - BUT the reconstruction target remains the original residual (no shift injected into recon),
        i.e., we quantize residual, not residual+e_domain

    This avoids "leaking" e_domain into the reconstructed representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        codebook_size: int,
        num_layers: int,
        num_shared_layers: int,
        num_domains: int,
        domain_embed_init: str = "random",  # "random" or "mean"
    ):
        super().__init__(input_dim, hidden_dim, codebook_size, num_layers)

        assert 0 <= num_shared_layers <= num_layers
        assert domain_embed_init in ("random", "mean")

        self.num_shared_layers = num_shared_layers
        self.num_private_layers = num_layers - num_shared_layers
        self.num_domains = num_domains
        self.domain_embed_init = domain_embed_init

        # Domain embeddings live in the same space as encoder output (hidden_dim).
        self.domain_embeddings = nn.Embedding(num_domains, hidden_dim)
        # If "mean", you will call init_domain_embeddings_with_mean(...) externally.

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
            x = batch["embed"].to(device)
            domain = batch["domain_id"].to(device).long()

            h = self.encoder(x)  # [B, hidden_dim]
            all_hidden.append(h)
            all_domain_ids.append(domain)

        all_hidden = torch.cat(all_hidden, dim=0)         # [N, hidden_dim]
        all_domain_ids = torch.cat(all_domain_ids, dim=0) # [N]

        for d in range(self.num_domains):
            mask = (all_domain_ids == d)
            if mask.any():
                mean_h = all_hidden[mask].mean(dim=0)
                self.domain_embeddings.weight.data[d].copy_(mean_h)

        self.train()

    def _nearest_code(self, x: torch.Tensor, codebook: nn.Embedding) -> torch.Tensor:
        """
        x: [B, hidden_dim]
        codebook.weight: [K, hidden_dim]
        return idx: [B]
        """
        # cdist expects [B, 1, D] vs [1, K, D] -> [B, K]
        dist = torch.cdist(x.unsqueeze(1), codebook.weight.unsqueeze(0)).squeeze(1)
        return dist.argmin(dim=1)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor):
        """
        x: [B, input_dim]
        domain_ids: [B] int64
        returns:
          quantized_latent: [B, hidden_dim]
          indices: [B, num_layers]
          losses: dict of tensors
        """
        domain_ids = domain_ids.long()
        x_orig = x

        # Encode
        h = self.encoder(x)  # [B, hidden_dim]

        indices = []
        codebook_loss = torch.zeros([], device=h.device)

        # ======================
        # 1) Shared layers (0..N-1): standard RQ on residual r
        # ======================
        q_shared_sum = torch.zeros_like(h)
        r = h
        for layer in range(self.num_shared_layers):
            codebook = self.codebooks[layer]
            idx = self._nearest_code(r, codebook)
            indices.append(idx)

            q = codebook(idx)
            codebook_loss = codebook_loss + F.mse_loss(q.detach(), r) + \
                            self.commitment_weight * F.mse_loss(q, r.detach())

            q_shared_sum = q_shared_sum + q
            r = r - q.detach()

        # ======================
        # 2) Private layers (N..L-1): inject e_domain ONCE into residual, then quantize in shifted space
        # ======================
        if self.num_private_layers > 0:
            e = self.domain_embeddings(domain_ids)  # [B, hidden_dim]

            # shift ONCE at boundary
            r_tilde = r + e  # this is the space we quantize in private layers

            q_private_sum = torch.zeros_like(h)

            for layer in range(self.num_shared_layers, self.num_layers):
                codebook = self.codebooks[layer]

                # assignment in shifted space
                idx = self._nearest_code(r_tilde, codebook)
                indices.append(idx)

                q = codebook(idx)

                # loss in shifted space (so e_domain participates in loss)
                codebook_loss = codebook_loss + F.mse_loss(q.detach(), r_tilde) + \
                                self.commitment_weight * F.mse_loss(q, r_tilde.detach())

                q_private_sum = q_private_sum + q
                r_tilde = r_tilde - q.detach()

            # IMPORTANT: subtract e ONCE so total quantized latent targets original h
            quantized_latent = q_shared_sum + (q_private_sum - e)
        else:
            quantized_latent = q_shared_sum

        # Decode + recon loss in input space
        recon = self.decoder(quantized_latent)
        recon_loss = F.mse_loss(recon, x_orig)

        total_loss = recon_loss + codebook_loss

        losses = {
            "recon": recon_loss,
            "codebook": codebook_loss,
            "total": total_loss,
        }

        indices = torch.stack(indices, dim=1)  # [B, L]
        return quantized_latent, indices, losses