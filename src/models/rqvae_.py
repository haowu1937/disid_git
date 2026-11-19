import torch
import torch.nn as nn
import torch.nn.functional as F


class RQVAE(nn.Module):
    """
    Residual Quantization VAE-style module (RQ-VAE / VQ-style training)

    Key fixes vs. your version:
    1) Correct VQ losses (codebook loss + commitment loss) detach directions.
    2) Add Straight-Through Estimator (STE) so recon gradients flow decoder -> encoder.
    3) Use fast squared L2 distance instead of torch.cdist broadcasting.
    4) Proper tensor init for losses + optional normalization by num_layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        codebook_size: int,
        num_layers: int,
        commitment_weight: float = 0.25,
        codebook_weight: float = 1.0,
        normalize_codebook_loss: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        self.commitment_weight = float(commitment_weight)
        self.codebook_weight = float(codebook_weight)
        self.normalize_codebook_loss = bool(normalize_codebook_loss)

        self.encoder = nn.Linear(input_dim, hidden_dim)

        # One embedding table per layer
        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, hidden_dim) for _ in range(num_layers)]
        )

        self.decoder = nn.Linear(hidden_dim, input_dim)

        # Optional: small init helps stability
        for emb in self.codebooks:
            nn.init.uniform_(emb.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    @staticmethod
    def _nearest_code(residual: torch.Tensor, codebook_weight: torch.Tensor) -> torch.Tensor:
        """
        residual: [B, D]
        codebook_weight: [K, D]
        return idx: [B]
        """
        # Squared L2: ||r-c||^2 = ||r||^2 + ||c||^2 - 2 r·c
        r2 = (residual ** 2).sum(dim=1, keepdim=True)          # [B, 1]
        c2 = (codebook_weight ** 2).sum(dim=1).unsqueeze(0)    # [1, K]
        dist = r2 + c2 - 2.0 * residual @ codebook_weight.t()  # [B, K]
        return dist.argmin(dim=1)

    @staticmethod
    def _extract_x_from_batch(batch) -> torch.Tensor:
        """
        Expected current dataset format:
          batch = (x, d)

        This helper keeps the init method slightly more robust without changing training logic.
        """
        if isinstance(batch, (list, tuple)):
            if len(batch) == 0:
                raise ValueError("Received an empty batch.")
            return batch[0]
        if isinstance(batch, dict):
            if "x" in batch:
                return batch["x"]
            raise ValueError("Batch dict does not contain key 'x'.")
        if torch.is_tensor(batch):
            return batch
        raise TypeError(f"Unsupported batch type for KMeans init: {type(batch)}")

    @staticmethod
    def _run_kmeans_with_fallback(
        samples_cpu: torch.Tensor,
        codebook_size: int,
        seed: int,
        layer: int,
    ) -> torch.Tensor:
        """
        samples_cpu: [N, D] on CPU, float32
        returns:
          centers_cpu: [codebook_size, D] on CPU, float32

        Fallback strategy when N < codebook_size:
          1) Run KMeans with n_clusters = N
          2) Pad the remaining centers by sampling existing samples with replacement

        This avoids crashes while keeping behavior simple and reproducible.
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError as e:
            raise ImportError(
                "KMeans initialization requires scikit-learn. "
                "Please install it first, e.g. `pip install scikit-learn`."
            ) from e

        if samples_cpu.ndim != 2:
            raise ValueError(f"samples_cpu must be 2D, got shape={tuple(samples_cpu.shape)}")

        num_samples, dim = samples_cpu.shape
        if num_samples <= 0:
            raise ValueError("Cannot run KMeans init with zero samples.")

        if num_samples == 1:
            print(
                f"[KMeans Init][Layer {layer}] only 1 sample is available; "
                f"repeat it to fill all {codebook_size} codebook entries."
            )
            return samples_cpu.repeat(codebook_size, 1)

        if num_samples < codebook_size:
            print(
                f"[KMeans Init][Layer {layer}] num_samples={num_samples} < codebook_size={codebook_size}; "
                f"run KMeans with n_clusters={num_samples} and pad the rest by resampling existing samples."
            )
            n_clusters = num_samples
        else:
            n_clusters = codebook_size

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=10,
        )
        kmeans.fit(samples_cpu.numpy())
        centers_cpu = torch.from_numpy(kmeans.cluster_centers_).to(dtype=torch.float32)

        # Pad to full codebook size if needed
        if centers_cpu.size(0) < codebook_size:
            pad_count = codebook_size - centers_cpu.size(0)
            g = torch.Generator(device="cpu")
            g.manual_seed(seed + 1000 + layer)
            pad_idx = torch.randint(
                low=0,
                high=num_samples,
                size=(pad_count,),
                generator=g,
            )
            extra_centers = samples_cpu.index_select(0, pad_idx)
            centers_cpu = torch.cat([centers_cpu, extra_centers], dim=0)

        # Just in case, enforce exact shape [K, D]
        if centers_cpu.size(0) != codebook_size or centers_cpu.size(1) != dim:
            raise RuntimeError(
                f"Unexpected centers shape after KMeans fallback: "
                f"got {tuple(centers_cpu.shape)}, expected ({codebook_size}, {dim})"
            )

        return centers_cpu

    @torch.no_grad()
    def init_codebooks_with_kmeans(
        self,
        dataloader,
        device,
        max_init_batches=None,
        seed: int = 42,
    ) -> None:
        """
        Initialize codebook embeddings using KMeans on encoder outputs and
        layer-wise residual quantization.

        Procedure:
          1) Collect x from dataloader
          2) Encode h = encoder(x)
          3) For each codebook layer:
               - run KMeans on current residual
               - write cluster centers into codebook weights
               - assign each sample to nearest center
               - update residual = residual - assigned_center

        Args:
            dataloader:
                DataLoader over training items.
            device:
                Device used to run the encoder.
            max_init_batches:
                If not None, only use the first max_init_batches batches.
                Useful to speed up initialization on very large datasets.
            seed:
                Random seed for reproducible KMeans / fallback sampling.
        """
        was_training = self.training
        self.eval()

        if max_init_batches is not None:
            max_init_batches = int(max_init_batches)
            if max_init_batches <= 0:
                max_init_batches = None

        print("[KMeans Init] Start codebook initialization with KMeans.")
        print(
            f"[KMeans Init] Settings: "
            f"num_layers={self.num_layers}, codebook_size={self.codebook_size}, "
            f"hidden_dim={self.hidden_dim}, max_init_batches={max_init_batches}, seed={seed}"
        )

        all_h_cpu = []
        used_batches = 0
        used_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            if max_init_batches is not None and batch_idx >= max_init_batches:
                break

            x = self._extract_x_from_batch(batch)
            x = x.to(device, non_blocking=True).float()

            h = self.encoder(x)  # [B, D]
            all_h_cpu.append(h.detach().to("cpu", dtype=torch.float32, non_blocking=False))

            used_batches += 1
            used_samples += int(h.size(0))

            if used_batches % 10 == 0:
                print(
                    f"[KMeans Init] Collected encoder features from {used_batches} batches "
                    f"({used_samples} samples) ..."
                )

        if len(all_h_cpu) == 0:
            raise ValueError("No samples were collected for KMeans initialization.")

        residual_cpu = torch.cat(all_h_cpu, dim=0).contiguous()  # [N, D], CPU float32

        print(
            f"[KMeans Init] Finished feature collection: "
            f"batches={used_batches}, samples={used_samples}, residual_shape={tuple(residual_cpu.shape)}"
        )

        for layer in range(self.num_layers):
            print(f"[KMeans Init][Layer {layer}] Running KMeans on residuals ...")

            centers_cpu = self._run_kmeans_with_fallback(
                samples_cpu=residual_cpu,
                codebook_size=self.codebook_size,
                seed=seed + layer,
                layer=layer,
            )

            # Directly write centers into the current codebook weights
            emb = self.codebooks[layer]
            emb.weight.copy_(
                centers_cpu.to(device=emb.weight.device, dtype=emb.weight.dtype)
            )

            # Hard assignment with the just-initialized codebook
            idx = self._nearest_code(residual_cpu, centers_cpu)      # [N]
            assigned_cpu = F.embedding(idx, centers_cpu)             # [N, D]

            used_codes = int(idx.unique().numel())
            mean_residual_norm_before = float(residual_cpu.norm(dim=1).mean().item())

            # Residual update for next layer
            residual_cpu = residual_cpu - assigned_cpu

            mean_residual_norm_after = float(residual_cpu.norm(dim=1).mean().item())

            print(
                f"[KMeans Init][Layer {layer}] done | "
                f"used_codes={used_codes}/{self.codebook_size} | "
                f"mean_residual_l2_before={mean_residual_norm_before:.6f} | "
                f"mean_residual_l2_after={mean_residual_norm_after:.6f}"
            )

        print("[KMeans Init] All codebooks initialized successfully.")

        if was_training:
            self.train()

    def forward(self, x: torch.Tensor):
        """
        x: [B, input_dim]
        returns:
          quantized: [B, hidden_dim]
          indices:   [B, num_layers]
          losses: dict
        """
        x_orig = x
        h = self.encoder(x)  # [B, D]

        quantized = torch.zeros_like(h)
        residual = h

        indices = []
        codebook_loss = torch.zeros([], device=h.device, dtype=h.dtype)
        commit_loss = torch.zeros([], device=h.device, dtype=h.dtype)

        for layer in range(self.num_layers):
            emb = self.codebooks[layer]              # nn.Embedding
            w = emb.weight                           # [K, D]

            # 1) hard assignment (non-differentiable wrt idx)
            idx = self._nearest_code(residual, w)    # [B]
            indices.append(idx)

            # 2) embedding lookup: quantized vector
            q = emb(idx)                             # [B, D]

            # 3) VQ losses (detach directions are IMPORTANT)
            #    - codebook loss updates codebook (pull code towards encoder output)
            #    - commitment loss updates encoder (encourage encoder to commit)
            codebook_loss = codebook_loss + F.mse_loss(q, residual.detach())
            commit_loss = commit_loss + F.mse_loss(residual, q.detach())

            # 4) STE: forward uses q, backward acts like identity on residual
            q_st = residual + (q - residual).detach()

            # 5) accumulate and update quantized
            quantized = quantized + q_st

            # For next layer residual:
            residual = residual - q.detach()

        # Optional normalization so loss scale doesn't grow with num_layers
        if self.normalize_codebook_loss and self.num_layers > 0:
            codebook_loss = codebook_loss / self.num_layers
            commit_loss = commit_loss / self.num_layers

        # reconstruction
        recon = self.decoder(quantized)
        recon_loss = F.mse_loss(recon, x_orig)

        total_loss = recon_loss + self.codebook_weight * codebook_loss + self.commitment_weight * commit_loss

        losses = {
            "recon": recon_loss,
            "codebook": codebook_loss,
            "commit": commit_loss,
            "total": total_loss,
        }

        return quantized, torch.stack(indices, dim=1), losses