# src/models/opq_sid.py

import json
import math
import os
from typing import Any, Dict, Optional

import numpy as np


class OPQSID:
    """
    OPQ semantic ID generator.

    This is a non-gradient baseline.

    Input:
        item embeddings: [N, D]

    Output:
        raw semantic IDs: [N, n_codebook], each value in [0, codebook_size - 1]
        offset tokens:    [N, n_codebook], RPG-style token IDs

    RPG-style token layout when codebook_size=256 and n_codebook=32:
        0: padding
        1-256: digit 0
        257-512: digit 1
        ...
        eos_token = n_codebook * codebook_size + 1
    """

    def __init__(
        self,
        input_dim: int,
        n_codebook: int,
        codebook_size: int = 256,
        metric: str = "inner_product",
        opq_use_gpu: bool = False,
        opq_gpu_id: int = 0,
        faiss_omp_num_threads: Optional[int] = 32,
        l2_normalize: bool = False,
        opq_output_dim: Optional[int] = None,
    ):
        self.input_dim = int(input_dim)
        self.n_codebook = int(n_codebook)
        self.codebook_size = int(codebook_size)
        self.metric = str(metric).lower()
        self.opq_use_gpu = bool(opq_use_gpu)
        self.opq_gpu_id = int(opq_gpu_id)
        self.faiss_omp_num_threads = faiss_omp_num_threads
        self.l2_normalize = bool(l2_normalize)
        self.opq_output_dim = opq_output_dim

        self.n_bits = self._get_n_bits(self.codebook_size)
        self.index_factory = self._build_index_factory()
        self.index = None

        self._validate()

    @staticmethod
    def _get_n_bits(codebook_size: int) -> int:
        bits = math.log2(codebook_size)

        if not bits.is_integer():
            raise ValueError(
                f"codebook_size must be a power of 2, got {codebook_size}."
            )

        return int(bits)

    def _build_index_factory(self) -> str:
        if self.opq_output_dim is None:
            return f"OPQ{self.n_codebook},IVF1,PQ{self.n_codebook}x{self.n_bits}"

        return (
            f"OPQ{self.n_codebook}_{int(self.opq_output_dim)},"
            f"IVF1,"
            f"PQ{self.n_codebook}x{self.n_bits}"
        )

    def _validate(self) -> None:
        pq_dim = self.input_dim if self.opq_output_dim is None else int(self.opq_output_dim)

        if pq_dim % self.n_codebook != 0:
            raise ValueError(
                f"PQ dimension must be divisible by n_codebook. "
                f"Got pq_dim={pq_dim}, n_codebook={self.n_codebook}. "
                f"For input_dim=2048, recommended n_codebook values include "
                f"4, 8, 16, 32, 64."
            )

        if self.metric not in {"inner_product", "l2"}:
            raise ValueError(
                f"Unsupported metric={self.metric}. "
                f"Expected 'inner_product' or 'l2'."
            )

    def _prepare_embeddings(self, x: np.ndarray) -> np.ndarray:
        import faiss

        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape={x.shape}")

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got embedding_dim={x.shape[1]}"
            )

        x = np.ascontiguousarray(x)

        if self.l2_normalize:
            x = x.copy()
            faiss.normalize_L2(x)

        return x

    def _faiss_metric(self):
        import faiss

        if self.metric == "inner_product":
            return faiss.METRIC_INNER_PRODUCT

        return faiss.METRIC_L2

    def _build_faiss_index(self):
        import faiss

        if self.faiss_omp_num_threads is not None:
            faiss.omp_set_num_threads(int(self.faiss_omp_num_threads))

        index = faiss.index_factory(
            self.input_dim,
            self.index_factory,
            self._faiss_metric(),
        )

        if self.opq_use_gpu:
            if not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError(
                    "opq_use_gpu=True requires faiss-gpu, but current FAISS "
                    "installation does not expose GPU APIs."
                )

            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)

            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.n_codebook >= 56

            index = faiss.index_cpu_to_gpu(
                res,
                self.opq_gpu_id,
                index,
                co,
            )

        return index

    @staticmethod
    def _to_cpu_index(index):
        import faiss

        try:
            return faiss.index_gpu_to_cpu(index)
        except Exception:
            return index

    def _decode_pq_codes(self, pq_codes: np.ndarray) -> np.ndarray:
        """
        Decode packed FAISS PQ codes into [N, n_codebook].

        For codebook_size=256, each code uses 8 bits, so the packed code
        can be read directly as uint8.
        """
        import faiss

        pq_codes = np.asarray(pq_codes, dtype=np.uint8)
        n_items, n_bytes = pq_codes.shape

        if self.n_bits == 8:
            if n_bytes < self.n_codebook:
                raise RuntimeError(
                    f"PQ code has {n_bytes} bytes, but n_codebook={self.n_codebook}."
                )
            return pq_codes[:, : self.n_codebook].astype(np.int64)

        sem_ids = np.empty((n_items, self.n_codebook), dtype=np.int64)

        for i, u8code in enumerate(pq_codes):
            u8code = np.ascontiguousarray(u8code, dtype=np.uint8)
            reader = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)

            for digit in range(self.n_codebook):
                sem_ids[i, digit] = reader.read(self.n_bits)

        return sem_ids

    def _extract_sem_ids_from_index(self, index, expected_items: int) -> np.ndarray:
        """
        Extract PQ codes from IVF1 list.

        This follows the RPG implementation style:
            index -> index.index -> extract_index_ivf -> invlists -> get_codes(0)
        """
        import faiss

        index = self._to_cpu_index(index)

        if not hasattr(index, "index"):
            raise RuntimeError(
                "Expected FAISS IndexPreTransform from OPQ...,IVF1,PQ..., "
                "but index has no `.index` attribute."
            )

        ivf_index = faiss.downcast_index(index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists

        list_size = invlists.list_size(0)

        if list_size != expected_items:
            raise RuntimeError(
                f"Expected IVF list size={expected_items}, got {list_size}. "
                f"Make sure index_factory uses IVF1 and all vectors were added."
            )

        pq_codes = faiss.rev_swig_ptr(
            invlists.get_codes(0),
            list_size * invlists.code_size,
        )

        pq_codes = pq_codes.reshape(list_size, invlists.code_size)

        sem_ids = self._decode_pq_codes(pq_codes)

        if sem_ids.shape != (expected_items, self.n_codebook):
            raise RuntimeError(
                f"Unexpected sem_ids shape={sem_ids.shape}, "
                f"expected ({expected_items}, {self.n_codebook})."
            )

        return sem_ids

    def fit_transform(
        self,
        embeddings: np.ndarray,
        train_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Train OPQ/PQ and generate raw semantic IDs.

        Args:
            embeddings:
                all item embeddings, shape [N, D]
            train_mask:
                optional boolean mask. If provided, only selected items are used
                to train OPQ/PQ, but all items are encoded.

        Returns:
            sem_ids:
                shape [N, n_codebook], values in [0, codebook_size - 1]
        """
        embeddings = self._prepare_embeddings(embeddings)
        n_items = embeddings.shape[0]

        if train_mask is None:
            train_embeddings = embeddings
        else:
            train_mask = np.asarray(train_mask, dtype=bool)

            if train_mask.shape[0] != n_items:
                raise ValueError(
                    f"train_mask length mismatch: got {train_mask.shape[0]}, "
                    f"expected {n_items}."
                )

            if train_mask.sum() <= 0:
                raise ValueError("train_mask selects zero items.")

            train_embeddings = embeddings[train_mask]

        print(f"[OPQ] index_factory = {self.index_factory}")
        print(f"[OPQ] metric = {self.metric}")
        print(f"[OPQ] train embeddings = {train_embeddings.shape}")
        print(f"[OPQ] all embeddings = {embeddings.shape}")

        index = self._build_faiss_index()

        print("[OPQ] Training FAISS OPQ/PQ index...")
        index.train(train_embeddings)

        print("[OPQ] Adding all embeddings...")
        index.add(embeddings)

        print("[OPQ] Extracting PQ codes as semantic IDs...")
        sem_ids = self._extract_sem_ids_from_index(index, expected_items=n_items)

        self.index = self._to_cpu_index(index)

        return sem_ids

    def sem_ids_to_tokens(self, sem_ids: np.ndarray) -> np.ndarray:
        """
        Convert raw OPQ IDs to RPG-style offset tokens.

        raw:
            digit 0: 0 ~ K-1
            digit 1: 0 ~ K-1

        tokenized:
            digit 0: 1 ~ K
            digit 1: K+1 ~ 2K
        """
        sem_ids = np.asarray(sem_ids, dtype=np.int64)

        if sem_ids.ndim != 2:
            raise ValueError(f"Expected sem_ids to be 2D, got {sem_ids.shape}")

        if sem_ids.shape[1] != self.n_codebook:
            raise ValueError(
                f"Expected sem_ids.shape[1]={self.n_codebook}, got {sem_ids.shape[1]}"
            )

        offsets = np.arange(self.n_codebook, dtype=np.int64) * self.codebook_size + 1
        return sem_ids + offsets.reshape(1, -1)

    @property
    def padding_token(self) -> int:
        return 0

    @property
    def eos_token(self) -> int:
        return self.n_codebook * self.codebook_size + 1

    @property
    def vocab_size(self) -> int:
        return self.eos_token + 1

    def save_index(self, path: str) -> None:
        if self.index is None:
            raise RuntimeError("No FAISS index available. Run fit_transform() first.")

        import faiss

        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)

    def save_metadata(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        metadata = {
            "method": "opq",
            "input_dim": self.input_dim,
            "n_codebook": self.n_codebook,
            "codebook_size": self.codebook_size,
            "n_bits": self.n_bits,
            "metric": self.metric,
            "index_factory": self.index_factory,
            "opq_output_dim": self.opq_output_dim,
            "l2_normalize": self.l2_normalize,
            "padding_token": self.padding_token,
            "eos_token": self.eos_token,
            "vocab_size": self.vocab_size,
        }

        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)


def compute_opq_sid_metrics(
    sem_ids: np.ndarray,
    codebook_size: int,
) -> Dict[str, Any]:
    """
    Basic OPQ SID diagnostics:
        - per-digit entropy
        - per-digit perplexity
        - per-digit utilization
        - full-SID collision rate
    """
    sem_ids = np.asarray(sem_ids, dtype=np.int64)

    if sem_ids.ndim != 2:
        raise ValueError(f"Expected sem_ids to be 2D, got {sem_ids.shape}")

    n_items, n_digit = sem_ids.shape
    codebook_size = int(codebook_size)

    per_layer = []

    for digit in range(n_digit):
        ids = sem_ids[:, digit]
        counts = np.bincount(ids, minlength=codebook_size).astype(np.float64)

        total = counts.sum()

        if total <= 0:
            entropy = 0.0
            perplexity = 1.0
            utilization = 0.0
        else:
            p = counts / total
            p = np.clip(p, 1e-12, None)

            entropy = float(-(p * np.log(p)).sum())
            perplexity = float(np.exp(entropy))
            utilization = float((counts > 0).sum() / max(1, codebook_size))

        per_layer.append(
            {
                "digit": digit,
                "entropy": entropy,
                "perplexity": perplexity,
                "utilization": utilization,
            }
        )

    unique_sids, counts = np.unique(sem_ids, axis=0, return_counts=True)

    num_groups = int(unique_sids.shape[0])
    num_collision_groups = int((counts >= 2).sum())
    num_collision_items = int(counts[counts >= 2].sum())

    return {
        "num_items": int(n_items),
        "num_groups": num_groups,
        "collision_group_rate": float(num_collision_groups / max(1, num_groups)),
        "collision_item_rate": float(num_collision_items / max(1, n_items)),
        "per_layer": per_layer,
    }