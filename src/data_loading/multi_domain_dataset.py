from typing import Union, Tuple
import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import math
import numpy as np
from typing import Iterator, List, Dict, Optional
from torch.utils.data import BatchSampler


class DomainBalancedBatchSampler(BatchSampler):
    """
    BatchSampler for multi-domain dataset.

    It generates a list of sample indices for each batch, with domain-balanced sampling.

    Compatible with datasets that provide:
        - __len__()
        - domain_ids: 1D array-like, length == len(dataset)

    Sampling modes:
        - uniform: all domains have equal probability
        - temperature: interpolate between natural frequency and balanced frequency

    Args:
        dataset:
            dataset with field `domain_ids`
        batch_size:
            batch size
        drop_last:
            whether to drop the last incomplete batch
        mode:
            "uniform" | "temperature"
        rho:
            interpolation strength for temperature mode
        tau:
            balance temperature for temperature mode
        replacement:
            whether to sample with replacement within each domain
        num_batches:
            number of batches per epoch; if None, computed from dataset size
        seed:
            random seed
    """
    def __init__(
        self,
        dataset,
        batch_size: int,
        drop_last: bool = False,
        mode: str = "temperature",
        rho: float = 0.7,
        tau: float = 0.5,
        replacement: bool = True,
        num_batches: Optional[int] = None,
        seed: int = 42,
    ):
        if not hasattr(dataset, "domain_ids"):
            raise ValueError("dataset must have attribute `domain_ids`")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.mode = mode
        self.rho = rho
        self.tau = tau
        self.replacement = replacement
        self.seed = seed

        self.domain_ids = np.asarray(dataset.domain_ids)
        if self.domain_ids.ndim != 1:
            raise ValueError(f"dataset.domain_ids must be 1D, got shape={self.domain_ids.shape}")

        self.num_samples = len(dataset)
        self.unique_domains = sorted(np.unique(self.domain_ids).tolist())
        self.num_domains = len(self.unique_domains)

        # Build domain -> global indices
        self.domain_to_indices: Dict[int, np.ndarray] = {}
        self.domain_counts: Dict[int, int] = {}
        for d in self.unique_domains:
            idxs = np.where(self.domain_ids == d)[0]
            self.domain_to_indices[d] = idxs
            self.domain_counts[d] = len(idxs)

        # Number of batches per epoch
        if num_batches is None:
            if drop_last:
                self.num_batches = self.num_samples // batch_size
            else:
                self.num_batches = math.ceil(self.num_samples / batch_size)
        else:
            self.num_batches = num_batches

        # Domain sampling probabilities
        self.domain_probs = self._build_domain_probs()

    def _build_domain_probs(self) -> np.ndarray:
        counts = np.array([self.domain_counts[d] for d in self.unique_domains], dtype=np.float64)
        natural = counts / counts.sum()

        if self.mode == "uniform":
            probs = np.ones_like(counts) / len(counts)

        elif self.mode == "temperature":
            # balanced part: n_d^{-tau}
            balanced = counts ** (-self.tau)
            balanced = balanced / balanced.sum()

            probs = (1.0 - self.rho) * natural + self.rho * balanced
            probs = probs / probs.sum()

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return probs

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed)

        # for no-replacement mode, we maintain shuffled pools
        domain_pools = {}
        domain_ptrs = {}

        if not self.replacement:
            for d in self.unique_domains:
                arr = self.domain_to_indices[d].copy()
                rng.shuffle(arr)
                domain_pools[d] = arr
                domain_ptrs[d] = 0

        for _ in range(self.num_batches):
            # Step 1: sample domains for this batch
            sampled_domains = rng.choice(
                self.unique_domains,
                size=self.batch_size,
                replace=True,
                p=self.domain_probs,
            )

            batch_indices = []

            # Step 2: for each chosen domain, sample one item index
            for d in sampled_domains:
                if self.replacement:
                    idx = rng.choice(self.domain_to_indices[d])
                else:
                    ptr = domain_ptrs[d]
                    pool = domain_pools[d]

                    if ptr >= len(pool):
                        # reshuffle and restart
                        pool = self.domain_to_indices[d].copy()
                        rng.shuffle(pool)
                        domain_pools[d] = pool
                        domain_ptrs[d] = 0
                        ptr = 0

                    idx = pool[ptr]
                    domain_ptrs[d] += 1

                batch_indices.append(int(idx))

            yield batch_indices

    def __len__(self) -> int:
        return self.num_batches


class MultiDomainItemDataset(Dataset):
    """
    Dataset for semantic ID codebook training.
    Loads content embeddings of items from multiple domains along with their domain IDs,
    which are used for training RQ-VAE or hierarchical RQ-VAE models.
    """
    def __init__(
        self,
        domain_configs: List[Dict[str, str]],
        expected_embed_dim: Optional[int] = None,
        return_item_id: bool = False,
    ):
        """
        Args:
            domain_configs: configuration dictionary for each domain, must contain:
                - 'name': domain name (string)
                - 'embed_path': path to content embedding .npy file
                - 'id_map_path': path to item ID mapping .npy file (optional, set to None if IDs are not required)
                - (optional) 'user_seq_path' not required here
            expected_embed_dim: expected embedding dimension for validation
            return_item_id: whether to return the global item ID in __getitem__
        """
        self.domain_names = []
        self.item_embeds_list = []          # temporarily store embeddings of each domain before merging
        self.domain_ids_list = []            # temporarily store domain labels of each domain
        self.item_global_ids = []             # list of all global item IDs
        self.return_item_id = return_item_id

        global_idx_start = 0
        for cfg in domain_configs:
            domain_name = cfg['name']
            embed_path = cfg['embed_path']
            id_map_path = cfg.get('id_map_path')  # may be None

            # 1. Load embeddings
            if not os.path.exists(embed_path):
                raise FileNotFoundError(f"Embedding file not found: {embed_path}")
            embeds = np.load(embed_path)                # shape (num_items, embed_dim)
            if expected_embed_dim is not None:
                assert embeds.shape[1] == expected_embed_dim, \
                    f"Domain {domain_name} embedding dim {embeds.shape[1]} != expected {expected_embed_dim}"
            num_items = embeds.shape[0]

            # 2. Load ID mapping (if provided)
            if id_map_path is not None and os.path.exists(id_map_path):
                raw_ids = np.load(id_map_path, allow_pickle=True)
                # Handle possible byte strings
                if raw_ids.dtype.type is np.bytes_:
                    raw_ids = np.array([id_.decode('utf-8') for id_ in raw_ids])
                elif raw_ids.dtype.type is np.str_:
                    raw_ids = raw_ids.astype(str)
                if raw_ids.ndim > 1:
                    raw_ids = raw_ids.reshape(-1)
                assert len(raw_ids) == num_items, \
                    f"Domain {domain_name}: #embeds {num_items} != #ids {len(raw_ids)}"
                global_ids = raw_ids
            else:
                # If no ID mapping is provided, use the within-domain index as the ID
                global_ids = [i for i in range(num_items)]

            # 3. Store data of the current domain
            self.domain_names.append(domain_name)
            self.item_embeds_list.append(embeds)
            self.domain_ids_list.append(np.full(num_items, len(self.domain_names)-1, dtype=int))
            self.item_global_ids.extend(global_ids)

        # Merge all embeddings and domain labels
        self.item_embeds = np.vstack(self.item_embeds_list)               # (total_items, embed_dim)
        self.domain_ids = np.concatenate(self.domain_ids_list)            # (total_items,)

        # Validate consistency
        assert len(self.item_global_ids) == self.item_embeds.shape[0] == len(self.domain_ids)

    def __len__(self) -> int:
        return self.item_embeds.shape[0]

    def __getitem__(self, idx: int) -> Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, str]]:
        """
        Returns:
            if return_item_id=False: (embed, domain_id)
            if return_item_id=True:  (embed, domain_id, global_item_id)
        """
        embed = self.item_embeds[idx]
        domain_id = self.domain_ids[idx]
        if self.return_item_id:
            return embed, domain_id, self.item_global_ids[idx]
        else:
            return embed, domain_id

    def get_domain_stats(self) -> Dict[str, int]:
        """Return the item count statistics for each domain."""
        stats = {}
        for i, name in enumerate(self.domain_names):
            stats[name] = np.sum(self.domain_ids == i)
        return stats

def collate_flattened_semantic_tokens(
    batch: List[torch.Tensor],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Collate function that flattens [T, L] into [T*L], then pads.

    Args:
        batch:
            list of LongTensor [T_i, L]
        pad_token_id:
            a special token ID not used by real semantic IDs

    Returns:
        {
            "input_ids": LongTensor [B, S_max],
            "attention_mask": LongTensor [B, S_max],
            "lengths": LongTensor [B],          # flattened lengths
            "item_lengths": LongTensor [B],     # original item counts
        }
    """
    flat_batch = [x.reshape(-1) for x in batch]
    lengths = torch.tensor([x.size(0) for x in flat_batch], dtype=torch.long)
    item_lengths = torch.tensor([x.size(0) for x in batch], dtype=torch.long)

    padded = pad_sequence(
        flat_batch,
        batch_first=True,
        padding_value=pad_token_id,
    )  # [B, S_max]

    attention_mask = (padded != pad_token_id).long()

    return {
        "input_ids": padded,            # [B, S_max]
        "attention_mask": attention_mask,
        "lengths": lengths,             # flattened token lengths
        "item_lengths": item_lengths,   # number of items before flattening
    }