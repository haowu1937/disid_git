import glob
import os
from typing import Any, Dict, List, Optional

import torch
import tfrecord
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


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
        "input_ids": padded,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "item_lengths": item_lengths,
    }


def collate_semantic_item_sequences(
    batch: List[torch.Tensor],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for samples shaped [T, L].

    Args:
        batch:
            list of LongTensor [T_i, L]
        pad_token_id:
            a special token ID not used by real semantic IDs

    Returns:
        {
            "semantic_ids": LongTensor [B, T_max, L],
            "item_attention_mask": LongTensor [B, T_max],
            "lengths": LongTensor [B],
        }
    """
    if len(batch) == 0:
        raise ValueError("Empty batch.")

    batch_size = len(batch)
    sid_width = batch[0].size(1)
    lengths = torch.tensor([x.size(0) for x in batch], dtype=torch.long)
    max_t = int(lengths.max().item())

    padded = torch.full(
        (batch_size, max_t, sid_width),
        fill_value=pad_token_id,
        dtype=torch.long,
    )

    item_attention_mask = torch.zeros(batch_size, max_t, dtype=torch.long)

    for i, x in enumerate(batch):
        t = x.size(0)
        padded[i, :t] = x
        item_attention_mask[i, :t] = 1

    return {
        "semantic_ids": padded,                # [B, T_max, L]
        "item_attention_mask": item_attention_mask,  # [B, T_max]
        "lengths": lengths,                    # [B]
    }


def collate_next_item_semantic_batch(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Collate for next-item prediction.

    Returns:
        {
            "history_semantic_ids": [B, T_max, L],
            "history_attention_mask": [B, T_max],
            "history_lengths": [B],
            "target_semantic_ids": [B, L],
        }
    """
    histories = [x["history_semantic_ids"] for x in batch]
    targets = [x["target_semantic_ids"] for x in batch]

    hist_batch = collate_semantic_item_sequences(
        histories,
        pad_token_id=pad_token_id,
    )

    target_tensor = torch.stack(targets, dim=0)  # [B, L]

    return {
        "history_semantic_ids": hist_batch["semantic_ids"],
        "history_attention_mask": hist_batch["item_attention_mask"],
        "history_lengths": hist_batch["lengths"],
        "target_semantic_ids": target_tensor,
    }


def collate_next_item_t5_batch(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Collate function used by T5 next-item training.

    It first pads history sequences as [B, T_max, L], then flattens them
    into encoder input [B, T_max * L].

    Returns:
        {
            "input_ids": LongTensor [B, S],
            "attention_mask": LongTensor [B, S],
            "history_lengths": LongTensor [B],   # number of history items
            "target_semantic_ids": LongTensor [B, L],
            "sid_width": LongTensor [],          # scalar tensor
        }
    """
    batch_dict = collate_next_item_semantic_batch(
        batch=batch,
        pad_token_id=pad_token_id,
    )

    history_semantic_ids = batch_dict["history_semantic_ids"]  # [B, T_max, L]
    targets = batch_dict["target_semantic_ids"]                # [B, L]

    batch_size, t_max, sid_width = history_semantic_ids.shape
    flat_history = history_semantic_ids.reshape(batch_size, t_max * sid_width)
    attention_mask = (flat_history != pad_token_id).long()

    return {
        "input_ids": flat_history,
        "attention_mask": attention_mask,
        "history_lengths": batch_dict["history_lengths"],
        "target_semantic_ids": targets,
        "sid_width": torch.tensor(sid_width, dtype=torch.long),
    }


class AmazonSeqDataset(Dataset):
    """
    Amazon user behavior sequence dataset for downstream semantic-ID models.

    This dataset reads Amazon TFRecord gzip files, extracts `sequence_data`
    from each example, and maps raw item IDs to semantic ID tokens using
    a pre-generated semantic codebook.

    Each sample is a user behavior sequence:
        raw item sequence: [item_1, item_2, ..., item_T]

    After mapping with the semantic codebook:
        semantic sequence: [[sid_11, sid_12, ..., sid_1L],
                            [sid_21, sid_22, ..., sid_2L],
                            ...
                            [sid_T1, sid_T2, ..., sid_TL]]

    Output of __getitem__:
        LongTensor of shape [T, L]
    where:
        T = number of interacted items in the sequence
        L = semantic ID width for each item
    """

    def __init__(
        self,
        tfrecord_dir: str,
        semantic_id_pt_path: str,
        tfrecord_pattern: str = "partition_*.tfrecord.gz",
        skip_unknown_items: bool = False,
        min_seq_len: int = 1,
        max_seq_len: Optional[int] = None,
        drop_last_item_for_target: bool = False,
    ):
        """
        Args:
            tfrecord_dir:
                Directory containing TFRecord gzip files.
                Example: ./data/beauty/training

            semantic_id_pt_path:
                Path to semantic_ids_all.pt file.

            tfrecord_pattern:
                File glob pattern under tfrecord_dir.

            skip_unknown_items:
                If True, items not found in the codebook are skipped.
                If False, a missing item raises KeyError.

            min_seq_len:
                Minimum mapped sequence length to keep.

            max_seq_len:
                If not None, truncate each sequence to the last max_seq_len items.

            drop_last_item_for_target:
                If True, __getitem__ returns only the prefix sequence [:-1].
                This is useful if you want to reserve the last item as prediction target.
                In that case you would typically use a dedicated dataset variant,
                but this switch can help quick experiments.
        """
        super().__init__()
        self.tfrecord_dir = tfrecord_dir
        self.semantic_id_pt_path = semantic_id_pt_path
        self.skip_unknown_items = skip_unknown_items
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.drop_last_item_for_target = drop_last_item_for_target

        obj = torch.load(semantic_id_pt_path, map_location="cpu")

        required_keys = ["semantic_id", "item_id"]
        for key in required_keys:
            if key not in obj:
                raise KeyError(
                    f"Missing key '{key}' in semantic codebook file: {semantic_id_pt_path}"
                )

        semantic_id = obj["semantic_id"].long()   # [N, L]
        item_id = obj["item_id"].long()           # [N]

        if semantic_id.ndim != 2:
            raise ValueError(
                f"Expected semantic_id to have shape [N, L], got {tuple(semantic_id.shape)}"
            )
        if item_id.ndim != 1:
            raise ValueError(
                f"Expected item_id to have shape [N], got {tuple(item_id.shape)}"
            )
        if semantic_id.size(0) != item_id.size(0):
            raise ValueError(
                f"semantic_id and item_id length mismatch: "
                f"{semantic_id.size(0)} vs {item_id.size(0)}"
            )

        self.sid_width = int(semantic_id.size(1))
        self.num_items = int(item_id.size(0))
        self.max_sid_token = int(semantic_id.max().item())
        self.sid_vocab_size = self.max_sid_token + 1

        self.item_to_sid: Dict[int, torch.Tensor] = {}
        for i in range(self.num_items):
            self.item_to_sid[int(item_id[i].item())] = semantic_id[i].clone()

        pattern = os.path.join(tfrecord_dir, tfrecord_pattern)
        self.tfrecord_files = sorted(glob.glob(pattern))
        if not self.tfrecord_files:
            raise FileNotFoundError(f"No TFRecord files found with pattern: {pattern}")

        self.samples: List[torch.Tensor] = []
        self.raw_lengths: List[int] = []
        self.mapped_lengths: List[int] = []

        description = {
            "sequence_data": "int",
        }

        for path in self.tfrecord_files:
            loader = tfrecord.tfrecord_loader(
                data_path=path,
                index_path=None,
                description=description,
                compression_type="gzip",
            )

            for record in loader:
                raw_seq = record.get("sequence_data", None)
                if raw_seq is None:
                    continue

                raw_seq = [int(x) for x in raw_seq]
                if len(raw_seq) == 0:
                    continue

                if self.max_seq_len is not None:
                    raw_seq = raw_seq[-self.max_seq_len:]

                sid_seq = self._map_item_sequence_to_sid(raw_seq)
                if sid_seq is None:
                    continue

                if self.drop_last_item_for_target:
                    if sid_seq.size(0) < 2:
                        continue
                    sid_seq = sid_seq[:-1]

                if sid_seq.size(0) < self.min_seq_len:
                    continue

                self.samples.append(sid_seq)  # [T, L]
                self.raw_lengths.append(len(raw_seq))
                self.mapped_lengths.append(int(sid_seq.size(0)))

        if len(self.samples) == 0:
            raise RuntimeError(
                "No valid samples loaded. Please check TFRecord contents and codebook coverage."
            )

    def _map_item_sequence_to_sid(self, item_seq: List[int]) -> Optional[torch.Tensor]:
        """
        Map raw item IDs to semantic IDs.

        Args:
            item_seq: list of raw item IDs

        Returns:
            LongTensor [T, L], or None if sequence becomes empty
        """
        sid_rows: List[torch.Tensor] = []

        for item in item_seq:
            if item in self.item_to_sid:
                sid_rows.append(self.item_to_sid[item])
            else:
                if self.skip_unknown_items:
                    continue
                raise KeyError(
                    f"Item ID {item} not found in semantic codebook: {self.semantic_id_pt_path}"
                )

        if len(sid_rows) == 0:
            return None

        return torch.stack(sid_rows, dim=0).long()  # [T, L]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            LongTensor [T, L]
        """
        return self.samples[idx]

    def get_pad_token_id(self) -> int:
        """
        Returns a pad token ID that does not conflict with real semantic IDs.
        """
        return self.sid_vocab_size

    def summary(self) -> Dict[str, Any]:
        return {
            "num_samples": len(self.samples),
            "sid_width": self.sid_width,
            "sid_vocab_size": self.sid_vocab_size,
            "pad_token_id": self.get_pad_token_id(),
            "min_mapped_len": min(self.mapped_lengths),
            "max_mapped_len": max(self.mapped_lengths),
            "avg_mapped_len": sum(self.mapped_lengths) / len(self.mapped_lengths),
        }


class AmazonNextItemSemanticDataset(Dataset):
    """
    Wrapper dataset for next-item prediction.

    Given a base dataset that returns [T, L], this wrapper builds:
        encoder_input = sequence[:-1]
        target = sequence[-1]

    Output:
        {
            "history_semantic_ids": [T-1, L],
            "target_semantic_ids": [L],
        }
    """

    def __init__(self, base_dataset: AmazonSeqDataset):
        self.base_dataset = base_dataset
        self.valid_indices = [
            i for i in range(len(base_dataset))
            if base_dataset[i].size(0) >= 2
        ]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.base_dataset[self.valid_indices[idx]]  # [T, L]
        return {
            "history_semantic_ids": seq[:-1],  # [T-1, L]
            "target_semantic_ids": seq[-1],    # [L]
        }