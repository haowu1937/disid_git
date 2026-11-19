import glob
import os
from typing import Dict, List, Optional

import torch
import tfrecord
from torch.utils.data import Dataset

from src.data_loading.semantic_dataset import collate_next_item_t5_batch


class NextItemEvalDataset(Dataset):
    """
    Evaluation dataset for next-item prediction.

    Each sample returns:
        {
            "history_semantic_ids": LongTensor [T-1, L],
            "target_semantic_ids": LongTensor [L],
            "target_item_id": LongTensor [],
        }

    This dataset keeps the raw target item id so we can compute item-level
    Recall/NDCG after mapping predicted semantic IDs back to candidate items.
    """

    def __init__(
        self,
        tfrecord_dir: str,
        semantic_id_pt_path: str,
        tfrecord_pattern: str = "partition_*.tfrecord.gz",
        skip_unknown_items: bool = False,
        min_seq_len: int = 2,
        max_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.tfrecord_dir = tfrecord_dir
        self.semantic_id_pt_path = semantic_id_pt_path
        self.skip_unknown_items = skip_unknown_items
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

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

        self.samples: List[Dict[str, torch.Tensor]] = []

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

                mapped_raw_items: List[int] = []
                mapped_sid_rows: List[torch.Tensor] = []

                for item in raw_seq:
                    if item in self.item_to_sid:
                        mapped_raw_items.append(item)
                        mapped_sid_rows.append(self.item_to_sid[item])
                    else:
                        if self.skip_unknown_items:
                            continue
                        raise KeyError(
                            f"Item ID {item} not found in semantic codebook: {semantic_id_pt_path}"
                        )

                if len(mapped_sid_rows) < self.min_seq_len:
                    continue

                sid_seq = torch.stack(mapped_sid_rows, dim=0).long()  # [T, L]

                self.samples.append(
                    {
                        "history_semantic_ids": sid_seq[:-1],                  # [T-1, L]
                        "target_semantic_ids": sid_seq[-1],                    # [L]
                        "target_item_id": torch.tensor(mapped_raw_items[-1]),   # []
                    }
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                "No valid evaluation samples loaded. Please check TFRecord contents and codebook coverage."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]

    def get_pad_token_id(self) -> int:
        return self.sid_vocab_size


def collate_next_item_eval_t5_batch(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Returns:
        {
            "input_ids": LongTensor [B, S],
            "attention_mask": LongTensor [B, S],
            "history_lengths": LongTensor [B],
            "target_semantic_ids": LongTensor [B, L],
            "sid_width": LongTensor [],
            "target_item_ids": LongTensor [B],
        }
    """
    t5_batch = collate_next_item_t5_batch(
        batch=batch,
        pad_token_id=pad_token_id,
    )

    target_item_ids = torch.stack([x["target_item_id"] for x in batch], dim=0).long()

    t5_batch["target_item_ids"] = target_item_ids
    return t5_batch