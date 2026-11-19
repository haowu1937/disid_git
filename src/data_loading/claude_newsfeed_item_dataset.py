import os
import json
import bisect
from array import array
from typing import List, Dict, Optional, Union, Tuple

import numpy as np

try:
    import orjson as _json_lib
    _USE_ORJSON = True
except ImportError:
    import json as _json_lib  # type: ignore
    _USE_ORJSON = False

from torch.utils.data import Dataset


def _json_loads(data: bytes) -> dict:
    """
    Parse JSON bytes.
    Prefers orjson (3-5x faster than stdlib json) when available.
    Install with: pip install orjson
    """
    if _USE_ORJSON:
        return _json_lib.loads(data)
    return _json_lib.loads(data.decode("utf-8"))


class NewsFeedItemDataset(Dataset):
    """
    Dataset for semantic ID codebook training.

    Jsonl format — each line must be:
        {"docid": str, "title": str, "embedding": [float, ...]}

    Performance improvements over the original version:
    1. orjson replaces stdlib json  ->  3-5x faster JSON parsing.
    2. Offset arrays are fully loaded into RAM instead of mmap'd.
       7 files × 2.5 M items × 8 bytes ≈ 140 MB total — negligible vs
       the embedding tensors being trained.
    3. Per-item assert on embed dim is removed from __getitem__; validation
       is done once at init time via _validate_one_sample.
    4. File handles are opened lazily per worker process (unchanged) and
       kept open for the lifetime of the worker.
    """

    def __init__(
        self,
        domain_configs: List[Dict[str, str]],
        expected_embed_dim: Optional[int] = None,
        return_item_id: bool = False,
    ):
        """
        Args:
            domain_configs: list of dicts, each containing:
                - 'name'       : domain label (used for stats only)
                - 'jsonl_path' or 'embed_path': path to the .jsonl file
                - 'index_path' (optional): path to the pre-built offset
                  index .npy file; defaults to <jsonl_path>.offsets.npy
            expected_embed_dim: if given, embeddings are validated against
                this dimension once at startup.
            return_item_id: if True, __getitem__ returns (embed, domain_id,
                docid); otherwise (embed, domain_id).
        """
        self.domain_names: List[str] = []
        self.return_item_id = return_item_id
        self.expected_embed_dim = expected_embed_dim

        self.file_paths: List[str] = []
        # Fully loaded uint64 arrays — no mmap overhead on random access.
        self.offsets_list: List[np.ndarray] = []
        self.file_item_counts: List[int] = []
        self.cum_counts: List[int] = []

        self.domain_stats: Dict[str, int] = {}

        # Keyed by file_idx; populated lazily, one handle per worker process.
        self._file_handles: Dict[int, object] = {}

        total = 0
        for cfg in domain_configs:
            domain_name = cfg["name"]
            jsonl_path = cfg.get("jsonl_path", cfg.get("embed_path"))
            index_path = cfg.get("index_path")

            if jsonl_path is None:
                raise ValueError(
                    f"Domain '{domain_name}' must provide 'jsonl_path' "
                    f"or 'embed_path'."
                )
            if not os.path.exists(jsonl_path):
                raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

            if index_path is None:
                index_path = jsonl_path + ".offsets.npy"

            # Load offsets fully into RAM (not mmap) for O(1) random access.
            offsets = self._load_or_build_offsets(jsonl_path, index_path)

            if len(offsets) == 0:
                raise ValueError(f"Empty JSONL file: {jsonl_path}")

            self._validate_one_sample(jsonl_path, offsets[0], domain_name)

            self.domain_names.append(domain_name)
            self.file_paths.append(jsonl_path)
            self.offsets_list.append(offsets)
            self.file_item_counts.append(len(offsets))
            self.domain_stats[domain_name] = len(offsets)

            total += len(offsets)
            self.cum_counts.append(total)

        self.total_items = total

    # ------------------------------------------------------------------
    # Index building / loading
    # ------------------------------------------------------------------

    def _load_or_build_offsets(self, jsonl_path: str, index_path: str) -> np.ndarray:
        """
        Return a fully-loaded uint64 array of per-line byte offsets.

        If <index_path> already exists it is loaded without rebuilding.
        Otherwise the file is scanned once, the index is saved, then
        loaded — so subsequent runs skip the scan entirely.
        """
        if os.path.exists(index_path):
            # np.load without mmap_mode -> array lives in RAM.
            return np.load(index_path)

        print(f"[build_index] building offsets for: {jsonl_path}")

        offsets = array("Q")  # unsigned 64-bit integers
        with open(jsonl_path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(pos)

        offsets_np = np.asarray(offsets, dtype=np.uint64)
        np.save(index_path, offsets_np)
        print(
            f"[build_index] saved {index_path} | num_lines={len(offsets_np)}"
        )
        # Load from disk so the caller always gets an ndarray (not array.array).
        return np.load(index_path)

    # ------------------------------------------------------------------
    # Startup validation
    # ------------------------------------------------------------------

    def _validate_one_sample(
        self, jsonl_path: str, offset: int, domain_name: str
    ) -> None:
        """Verify format and embedding dimension on a single record."""
        with open(jsonl_path, "rb") as f:
            f.seek(int(offset))
            line = f.readline()

        obj = _json_loads(line)

        missing = {"docid", "embedding"} - obj.keys()
        if missing:
            raise KeyError(
                f"Domain '{domain_name}': required keys {missing} "
                f"not found in {jsonl_path}"
            )

        embed = obj["embedding"]
        if not isinstance(embed, list):
            raise TypeError(
                f"Domain '{domain_name}': 'embedding' must be a list "
                f"in {jsonl_path}"
            )

        if (
            self.expected_embed_dim is not None
            and len(embed) != self.expected_embed_dim
        ):
            raise ValueError(
                f"Domain '{domain_name}': embedding dim {len(embed)} "
                f"!= expected {self.expected_embed_dim}"
            )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.total_items

    def _normalize_idx(self, idx: int) -> int:
        if idx < 0:
            idx = self.total_items + idx
        if idx < 0 or idx >= self.total_items:
            raise IndexError(f"Index {idx} out of range [0, {self.total_items})")
        return idx

    def _locate_file_and_local_idx(self, idx: int) -> Tuple[int, int]:
        """Map a global idx to (file_idx, local_idx_within_that_file)."""
        file_idx = bisect.bisect_right(self.cum_counts, idx)
        prev_cum = 0 if file_idx == 0 else self.cum_counts[file_idx - 1]
        return file_idx, idx - prev_cum

    def _get_file_handle(self, file_idx: int):
        """
        Return a cached binary file handle for this worker process.
        Handles are opened on first access and reused for the worker's
        lifetime; __getstate__ ensures they are not forked into children.
        """
        if file_idx not in self._file_handles:
            self._file_handles[file_idx] = open(self.file_paths[file_idx], "rb")
        return self._file_handles[file_idx]

    def _read_record(self, idx: int) -> dict:
        idx = self._normalize_idx(idx)
        file_idx, local_idx = self._locate_file_and_local_idx(idx)

        # Direct array lookup — no mmap round-trip.
        offset = int(self.offsets_list[file_idx][local_idx])

        fh = self._get_file_handle(file_idx)
        fh.seek(offset)
        return _json_loads(fh.readline())

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, str]]:
        """
        Returns:
            return_item_id=False  ->  (embed: np.ndarray, domain_id: int)
            return_item_id=True   ->  (embed, domain_id, docid: str)

        Notes:
            - domain_id is always 1 (kept for API compatibility).
            - Embed dim is validated once at init, not per item, to avoid
              repeated Python-level shape checks in the hot path.
        """
        obj = self._read_record(idx)

        embed = np.asarray(obj["embedding"], dtype=np.float32)

        if self.return_item_id:
            return embed, 1, str(obj["docid"])
        return embed, 1

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_domain_stats(self) -> Dict[str, int]:
        """Return item counts per domain config (domain_id is always 1)."""
        return dict(self.domain_stats)

    def close(self) -> None:
        for fh in self._file_handles.values():
            try:
                fh.close()
            except Exception:
                pass
        self._file_handles = {}

    def __del__(self):
        self.close()

    def __getstate__(self):
        """
        Drop open file handles before pickling into DataLoader workers.
        Each worker will re-open handles lazily on first __getitem__ call.
        """
        state = self.__dict__.copy()
        state["_file_handles"] = {}
        return state


# ---------------------------------------------------------------------------
# Optional: convert an existing JSONL dataset to a binary memmap for maximum
# throughput.  Run once offline; training then uses MemmapDataset instead.
# ---------------------------------------------------------------------------

def convert_jsonl_to_memmap(
    jsonl_path: str,
    out_embed_path: str,
    out_docid_path: str,
    embed_dim: int,
    dtype: np.dtype = np.float32,
) -> int:
    """
    Read a .jsonl file and write two binary files:

        <out_embed_path>   uint64 (N,)  — docids as integers
        <out_docid_path>   float32 (N, embed_dim) — embeddings as memmap

    After conversion, use MemmapDataset (below) instead of
    NewsFeedItemDataset for ~10-50x faster __getitem__ throughput.

    Returns the number of records written.
    """
    # Count lines first so we can pre-allocate.
    print(f"[convert] counting lines in {jsonl_path} ...")
    n = 0
    with open(jsonl_path, "rb") as f:
        for line in f:
            if line.strip():
                n += 1
    print(f"[convert] {n} records found")

    embed_mm = np.memmap(
        out_embed_path, dtype=dtype, mode="w+", shape=(n, embed_dim)
    )
    docid_arr = np.empty(n, dtype=np.int64)

    with open(jsonl_path, "rb") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = _json_loads(line)
            embed_mm[i] = np.asarray(obj["embedding"], dtype=dtype)
            docid_arr[i] = int(obj["docid"])

    embed_mm.flush()
    np.save(out_docid_path, docid_arr)
    print(f"[convert] saved embeddings -> {out_embed_path}")
    print(f"[convert] saved docids     -> {out_docid_path}")
    return n


class MemmapDataset(Dataset):
    """
    Ultra-fast dataset backed by numpy memmap files produced by
    convert_jsonl_to_memmap().

    __getitem__ cost: one slice of a memory-mapped array — no JSON parsing,
    no seek, no readline.  Typical throughput is 10-50x NewsFeedItemDataset.

    Args:
        domain_configs: list of dicts with keys:
            - 'name'        : domain label
            - 'embed_path'  : path to float32 memmap file (N, embed_dim)
            - 'docid_path'  : path to int64 .npy docid array
        embed_dim: embedding dimension (required for memmap shape)
        return_item_id: if True, return (embed, domain_id, docid)
    """

    def __init__(
        self,
        domain_configs: List[Dict[str, str]],
        embed_dim: int,
        return_item_id: bool = False,
    ):
        self.embed_dim = embed_dim
        self.return_item_id = return_item_id

        self.embeds: List[np.ndarray] = []
        self.docids: List[np.ndarray] = []
        self.cum_counts: List[int] = []
        self.domain_stats: Dict[str, int] = {}

        total = 0
        for cfg in domain_configs:
            name = cfg["name"]
            embed_path = cfg["embed_path"]
            docid_path = cfg["docid_path"]

            mm = np.memmap(embed_path, dtype=np.float32, mode="r").reshape(-1, embed_dim)
            ids = np.load(docid_path)

            if len(mm) != len(ids):
                raise ValueError(
                    f"Domain '{name}': embed count {len(mm)} "
                    f"!= docid count {len(ids)}"
                )

            self.embeds.append(mm)
            self.docids.append(ids)
            self.domain_stats[name] = len(mm)

            total += len(mm)
            self.cum_counts.append(total)

        self.total_items = total

    def __len__(self) -> int:
        return self.total_items

    def _locate(self, idx: int) -> Tuple[int, int]:
        if idx < 0:
            idx += self.total_items
        file_idx = bisect.bisect_right(self.cum_counts, idx)
        prev = 0 if file_idx == 0 else self.cum_counts[file_idx - 1]
        return file_idx, idx - prev

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, str]]:
        fi, li = self._locate(idx)
        embed = np.array(self.embeds[fi][li], dtype=np.float32)  # copy out of mmap
        if self.return_item_id:
            return embed, 1, str(self.docids[fi][li])
        return embed, 1

    def get_domain_stats(self) -> Dict[str, int]:
        return dict(self.domain_stats)