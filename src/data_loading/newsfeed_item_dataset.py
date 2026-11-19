import os
import bisect
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset


class NewsFeedItemDataset(Dataset):
    """
    Dataset for semantic ID codebook training on parquet files.

    Expected parquet schema:
        - docid: string
        - embedding: fixed_size_list<float32>[D]

    Notes:
        - domain_id is fixed to 1
        - return_item_id=True returns docid
        - supports:
            1) row-group lazy loading
            2) load all parquet data into memory
    """

    def __init__(
        self,
        domain_configs: List[Dict[str, str]],
        expected_embed_dim: Optional[int] = None,
        return_item_id: bool = False,
        load_all_to_memory: bool = False,
    ):
        """
        Args:
            domain_configs: configuration dictionary for each shard/domain, must contain:
                - 'name': shard/domain name
                - 'parquet_path': path to parquet file
                  or reuse 'embed_path' to point to parquet file
            expected_embed_dim: expected embedding dimension
            return_item_id: whether to return docid in __getitem__
            load_all_to_memory: if True, load all parquet data into RAM at init
        """
        self.domain_names: List[str] = []
        self.return_item_id = return_item_id
        self.expected_embed_dim = expected_embed_dim
        self.load_all_to_memory = load_all_to_memory

        self.file_paths: List[str] = []
        self.file_row_counts: List[int] = []
        self.file_cum_counts: List[int] = []

        # per file: cumulative row counts of row groups
        self.file_row_group_cum_counts: List[List[int]] = []

        self.domain_stats: Dict[str, int] = {}

        # lazy-opened per worker/process
        self._pf_cache: Dict[int, pq.ParquetFile] = {}

        # LRU cache for row groups:
        # key = (file_idx, row_group_idx)
        # value = {"embeddings": np.ndarray, "docids": Optional[List[str]]}
        self._row_group_cache: "OrderedDict[Tuple[int, int], Dict[str, object]]" = OrderedDict()
        self._row_group_cache_size = 2

        # in-memory storage
        self._in_memory_embeddings: Optional[np.ndarray] = None
        self._in_memory_docids: Optional[np.ndarray] = None

        total_items = 0
        inferred_dim = expected_embed_dim

        for cfg in domain_configs:
            domain_name = cfg["name"]
            parquet_path = cfg.get("parquet_path", cfg.get("embed_path"))

            if parquet_path is None:
                raise ValueError(
                    f"Domain {domain_name} must provide 'parquet_path' "
                    f"or reuse 'embed_path' to point to the parquet file."
                )
            if not os.path.exists(parquet_path):
                raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

            pf = pq.ParquetFile(
                parquet_path,
                memory_map=True,
                pre_buffer=False,
            )

            schema = pf.schema_arrow
            self._validate_schema(schema, parquet_path)

            num_rows = pf.metadata.num_rows
            if num_rows <= 0:
                raise ValueError(f"Empty parquet file: {parquet_path}")

            file_embed_dim = self._get_embed_dim_from_schema(schema)
            if inferred_dim is None:
                inferred_dim = file_embed_dim
            elif file_embed_dim is not None and file_embed_dim != inferred_dim:
                raise ValueError(
                    f"Embedding dim mismatch in {parquet_path}: "
                    f"{file_embed_dim} != expected {inferred_dim}"
                )

            rg_cum_counts = []
            running = 0
            for rg_idx in range(pf.metadata.num_row_groups):
                rg_rows = pf.metadata.row_group(rg_idx).num_rows
                running += rg_rows
                rg_cum_counts.append(running)

            if running != num_rows:
                raise ValueError(
                    f"Row-group row count mismatch in {parquet_path}: "
                    f"sum(row_groups)={running}, num_rows={num_rows}"
                )

            self.domain_names.append(domain_name)
            self.file_paths.append(parquet_path)
            self.file_row_counts.append(num_rows)
            self.file_row_group_cum_counts.append(rg_cum_counts)
            self.domain_stats[domain_name] = num_rows

            total_items += num_rows
            self.file_cum_counts.append(total_items)

        self.total_items = total_items
        self.expected_embed_dim = inferred_dim

        if self.expected_embed_dim is None:
            raise ValueError(
                "Cannot infer embedding dim from parquet schema. "
                "Please set expected_embed_dim explicitly."
            )

        if self.load_all_to_memory:
            self._load_all_parquets_to_memory()

    def _validate_schema(self, schema: pa.Schema, parquet_path: str) -> None:
        if schema.get_field_index("embedding") < 0:
            raise KeyError(f"'embedding' column not found in {parquet_path}")

        if self.return_item_id and schema.get_field_index("docid") < 0:
            raise KeyError(f"'docid' column not found in {parquet_path}")

        emb_field = schema.field("embedding")
        emb_type = emb_field.type

        if not (
            pa.types.is_fixed_size_list(emb_type)
            or pa.types.is_list(emb_type)
            or pa.types.is_large_list(emb_type)
        ):
            raise TypeError(
                f"'embedding' must be a list/fixed_size_list column, "
                f"but got {emb_type} in {parquet_path}"
            )

        if pa.types.is_fixed_size_list(emb_type):
            dim = emb_type.list_size
            if self.expected_embed_dim is not None and dim != self.expected_embed_dim:
                raise ValueError(
                    f"Embedding dim mismatch in {parquet_path}: "
                    f"{dim} != expected {self.expected_embed_dim}"
                )

    def _get_embed_dim_from_schema(self, schema: pa.Schema) -> Optional[int]:
        emb_type = schema.field("embedding").type
        if pa.types.is_fixed_size_list(emb_type):
            return emb_type.list_size
        return None

    def __len__(self) -> int:
        return self.total_items

    def _normalize_idx(self, idx: int) -> int:
        idx = int(idx)
        if idx < 0:
            idx += self.total_items
        if idx < 0 or idx >= self.total_items:
            raise IndexError(f"Index out of range: {idx}")
        return idx

    def _locate_file_and_local_idx(self, idx: int) -> Tuple[int, int]:
        file_idx = bisect.bisect_right(self.file_cum_counts, idx)
        prev_cum = 0 if file_idx == 0 else self.file_cum_counts[file_idx - 1]
        local_idx = idx - prev_cum
        return file_idx, local_idx

    def _locate_row_group_and_local_idx(self, file_idx: int, local_idx: int) -> Tuple[int, int]:
        rg_cum = self.file_row_group_cum_counts[file_idx]
        row_group_idx = bisect.bisect_right(rg_cum, local_idx)
        prev_cum = 0 if row_group_idx == 0 else rg_cum[row_group_idx - 1]
        row_idx_in_rg = local_idx - prev_cum
        return row_group_idx, row_idx_in_rg

    def _get_parquet_file(self, file_idx: int) -> pq.ParquetFile:
        if file_idx not in self._pf_cache:
            self._pf_cache[file_idx] = pq.ParquetFile(
                self.file_paths[file_idx],
                memory_map=True,
                pre_buffer=False,
            )
        return self._pf_cache[file_idx]

    def _combine_column(self, chunked_array: pa.ChunkedArray) -> pa.Array:
        if chunked_array.num_chunks == 1:
            return chunked_array.chunk(0)
        return chunked_array.combine_chunks()

    def _embedding_array_to_numpy(self, emb_array: pa.Array) -> np.ndarray:
        emb_type = emb_array.type

        if pa.types.is_fixed_size_list(emb_type):
            dim = emb_type.list_size
            flat = emb_array.values.to_numpy(zero_copy_only=False)
            start = emb_array.offset * dim
            end = start + len(emb_array) * dim
            emb = flat[start:end].reshape(len(emb_array), dim)
            return emb.astype(np.float32, copy=False)

        if pa.types.is_list(emb_type) or pa.types.is_large_list(emb_type):
            emb = np.asarray(emb_array.to_pylist(), dtype=np.float32)
            if emb.ndim != 2:
                raise ValueError(f"embedding array must be 2D, got shape={emb.shape}")
            if self.expected_embed_dim is not None and emb.shape[1] != self.expected_embed_dim:
                raise ValueError(
                    f"Embedding dim mismatch: {emb.shape[1]} != expected {self.expected_embed_dim}"
                )
            return emb

        raise TypeError(f"Unsupported embedding type: {emb_type}")

    def _read_full_file(self, file_idx: int) -> Dict[str, object]:
        pf = pq.ParquetFile(
            self.file_paths[file_idx],
            memory_map=True,
            pre_buffer=False,
        )

        columns = ["embedding"]
        if self.return_item_id:
            columns.append("docid")

        table = pf.read(columns=columns, use_threads=True)

        emb_array = self._combine_column(table.column("embedding"))
        embeddings = self._embedding_array_to_numpy(emb_array)

        result = {"embeddings": embeddings, "docids": None}

        if self.return_item_id:
            docid_array = self._combine_column(table.column("docid"))
            result["docids"] = np.asarray(docid_array.to_pylist(), dtype=object)

        return result

    def _load_all_parquets_to_memory(self) -> None:
        all_embeddings = []
        all_docids = [] if self.return_item_id else None

        for file_idx in range(len(self.file_paths)):
            loaded = self._read_full_file(file_idx)
            all_embeddings.append(loaded["embeddings"])

            if self.return_item_id:
                all_docids.append(loaded["docids"])

        self._in_memory_embeddings = np.concatenate(all_embeddings, axis=0)

        if self._in_memory_embeddings.shape != (self.total_items, self.expected_embed_dim):
            raise ValueError(
                f"In-memory embedding shape mismatch: "
                f"{self._in_memory_embeddings.shape} != "
                f"({self.total_items}, {self.expected_embed_dim})"
            )

        if self.return_item_id:
            self._in_memory_docids = np.concatenate(all_docids, axis=0)
            if len(self._in_memory_docids) != self.total_items:
                raise ValueError(
                    f"In-memory docid count mismatch: "
                    f"{len(self._in_memory_docids)} != {self.total_items}"
                )

        # once fully loaded, row-group cache becomes unnecessary
        self._row_group_cache.clear()
        self.close()

    def _load_row_group(self, file_idx: int, row_group_idx: int) -> Dict[str, object]:
        key = (file_idx, row_group_idx)

        if key in self._row_group_cache:
            self._row_group_cache.move_to_end(key)
            return self._row_group_cache[key]

        pf = self._get_parquet_file(file_idx)

        columns = ["embedding"]
        if self.return_item_id:
            columns.append("docid")

        table = pf.read_row_group(
            row_group_idx,
            columns=columns,
            use_threads=True,
        )

        emb_array = self._combine_column(table.column("embedding"))
        embeddings = self._embedding_array_to_numpy(emb_array)

        cached = {
            "embeddings": embeddings,
            "docids": None,
        }

        if self.return_item_id:
            docid_array = self._combine_column(table.column("docid"))
            cached["docids"] = docid_array.to_pylist()

        self._row_group_cache[key] = cached
        self._row_group_cache.move_to_end(key)

        while len(self._row_group_cache) > self._row_group_cache_size:
            self._row_group_cache.popitem(last=False)

        return cached

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, str]]:
        """
        Returns:
            if return_item_id=False: (embed, domain_id)
            if return_item_id=True:  (embed, domain_id, docid)
        """
        idx = self._normalize_idx(idx)
        domain_id = 1

        if self.load_all_to_memory:
            embed = self._in_memory_embeddings[idx]
            if self.return_item_id:
                docid = self._in_memory_docids[idx]
                return embed, domain_id, docid
            return embed, domain_id

        file_idx, local_idx = self._locate_file_and_local_idx(idx)
        row_group_idx, row_idx_in_rg = self._locate_row_group_and_local_idx(file_idx, local_idx)

        cached = self._load_row_group(file_idx, row_group_idx)

        embed = cached["embeddings"][row_idx_in_rg]

        if self.return_item_id:
            docid = cached["docids"][row_idx_in_rg]
            return embed, domain_id, docid
        else:
            return embed, domain_id

    def get_domain_stats(self) -> Dict[str, int]:
        return dict(self.domain_stats)

    def close(self):
        for pf in self._pf_cache.values():
            close_fn = getattr(pf, "close", None)
            if callable(close_fn):
                close_fn()
        self._pf_cache = {}

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pf_cache"] = {}
        state["_row_group_cache"] = OrderedDict()
        return state