# ID Mapping Pipeline Info

## 0. Final Goal

The goal of the whole data pipeline is to make the item ID space consistent between:

- `TFRecords["sequence_data"]`
- `semantic_ids_all.pt["item_id"]`
- `semantic_ids_domain_<domain_id>.pt["item_id"]`

The final expected ID space is:

```text
global incremental item id
```

Example:

```text
QB:   0 - 6120
TN:   6121 - 9454
Bili: 9455 - 147487
```

Therefore, the final aligned relation should be:

```text
TFRecord sequence_data item id
==
semantic_ids_all.pt["item_id"]
==
shard_00000_ids_incremental.npy[row_index]
```

---

## 1. Embedding Generation Stage

### Input

The embedding script reads the item CSV. The CSV contains at least:

```text
column 0: id
column 1: zh
```

The script extracts the numeric part of `id` as `raw_item_id`.

Example:

```text
"v613"      -> 613
"613"       -> 613
"item_613"  -> 613
```

### Output

For each domain, the script saves:

```text
CLIP_emb/
  shard_00000_ids.npy
  shard_00000_img.npy
  shard_00000_txt.npy
  shard_00000_fused.npy
```

The key mapping is:

```text
shard_00000_ids.npy[local_new_item_id] = raw_item_id
```

The embedding matrix is row-aligned with `shard_00000_ids.npy`:

```text
shard_00000_fused.npy[local_new_item_id]
corresponds to
shard_00000_ids.npy[local_new_item_id]
```

So at this stage:

```text
local_new_item_id == embedding row index
```

Example:

```text
shard_00000_ids.npy[0] = 613
shard_00000_fused.npy[0] = embedding of raw item 613
```

---

## 2. Initial TFRecord Construction Stage

### Input

The behaviour file contains raw item IDs, often with a `v` prefix:

```text
u100094 v671403 v1649 v13435 ...
```

The parser converts them into raw item ID strings:

```text
v671403 -> "671403"
v1649   -> "1649"
```

### Mapping

The script reads:

```text
shard_00000_ids.npy
```

and builds the reverse mapping:

```text
raw_item_id -> local_new_item_id
```

because:

```text
shard_00000_ids.npy[local_new_item_id] = raw_item_id
```

Example:

```text
shard_00000_ids.npy[0] = 613
shard_00000_ids.npy[1] = 991
```

Then:

```text
613 -> 0
991 -> 1
```

### Output

The initial TFRecord writes:

```text
sequence_data = [local_new_item_id, ...]
```

At this stage:

```text
TFRecord sequence_data is local_new_item_id / embedding row index
```

Example:

```text
raw behaviour item v613
    -> raw_item_id 613
    -> local_new_item_id 0
    -> TFRecord sequence_data stores 0
```

### Note on Embedding Feature

If the initial TFRecord also saves an `embedding` feature, it is correct to use:

```python
item_embeds[local_new_item_id]
```

because `local_new_item_id` is the row index of `shard_00000_fused.npy`.

---

## 3. Incremental ID Generation Stage

### Purpose

The initial `local_new_item_id` is only unique inside a single domain.

To train or evaluate across multiple domains, each domain needs a non-overlapping global ID range.

### Generated File

For each domain, generate:

```text
shard_00000_ids_incremental.npy
```

The meaning is:

```text
shard_00000_ids_incremental.npy[local_new_item_id] = global_incremental_item_id
```

Example for Bili:

```text
shard_00000_ids_incremental.npy[0] = 9455
shard_00000_ids_incremental.npy[1] = 9456
...
```

Therefore:

```text
Bili local_new_item_id 0 -> global_incremental_item_id 9455
Bili local_new_item_id 1 -> global_incremental_item_id 9456
```

### Domain Ranges

Current intended domain layout:

```text
QB:   0 - 6120
TN:   6121 - 9454
Bili: 9455 - 147487
```

This means:

```text
QB item count   = 6121
TN item count   = 3334
Bili item count = 138033
```

Important check:

```text
len(shard_00000_ids_incremental.npy)
must equal
len(shard_00000_ids.npy)
must equal
shard_00000_fused.npy.shape[0]
```

---

## 4. TFRecord Remapping Stage

### Input

The input TFRecords currently contain:

```text
sequence_data = [local_new_item_id, ...]
```

### Correct Mapping

The remapping script should convert:

```text
local_new_item_id -> global_incremental_item_id
```

The correct map is:

```python
id_map = {
    local_id: global_id
    for local_id, global_id in enumerate(shard_00000_ids_incremental.npy)
}
```

Equivalent meaning:

```text
id_map[0] = 9455
id_map[1] = 9456
...
```

### Incorrect Mapping to Avoid

Do not build the map as:

```python
zip(shard_00000_ids.npy, shard_00000_ids_incremental.npy)
```

because that produces:

```text
raw_item_id -> global_incremental_item_id
```

This is wrong if the input TFRecord already stores `local_new_item_id`.

### Correct Output

After remapping, the output TFRecords should contain:

```text
sequence_data = [global_incremental_item_id, ...]
```

Example:

```text
input TFRecord sequence_data:  [0, 1, 2]
Bili incremental ids:          [9455, 9456, 9457, ...]
output TFRecord sequence_data: [9455, 9456, 9457]
```

### Important Schema Note

The current remapping script rebuilds each TFRecord with only:

```text
sequence_data
```

This is acceptable if the downstream loader only needs `sequence_data`.

If the downstream loader also needs:

```text
user_id
text
embedding
```

then the remapping script must preserve those features explicitly.

---

## 5. Semantic ID Generation Stage

### Input

The semantic ID generation script uses `MultiDomainItemDataset`.

For each domain config, it loads:

```text
embed_path   = shard_00000_fused.npy
id_map_path  = shard_00000_ids_incremental.npy
```

### Dataset Behaviour

`MultiDomainItemDataset` loads the embedding matrix and the ID map.

When `return_item_id=True`, it returns:

```text
(embed, domain_id, global_item_id)
```

If `id_map_path = shard_00000_ids_incremental.npy`, then:

```text
global_item_id = global_incremental_item_id
```

Therefore, semantic ID generation saves:

```text
semantic_ids_all.pt["item_id"] = global_incremental_item_id
```

### Output

The semantic ID script saves:

```text
semantic_ids_all.pt
semantic_ids_domain_<domain_id>.pt
```

Each file contains:

```python
{
    "domain_id": ...,
    "item_id": ...,
    "semantic_id": ...,
}
```

The meaning is:

```text
item_id:     global incremental item id
semantic_id: RQ-VAE code indices for that item
```

Example:

```text
semantic_ids_all.pt["item_id"][i] = 9455
semantic_ids_all.pt["semantic_id"][i] = [code_l0, code_l1, code_l2, ...]
```

---

## 6. Final Correct ID Flow

The full pipeline should be:

```text
raw item id
    ↓
shard_00000_ids.npy inverse lookup
    ↓
local_new_item_id / embedding row index
    ↓
shard_00000_ids_incremental.npy lookup
    ↓
global_incremental_item_id
    ↓
TFRecord sequence_data
```

For semantic ID generation:

```text
embedding row index
    ↓
shard_00000_ids_incremental.npy[row index]
    ↓
global_incremental_item_id
    ↓
semantic_ids_all.pt["item_id"]
```

Therefore, after remapping:

```text
TFRecord sequence_data
==
semantic_ids_all.pt["item_id"]
```

---

## 7. Key ID Spaces

| ID Space | Meaning | Scope | Example |
|---|---|---:|---|
| `raw_item_id` | Original item ID parsed from CSV / behaviour | Dataset original space | `613`, `671403` |
| `local_new_item_id` | Row index inside one domain embedding matrix | Single domain | `0`, `1`, `2` |
| `global_incremental_item_id` | Non-overlapping ID across domains | Multi-domain global | `0`, `6121`, `9455` |
| `semantic_id` | RQ-VAE code indices | Codebook token space | `[12, 37, 201]` |

---

## 8. Current Expected Alignment

After the corrected pipeline:

```text
shard_00000_fused.npy row index
    -> shard_00000_ids_incremental.npy[row index]
    -> semantic_ids_all.pt["item_id"]
```

and

```text
raw behaviour item
    -> local_new_item_id
    -> global_incremental_item_id
    -> remapped TFRecord sequence_data
```

So both sides meet at:

```text
global_incremental_item_id
```

---

## 9. Validation Checklist

### Check incremental ID length

```python
import numpy as np

old_ids = np.load("CLIP_emb/shard_00000_ids.npy")
inc_ids = np.load("CLIP_emb/shard_00000_ids_incremental.npy")
emb = np.load("CLIP_emb/shard_00000_fused.npy")

assert len(old_ids) == len(inc_ids) == emb.shape[0]
print(old_ids.shape, inc_ids[:5], inc_ids[-5:], emb.shape)
```

### Check semantic ID item ID space

```python
import torch

pt = torch.load("semantic_ids_all.pt", map_location="cpu")
print(pt["item_id"][:20])
print(pt["item_id"].min(), pt["item_id"].max())
```

Expected for Bili:

```text
min item id >= 9455
max item id <= 147487
```

### Check remapped TFRecord ID range

```python
import tfrecord

loader = tfrecord.tfrecord_loader(
    data_path="remapped_incremental_tfrecord/testing/partition_0.tfrecord.gz",
    index_path=None,
    description={"sequence_data": "int"},
    compression_type="gzip",
)

for record in loader:
    seq = record["sequence_data"]
    print(seq[:20])
    print(min(seq), max(seq))
    break
```

Expected for Bili-only TFRecord:

```text
min sequence id >= 9455
max sequence id <= 147487
```

---

## 10. Common Failure Modes

### Failure Mode 1: `.pt["item_id"]` is raw item ID

Cause:

```text
id_map_path = shard_00000_ids.npy
```

Result:

```text
semantic_ids_all.pt["item_id"] = raw_item_id
TFRecord sequence_data = local_new_item_id or global_incremental_item_id
```

This is not aligned.

Fix:

```text
id_map_path = shard_00000_ids_incremental.npy
```

---

### Failure Mode 2: remap script uses raw IDs as keys

Wrong:

```python
id_map = {
    old_id: new_id
    for old_id, new_id in zip(shard_00000_ids.npy, shard_00000_ids_incremental.npy)
}
```

This maps:

```text
raw_item_id -> global_incremental_item_id
```

Correct:

```python
id_map = {
    local_id: global_id
    for local_id, global_id in enumerate(shard_00000_ids_incremental.npy)
}
```

This maps:

```text
local_new_item_id -> global_incremental_item_id
```

---

### Failure Mode 3: using global incremental ID to index single-domain embedding matrix

Wrong:

```python
item_embeds[global_incremental_item_id]
```

Correct during single-domain TFRecord construction:

```python
item_embeds[local_new_item_id]
```

The global incremental ID should be used for final `sequence_data`, not for indexing a single-domain embedding matrix unless all domain embeddings have been concatenated into one global matrix.

---

## 11. One-Sentence Summary

The final correct design is: `shard_00000_ids.npy` maps local row index to raw item ID; the initial TFRecord maps raw item ID back to local row index; `shard_00000_ids_incremental.npy` maps local row index to global incremental item ID; both remapped TFRecords and semantic ID `.pt` files should finally use this same global incremental item ID space.
