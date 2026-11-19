# Codebook Training Input Format

## `embed_path`
- File type: NumPy `.npy`
- Content: item embedding matrix with shape `(num_items, embed_dim)`
- Storage rule: one row per item, row order defines the item order for this domain
- Expected usage: all domains should use the same embedding dimension when merged
- Read process: `embeds = np.load(embed_path)`

## `id_map_path`
- File type: NumPy `.npy`
- Content: 1D item ID array with length `num_items`
- Allowed values: integer IDs, string IDs, or byte-string IDs
- Storage rule: the `i`-th ID must correspond to the `i`-th embedding row
- Read process: `raw_ids = np.load(id_map_path, allow_pickle=True)`
- Post-processing:
  - decode bytes to UTF-8 strings if needed
  - cast string arrays to `str`
  - flatten to 1D if stored with extra dimensions

## Validation and Merge Logic
- Check `len(raw_ids) == embeds.shape[0]`
- Optionally check `embeds.shape[1] == expected_embed_dim`
- Assign one integer domain label per domain: `0, 1, 2, ...`
- Merge all domains by `np.vstack(...)` for embeddings and `np.concatenate(...)` for domain labels

## Preprocessing Guidance for New Datasets
1. Save each domain's item embeddings as one 2D `.npy` file.
2. Save aligned item IDs as one 1D `.npy` file in the same row order.
3. Keep embedding dimension consistent across domains.
4. Ensure no row misalignment between embeddings and IDs.
5. If no ID file is available, the loader will fall back to within-domain indices `0..num_items-1`.