# FHCKM Dataset Format Summary

## Dataset Scope
- Root path: `/mnt/data_2/data_hao/DisenID/data/Amazon/FHCKM`
- Domain order from `FHCKM.pt_datasets`:
  - `Food`
  - `Home`
  - `CDs`
  - `Kindle`
  - `Movies`

## Interaction File Format
- Example file: `FHCKM.test.inter`
- Shape: `1,361,408 x 3`
- Columns:
  - `user_id:token`
  - `item_id_list:token_seq`
  - `item_id:token`

## Token Format
- `user_id:token` uses the format `<domain_id>-<user_index>`
- `item_id:token` uses the format `<domain_id>-<item_index>`
- Example:
  - `user_id = 0-0`
  - `item_id = 0-174`

## Item ID Mapping
- Item IDs are **not globally unique integers**.
- Each item is identified by a pair:
  - `domain_id`
  - `item_index` within that domain
- Therefore, the effective item key is:
  - `<domain_id>-<item_index>`
- This avoids collisions across domains even if two domains both contain item index `123`.

## Domain ID Mapping
- `0 -> Food`
- `1 -> Home`
- `2 -> CDs`
- `3 -> Kindle`
- `4 -> Movies`

## Embedding File Format
- Each domain has two embedding files:
  - `<Domain>.feat1CLS`
  - `<Domain>.feat2CLS`
- Both are binary `float32` files loaded with `np.fromfile(...)`
- Embedding dimension: `768`
- After loading, each file is reshaped to `[num_items, 768]`

## Embedding Shapes
- `Food`: `39670 x 768`
- `Home`: `185552 x 768`
- `CDs`: `64439 x 768`
- `Kindle`: `98111 x 768`
- `Movies`: `59203 x 768`

## Alignment Rule
- For a token `<did>-<iid>`, load the embedding from the corresponding domain file:
  - first select domain by `did`
  - then select row `iid` from that domain's embedding matrix
- `feat1CLS` and `feat2CLS` have the same number of items within each domain, so they are aligned row by row.