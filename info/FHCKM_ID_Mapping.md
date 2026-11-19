# FHCKM ID Mapping

Items from all domains are remapped into one unified global ID space.

## Original ID format

Original item IDs follow:

`domain_id-item_id`

- `domain_id`: domain index in the fixed domain list
- `item_id`: local item index within that domain, starting from 0

Example:

- `0-0` = the first item in Food
- `1-0` = the first item in Home

## Fixed domain order

| domain_id | domain | num_items |
|---|---|---:|
| 0 | Food | 39670 |
| 1 | Home | 185552 |
| 2 | CDs | 64439 |
| 3 | Kindle | 98111 |
| 4 | Movies | 59203 |



## Mapping rule

Domains are processed in the fixed order above.

Each domain receives a continuous global ID range:

`global_id = global_offset + item_id`

where `global_offset` is the total number of items in all previous domains.

## Global ID ranges

| domain | original ID range | global ID range |
|---|---|---|
| Food | `0-0` ~ `0-39669` | `0` ~ `39669` |
| Home | `1-0` ~ `1-185551` | `39670` ~ `225221` |
| CDs | `2-0` ~ `2-64438` | `225222` ~ `289660` |
| Kindle | `3-0` ~ `3-98110` | `289661` ~ `387771` |
| Movies | `4-0` ~ `4-59202` | `387772` ~ `446974` |

## Output file

For each domain, the script saves:

`{ROOT}/{domain}/shard_00000_ids_incremental.npy`

This file is a 1D `np.int64` array containing the remapped global IDs for that domain.

## Properties

- Global IDs are continuous
- Global IDs do not overlap across domains
- The mapping is fully determined by domain order and item counts

---

# Downstream ID Mapping

Downstream datasets are **not** merged into one shared downstream global space.

Instead, each downstream dataset is processed **independently**, and its incremental IDs always start from the first unused FHCKM global ID:

`446975`

This is because the FHCKM pre-trained domains occupy the full range:

`0 ~ 446974`

So for every downstream dataset:

`downstream_global_id = 446975 + item_id`

where `item_id` is the local item index inside that downstream dataset, starting from 0.

## Important difference from FHCKM mapping

FHCKM mapping uses **cumulative offsets across domains**, so different FHCKM domains occupy different non-overlapping global ranges.

Downstream mapping does **not** accumulate across downstream datasets.

That means:

- Scientific starts from `446975`
- Pantry also starts from `446975`
- Instruments also starts from `446975`
- Arts also starts from `446975`
- Office also starts from `446975`

These downstream datasets are treated as separate evaluation datasets, so their ID spaces do not need to be unique across each other.

## Downstream datasets

The following downstream datasets use this rule:

| domain | num_items | local item_id range | incremental ID range |
|---|---:|---|---|
| Scientific | 4385 | `0` ~ `4384` | `446975` ~ `451359` |
| Pantry | 4898 | `0` ~ `4897` | `446975` ~ `451872` |
| Instruments | 9964 | `0` ~ `9963` | `446975` ~ `456938` |
| Arts | 21019 | `0` ~ `21018` | `446975` ~ `467993` |
| Office | 25986 | `0` ~ `25985` | `446975` ~ `472960` |

## Excluded dataset

`Online Retail` is not included in this downstream incremental ID generation.

## Downstream output file

For each downstream dataset, the script saves:

`/mnt/data_2/data_hao/DisenID/data/Amazon/downstream/{domain}/shard_00000_ids_incremental.npy`

This file is a 1D `np.int64` array of length `num_items`, with values:

`[446975, 446976, ..., 446975 + num_items - 1]`

## Downstream properties

- Every downstream dataset starts from the same base ID: `446975`
- IDs are continuous **within** each downstream dataset
- IDs are **not required to be unique across downstream datasets**
- The mapping is fully determined by the downstream dataset's local item count
- Existing `.npy` files can be skipped if already generated correctly