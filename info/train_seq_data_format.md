# Training Data Format

## Overview
Training data is stored as multiple compressed TFRecord shards under `train/`, named like `partition_*.tfrecord.gz`.

## Record Schema
Each TFRecord example contains a single feature:
- `sequence_data`: list of integer IDs

From the inspected samples:
- keys: `['sequence_data']`
- type: Python `list` after loading
- observed length: `3`
- example values: `[9371, 6128, 6193]`

## Storage Format
- File format: TFRecord
- Compression: `gzip`
- Sharding: multiple partition files
- One training instance per record

## Reading Process
Data is read with `tfrecord.tfrecord_loader(...)` using:
- `data_path=<partition file>`
- `index_path=None`
- `description={"sequence_data": "int"}`
- `compression_type="gzip"`

The loader decodes each record into a dictionary, then `sequence_data` is cast to `int` and validated as a non-empty sequence.

## Preprocessing Guidance for Other Datasets
To prepare another dataset in the same format:
1. Convert each training sample into one integer ID sequence.
2. Store that sequence in the feature `sequence_data`.
3. Write records into sharded `.tfrecord.gz` files.
4. Keep the schema consistent across all shards.
5. Ensure IDs are already remapped to the final training vocabulary before serialization.