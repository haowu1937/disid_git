# Shared Dataset Structure

All processed data has been moved under:

`AI*0:/home/shared/${dataset_name}`

where `${dataset_name}` is either:

- `amazon`
- `ninerec`

## Folder Structure

Each dataset folder follows the same high-level structure:

```text
/home/shared/${dataset_name}
├── sids/
├── ${source_domain}/
│   ├── training/
│   ├── testing/
│   └── evaluation/
└── ${target_domain}/
    ├── training/
    ├── testing/
    └── evaluation/
```


## 1.Dataset Info



For the **Amazon** dataset:

- Source domain: `FHCKM`
- Target domain: `scientific`


For the **NineRec** dataset:

- Source domain: `bili`
- Target domain: `tn`


### Behaviour Sequence Data Format

Each domain folder contains `training`, `testing`, and `evaluation` subfolders.

These folders store behaviour sequence data in GRID-format TFRecords:

```text
partition_*.tfrecord.gz
```


Each TFRecord example contains a single feature:

- Key: `sequence_data`
- Type after loading: Python `list`
- Content: one full behaviour sequence from one user
- Example value:

```python
[9371, 6128, 6193]
```

**Reading Process**

The TFRecord files are read with `tfrecord.tfrecord_loader(...)` using:

```python
tfrecord.tfrecord_loader(
    data_path=<partition_file>,
    index_path=None,
    description={"sequence_data": "int"},
    compression_type="gzip",
)
```

## 2.SID Files

The `sids/` folder stores SID mapping files also in GRID format and they are already duduplicated with one additional layer of codebook, for example:

```text
sid_rqvae.pt
sid_dtrqvae.pt
```

These files provide the item ID to SID mapping used together with the behaviour sequence data.

#### SID file naming conventions:

- `rqvae`: standard RQ-VAE
- `dtrqvae`: Disentangled RQ-VAE
- `ds`: trained with domain-balanced batch sampler


## 3. Experiment Setting

We need results from the following two types of experiments:

- No PT Training

Train directly on the target domain.

- With PT Training

Pre-train on the source domain, then fine-tune on the target domain.