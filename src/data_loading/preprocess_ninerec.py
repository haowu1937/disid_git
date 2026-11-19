# preprocess_ninerec.py

import gzip
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import tfrecord


# ============================================================
# Hardcoded dataset paths
# ============================================================
DATASETS = {
    "QB": {
        "input_path": "/mnt/data_2/data_oleg/recsys/ninerec/Downstream_datasets/QB/QB_behaviour.tsv",
        "output_root_dir": "/mnt/data_2/data_oleg/recsys/ninerec/Downstream_datasets/QB",
    },
    "TN": {
        "input_path": "/mnt/data_2/data_oleg/recsys/ninerec/Downstream_datasets/TN/TN_behaviour.tsv",
        "output_root_dir": "/mnt/data_2/data_oleg/recsys/ninerec/Downstream_datasets/TN",
    },
    "Bili10Kv3": {
        "input_path": "/mnt/data_2/data_hao/DisenID/data/Bili_10k/Bili10Kv3_behaviour.tsv",
        "output_root_dir": "/mnt/data_2/data_hao/DisenID/data/Bili_10k",
    },
}

# Each split will be written to:
#   {output_root_dir}/train/partition_00000.tfrecord.gz
#   {output_root_dir}/valid/partition_00000.tfrecord.gz
#   {output_root_dir}/test/partition_00000.tfrecord.gz

RECORDS_PER_PARTITION = 100000
MIN_TOTAL_SEQ_LEN = 3
CLEAN_OLD_PARTITIONS = True

# Logical description you want
DESCRIPTION = {
    "sequence_data": "int",   # item id sequence, e.g. [48487,20811,42931,105021,58280,87702]
    "user_id": "str",         # keep original string, no preprocessing
}

# Actual description if using `tfrecord` package
# Note: this package stores strings as bytes, so user_id is written as UTF-8 bytes.
TFRECORD_PACKAGE_DESCRIPTION = {
    "sequence_data": "int",
    "user_id": "byte",
}


def parse_item_token(token: str) -> Optional[int]:
    """
    Convert raw item token to integer item id.
    - Remove leading 'v' / 'V' if present
    - Keep plain integer tokens unchanged
    """
    token = token.strip()
    if not token:
        return None

    if token.startswith(("v", "V")):
        token = token[1:]

    if not token:
        return None

    if not token.isdigit():
        raise ValueError(f"Invalid item token: {token}")

    return int(token)


def parse_behavior_line(line: str) -> Optional[Dict[str, object]]:
    """
    Parse one behavior line.

    Supported examples:
    QB/TN:
        u100    v521057 v120764 v398969 v36262 ...
    Bili10Kv3:
        (1000027,)      48487 20811 42931 105021 ...

    Returns:
        {
            "user_id": original user id string,
            "sequence_data": [int, int, ...],
        }
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) < 2:
        return None

    user_id = parts[0]  # keep original string exactly as is
    item_tokens = parts[1:]

    seq: List[int] = []
    for x in item_tokens:
        item_id = parse_item_token(x)
        if item_id is not None:
            seq.append(item_id)

    if len(seq) == 0:
        return None

    return {
        "user_id": user_id,
        "sequence_data": seq,
    }


def split_leave_one_out(item_seq: List[int]) -> Optional[Dict[str, List[int]]]:
    """
    Leave-one-out split for sequential recommendation.

    Given original sequence:
        [i1, i2, ..., i_{n-2}, i_{n-1}, i_n]

    We write:
        train = [i1, ..., i_{n-2}]
        valid = [i1, ..., i_{n-1}]
        test  = [i1, ..., i_n]

    Why this format?
    Because downstream NextItemSemanticDataset / SlidingWindowNextItemSemanticDataset
    use the LAST item in a sequence as the prediction target.

    So:
        - valid last item = original second last interaction
        - test  last item = original last interaction
    """
    if len(item_seq) < MIN_TOTAL_SEQ_LEN:
        return None

    return {
        "train": item_seq[:-2],
        "valid": item_seq[:-1],
        "test": item_seq[:],
    }


def cleanup_old_partitions(output_dir: Path) -> None:
    if not output_dir.exists():
        return

    for p in output_dir.glob("partition_*.tfrecord"):
        p.unlink()
    for p in output_dir.glob("partition_*.tfrecord.gz"):
        p.unlink()


def write_partition(records: List[Dict[str, object]], output_dir: Path, part_idx: int) -> Path:
    """
    Write one partition:
        partition_00000.tfrecord.gz

    We first write an uncompressed .tfrecord with tfrecord package,
    then gzip the whole file into .tfrecord.gz.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_tfrecord = output_dir / f"partition_{part_idx:05d}.tfrecord"
    final_tfrecord_gz = output_dir / f"partition_{part_idx:05d}.tfrecord.gz"

    writer = tfrecord.TFRecordWriter(str(tmp_tfrecord))
    try:
        for record in records:
            writer.write(
                {
                    "sequence_data": (record["sequence_data"], "int"),
                    "user_id": (str(record["user_id"]).encode("utf-8"), "byte"),
                }
            )
    finally:
        writer.close()

    with open(tmp_tfrecord, "rb") as f_in, gzip.open(final_tfrecord_gz, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    tmp_tfrecord.unlink()
    return final_tfrecord_gz


def flush_buffer_if_needed(
    split_name: str,
    buffer_map: Dict[str, List[Dict[str, object]]],
    part_idx_map: Dict[str, int],
    split_dir_map: Dict[str, Path],
) -> None:
    if len(buffer_map[split_name]) >= RECORDS_PER_PARTITION:
        out_path = write_partition(
            records=buffer_map[split_name],
            output_dir=split_dir_map[split_name],
            part_idx=part_idx_map[split_name],
        )
        print(f"[{split_name}] Wrote {out_path.name} with {len(buffer_map[split_name])} records")
        buffer_map[split_name] = []
        part_idx_map[split_name] += 1


def flush_remaining_buffers(
    buffer_map: Dict[str, List[Dict[str, object]]],
    part_idx_map: Dict[str, int],
    split_dir_map: Dict[str, Path],
) -> None:
    for split_name in ["train", "valid", "test"]:
        if buffer_map[split_name]:
            out_path = write_partition(
                records=buffer_map[split_name],
                output_dir=split_dir_map[split_name],
                part_idx=part_idx_map[split_name],
            )
            print(f"[{split_name}] Wrote {out_path.name} with {len(buffer_map[split_name])} records")
            buffer_map[split_name] = []
            part_idx_map[split_name] += 1


def process_dataset(
    name: str,
    input_path: str,
    output_root_dir: str,
) -> None:
    input_path = Path(input_path)
    output_root_dir = Path(output_root_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"[{name}] Input file not found: {input_path}")

    split_dir_map = {
        "train": output_root_dir / "train",
        "valid": output_root_dir / "valid",
        "test": output_root_dir / "test",
    }

    for split_dir in split_dir_map.values():
        split_dir.mkdir(parents=True, exist_ok=True)
        if CLEAN_OLD_PARTITIONS:
            cleanup_old_partitions(split_dir)

    buffer_map: Dict[str, List[Dict[str, object]]] = {
        "train": [],
        "valid": [],
        "test": [],
    }

    part_idx_map: Dict[str, int] = {
        "train": 0,
        "valid": 0,
        "test": 0,
    }

    total_lines = 0
    total_valid_users = 0
    total_skipped_parse = 0
    total_skipped_short = 0

    train_total_len = 0
    valid_total_len = 0
    test_total_len = 0

    print(f"\n===== Processing {name} =====")
    print(f"Input      : {input_path}")
    print(f"Output root: {output_root_dir}")

    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            total_lines += 1

            try:
                parsed = parse_behavior_line(line)
            except Exception as e:
                total_skipped_parse += 1
                print(f"[{name}] Skip line {line_no}: {e}")
                continue

            if parsed is None:
                total_skipped_parse += 1
                continue

            user_id = parsed["user_id"]
            full_seq = parsed["sequence_data"]

            split_seq_map = split_leave_one_out(full_seq)
            if split_seq_map is None:
                total_skipped_short += 1
                continue

            total_valid_users += 1
            train_total_len += len(split_seq_map["train"])
            valid_total_len += len(split_seq_map["valid"])
            test_total_len += len(split_seq_map["test"])

            for split_name in ["train", "valid", "test"]:
                record = {
                    "user_id": user_id,
                    "sequence_data": split_seq_map[split_name],
                }
                buffer_map[split_name].append(record)
                flush_buffer_if_needed(
                    split_name=split_name,
                    buffer_map=buffer_map,
                    part_idx_map=part_idx_map,
                    split_dir_map=split_dir_map,
                )

    flush_remaining_buffers(
        buffer_map=buffer_map,
        part_idx_map=part_idx_map,
        split_dir_map=split_dir_map,
    )

    print(f"[{name}] Done")
    print(f"[{name}] Total lines              : {total_lines}")
    print(f"[{name}] Valid users             : {total_valid_users}")
    print(f"[{name}] Skipped parse/empty      : {total_skipped_parse}")
    print(f"[{name}] Skipped short (<3 items) : {total_skipped_short}")
    print(f"[{name}] Train partitions        : {part_idx_map['train']}")
    print(f"[{name}] Valid partitions        : {part_idx_map['valid']}")
    print(f"[{name}] Test partitions         : {part_idx_map['test']}")

    if total_valid_users > 0:
        print(f"[{name}] Avg train seq len       : {train_total_len / total_valid_users:.2f}")
        print(f"[{name}] Avg valid seq len       : {valid_total_len / total_valid_users:.2f}")
        print(f"[{name}] Avg test seq len        : {test_total_len / total_valid_users:.2f}")


def main() -> None:
    print("Target logical description:")
    print(DESCRIPTION)
    print("Actual tfrecord package description:")
    print(TFRECORD_PACKAGE_DESCRIPTION)

    for name, cfg in DATASETS.items():
        process_dataset(
            name=name,
            input_path=cfg["input_path"],
            output_root_dir=cfg["output_root_dir"],
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()