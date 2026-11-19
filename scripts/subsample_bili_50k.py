import argparse
import random
from pathlib import Path

import tensorflow as tf


def count_tfrecord(path: Path) -> int:
    dataset = tf.data.TFRecordDataset(
        str(path),
        compression_type="GZIP",
    )
    return sum(1 for _ in dataset)


def write_selected_records(src_path: Path, dst_path: Path, selected_indices: set):
    dataset = tf.data.TFRecordDataset(
        str(src_path),
        compression_type="GZIP",
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    options = tf.io.TFRecordOptions(compression_type="GZIP")

    written = 0
    with tf.io.TFRecordWriter(str(dst_path), options=options) as writer:
        for idx, record in enumerate(dataset):
            if idx in selected_indices:
                writer.write(record.numpy())
                written += 1

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="/home/shared/ninerec/Bili_500K")
    parser.add_argument("--dst", type=str, default="/home/shared/ninerec/Bili_50K")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    # evaluation 在这里等价于 validation
    splits = ["training", "testing", "evaluation"]

    src_files = {
        split: src_root / split / "partition_0.tfrecord.gz"
        for split in splits
    }

    for split, path in src_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    print("Counting records...")
    counts = {}
    for split, path in src_files.items():
        counts[split] = count_tfrecord(path)
        print(f"{split}: {counts[split]}")

    # leave-one-out 数据通常要求三个 split 一一对应
    if len(set(counts.values())) != 1:
        raise ValueError(
            "Record counts are not equal across splits. "
            "This may break leave-one-out alignment:\n"
            + "\n".join([f"{k}: {v}" for k, v in counts.items()])
        )

    total_records = counts["training"]

    if args.sample_size > total_records:
        raise ValueError(
            f"sample-size={args.sample_size} is larger than total records={total_records}"
        )

    random.seed(args.seed)

    # 关键点：
    # 只在 index 层面抽一次样本，然后三个 split 使用完全相同的 index。
    # 这样 training/testing/evaluation 中同一个用户的 leave-one-out 关系不会被打乱。
    selected_indices = set(random.sample(range(total_records), args.sample_size))

    print(f"Selected {args.sample_size} aligned indices from {total_records} records.")

    for split in splits:
        src_path = src_files[split]
        dst_path = dst_root / split / "partition_0.tfrecord.gz"

        print(f"Writing {split}...")
        written = write_selected_records(src_path, dst_path, selected_indices)

        if written != args.sample_size:
            raise RuntimeError(
                f"{split}: expected {args.sample_size}, but wrote {written}"
            )

        print(f"Saved {written} records to {dst_path}")

    print("Done.")
    print(f"Output saved to: {dst_root}")


if __name__ == "__main__":
    main()