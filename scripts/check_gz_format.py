import os
import glob
from collections import Counter

import tfrecord


def inspect_training_tfrecord(
    tfrecord_dir: str,
    tfrecord_pattern: str = "partition_*.tfrecord.gz",
    max_records_per_file: int = 5,
):
    pattern = os.path.join(tfrecord_dir, tfrecord_pattern)
    tfrecord_files = sorted(glob.glob(pattern))

    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files found with pattern: {pattern}")

    print(f"Found {len(tfrecord_files)} files")
    print()

    description = {
        "sequence_data": "int",
    }

    total_records = 0
    total_valid_records = 0
    total_printed_records = 0
    seq_len_counter = Counter()

    # Global id range
    global_min_id = None
    global_max_id = None

    for path in tfrecord_files:
        print(f"===== FILE: {path} =====")

        loader = tfrecord.tfrecord_loader(
            data_path=path,
            index_path=None,
            description=description,
            compression_type="gzip",
        )

        file_records = 0
        file_valid = 0
        file_printed = 0

        # Per-file id range
        file_min_id = None
        file_max_id = None

        try:
            for i, record in enumerate(loader):
                file_records += 1
                total_records += 1

                should_print = file_printed < max_records_per_file

                if should_print:
                    print(f"\n--- RECORD {i} ---")
                    print(f"keys = {list(record.keys())}")
                    file_printed += 1
                    total_printed_records += 1

                raw_seq = record.get("sequence_data", None)

                if raw_seq is None:
                    if should_print:
                        print("sequence_data: MISSING")
                    continue

                raw_seq = [int(x) for x in raw_seq]

                if should_print:
                    print(f"sequence_data type   = {type(raw_seq).__name__}")
                    print(f"sequence_data length = {len(raw_seq)}")
                    print(f"sequence_data preview= {raw_seq[:20]}")

                if len(raw_seq) == 0:
                    if should_print:
                        print("status = EMPTY")
                    continue

                file_valid += 1
                total_valid_records += 1
                seq_len_counter[len(raw_seq)] += 1

                # Update per-record id range
                record_min_id = min(raw_seq)
                record_max_id = max(raw_seq)

                # Update per-file id range
                if file_min_id is None or record_min_id < file_min_id:
                    file_min_id = record_min_id
                if file_max_id is None or record_max_id > file_max_id:
                    file_max_id = record_max_id

                # Update global id range
                if global_min_id is None or record_min_id < global_min_id:
                    global_min_id = record_min_id
                if global_max_id is None or record_max_id > global_max_id:
                    global_max_id = record_max_id

                if should_print:
                    print(f"record_min_id = {record_min_id}")
                    print(f"record_max_id = {record_max_id}")
                    print("status = OK")

        except Exception as e:
            print(f"Failed to read file: {e}")

        print()
        print(f"file_records = {file_records}")
        print(f"file_valid   = {file_valid}")
        print(f"file_printed = {file_printed}")

        if file_min_id is not None:
            print(f"file_min_id  = {file_min_id}")
            print(f"file_max_id  = {file_max_id}")
        else:
            print("file_min_id  = N/A")
            print("file_max_id  = N/A")

        print()

    print("===== SUMMARY =====")
    print(f"total_files           = {len(tfrecord_files)}")
    print(f"total_records         = {total_records}")
    print(f"total_valid_records   = {total_valid_records}")
    print(f"total_printed_records = {total_printed_records}")

    if total_records > 0:
        valid_rate = total_valid_records / total_records
        print(f"valid_record_rate     = {valid_rate:.4f}")

    if global_min_id is not None:
        print()
        print(f"global_min_id = {global_min_id}")
        print(f"global_max_id = {global_max_id}")
    else:
        print()
        print("global_min_id = N/A")
        print("global_max_id = N/A")

    if seq_len_counter:
        print()
        print("sequence length distribution (top 20 by length):")
        for seq_len, cnt in sorted(seq_len_counter.items())[:20]:
            print(f"  len={seq_len}: {cnt}")

        min_len = min(seq_len_counter.keys())
        max_len = max(seq_len_counter.keys())

        print()
        print(f"min_seq_len = {min_len}")
        print(f"max_seq_len = {max_len}")


if __name__ == "__main__":
    inspect_training_tfrecord(
        tfrecord_dir="/home/shared/ninerec/qb/training/",
        tfrecord_pattern="partition_*.tfrecord.gz",
        max_records_per_file=2,
    )