import os
import numpy as np

domains = [
    {
        "name": "QB",
        "expected_num_items": 6121,
        "embed_path": "/mnt/data_2/data_oleg/recsys/ninerec/Downstream_datasets/QB/CLIP_emb/",
    },
    {
        "name": "TN",
        "expected_num_items": 3334,
        "embed_path": "/mnt/data_2/data_oleg/recsys/ninerec/Downstream_datasets/TN/CLIP_emb/",
    },
    {
        "name": "Bili",
        "expected_num_items": 60173,
        "embed_path": "/home/hao/FRec/DisenID/data/Bili_10k/CLIP_emb/",
    },
]


def produce_incremental_id():
    start_id = 0

    for domain in domains:
        name = domain["name"]
        expected_num_items = domain["expected_num_items"]
        embed_path = domain["embed_path"]

        old_ids_path = os.path.join(embed_path, "shard_00000_ids.npy")
        new_ids_path = os.path.join(embed_path, "shard_00000_ids_incremental.npy")

        if not os.path.exists(old_ids_path):
            raise FileNotFoundError(f"[{name}] file not found: {old_ids_path}")

        old_ids = np.load(old_ids_path, allow_pickle=False)

        if old_ids.ndim != 1:
            raise ValueError(
                f"[{name}] expected 1D array, but got shape {old_ids.shape}"
            )

        actual_num_items = len(old_ids)

        if actual_num_items != expected_num_items:
            raise ValueError(
                f"[{name}] item count mismatch: "
                f"expected {expected_num_items}, got {actual_num_items}"
            )

        end_id = start_id + actual_num_items
        incremental_ids = np.arange(start_id, end_id, dtype=np.int64)

        np.save(new_ids_path, incremental_ids)

        print(f"===== {name} =====")
        print(f"old ids path   : {old_ids_path}")
        print(f"new ids path   : {new_ids_path}")
        print(f"num items      : {actual_num_items}")
        print(f"id range       : [{start_id}, {end_id - 1}]")
        print(f"array dtype    : {incremental_ids.dtype}")
        print(f"array shape    : {incremental_ids.shape}")
        print(f"first 10 ids   : {incremental_ids[:10]}")
        print(f"last 10 ids    : {incremental_ids[-10:]}")
        print()

        start_id = end_id

    print(f"Total number of items: {start_id}")




if __name__ == "__main__":
    produce_incremental_id()