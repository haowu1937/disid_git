import os
import torch

INPUT_DIR = "/home/hao/FRec/DisenID/codebook/ninerec_256_incremental_id"
N = 10


def load_first_n_item_ids(pt_path, n):
    data = torch.load(pt_path, map_location="cpu")

    if not isinstance(data, dict):
        raise ValueError(f"{pt_path} is not a dict, got {type(data)}")

    if "item_id" not in data:
        raise KeyError(f"{pt_path} does not contain key 'item_id'")

    item_ids = data["item_id"]

    if not torch.is_tensor(item_ids):
        item_ids = torch.tensor(item_ids)

    first_n = item_ids[:n].tolist()
    total_num = item_ids.shape[0]
    return first_n, total_num


def main():
    pt_files = sorted(
        [f for f in os.listdir(INPUT_DIR) if f.endswith(".pt")]
    )

    if not pt_files:
        print(f"No .pt files found in {INPUT_DIR}")
        return

    for file_name in pt_files:
        pt_path = os.path.join(INPUT_DIR, file_name)
        try:
            first_n_item_ids, total_num = load_first_n_item_ids(pt_path, N)
            print(f"\n===== {file_name} =====")
            print(f"total item_ids: {total_num}")
            print(f"first {min(N, total_num)} item_ids: {first_n_item_ids}")
        except Exception as e:
            print(f"\n===== {file_name} =====")
            print(f"failed to load: {e}")


if __name__ == "__main__":
    main()