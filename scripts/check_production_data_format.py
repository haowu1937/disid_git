import os
import glob
import json
from collections import Counter, defaultdict

# ========= 这里改成你的目录或通配符 =========
file_glob = "../data/docid_title_embedding/*.jsonl"
# 例如:
# file_glob = "/mnt/data/xxx/*.jsonl"

# 每个文件打印前几条样本
show_n = 3

# 每个文件额外扫描多少行做统计
scan_n = 2000
# =======================================


def short_text(x, n=120):
    s = repr(x)
    if len(s) > n:
        return s[:n] + "...'"
    return s


def find_id_key(d):
    cand = [
        "item_id", "itemid", "id", "docid", "doc_id",
        "itemId", "itemID"
    ]
    for k in cand:
        if k in d:
            return k
    return None


def find_title_key(d):
    cand = [
        "item_title", "title", "name", "text",
        "itemTitle"
    ]
    for k in cand:
        if k in d:
            return k
    return None


def find_emb_key(d):
    cand = [
        "embedding", "emb", "vector", "item_embedding",
        "title_embedding", "embed", "feature"
    ]
    for k in cand:
        if k in d:
            return k
    return None


def get_emb_info(x):
    if isinstance(x, list):
        return {
            "type": "list",
            "len": len(x),
            "first_5": x[:5],
            "elem_type": type(x[0]).__name__ if len(x) > 0 else None,
        }
    return {
        "type": type(x).__name__,
        "len": None,
        "first_5": None,
        "elem_type": None,
    }


def check_one_file(path):
    print("=" * 100)
    print(f"FILE: {path}")

    bad_json = 0
    empty_line = 0
    total_seen = 0

    key_set_counter = Counter()
    id_key_counter = Counter()
    title_key_counter = Counter()
    emb_key_counter = Counter()
    id_type_counter = Counter()
    title_type_counter = Counter()
    emb_type_counter = Counter()
    emb_len_counter = Counter()
    missing_id = 0
    missing_title = 0
    missing_emb = 0

    shown = 0

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if total_seen >= scan_n and shown >= show_n:
                break

            line = line.strip()
            if not line:
                empty_line += 1
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                bad_json += 1
                if bad_json <= 3:
                    print(f"[bad json] line {i}: {e}")
                    print(short_text(line, 200))
                continue

            total_seen += 1

            if not isinstance(obj, dict):
                key_set_counter[f"<top:{type(obj).__name__}>"] += 1
                if shown < show_n:
                    print(f"\n[sample {shown + 1}] line {i}")
                    print(f"top-level type: {type(obj).__name__}")
                    print(f"value: {short_text(obj, 200)}")
                    shown += 1
                continue

            keys = tuple(sorted(obj.keys()))
            key_set_counter[keys] += 1

            id_key = find_id_key(obj)
            title_key = find_title_key(obj)
            emb_key = find_emb_key(obj)

            id_key_counter[id_key] += 1
            title_key_counter[title_key] += 1
            emb_key_counter[emb_key] += 1

            if id_key is None:
                missing_id += 1
                id_val = None
            else:
                id_val = obj[id_key]
                id_type_counter[type(id_val).__name__] += 1

            if title_key is None:
                missing_title += 1
                title_val = None
            else:
                title_val = obj[title_key]
                title_type_counter[type(title_val).__name__] += 1

            if emb_key is None:
                missing_emb += 1
                emb_val = None
            else:
                emb_val = obj[emb_key]
                emb_info = get_emb_info(emb_val)
                emb_type_counter[emb_info["type"]] += 1
                if emb_info["len"] is not None:
                    emb_len_counter[emb_info["len"]] += 1

            if shown < show_n:
                print(f"\n[sample {shown + 1}] line {i}")
                print("keys:", list(obj.keys()))

                print("id key:", id_key)
                print("id type:", type(id_val).__name__ if id_val is not None else None)
                print("id value:", short_text(id_val, 120))

                print("title key:", title_key)
                print("title type:", type(title_val).__name__ if title_val is not None else None)
                print("title value:", short_text(title_val, 120))

                print("embedding key:", emb_key)
                if emb_key is not None:
                    emb_info = get_emb_info(obj[emb_key])
                    print("embedding type:", emb_info["type"])
                    print("embedding len:", emb_info["len"])
                    print("embedding elem type:", emb_info["elem_type"])
                    print("embedding first 5:", emb_info["first_5"])
                else:
                    print("embedding: None")

                shown += 1

    print("\n--- summary ---")
    print("scanned valid json lines:", total_seen)
    print("empty lines:", empty_line)
    print("bad json lines:", bad_json)

    print("\nkey sets (top 5):")
    for k, v in key_set_counter.most_common(5):
        print(v, k)

    print("\nid key counts:")
    print(id_key_counter)

    print("\ntitle key counts:")
    print(title_key_counter)

    print("\nembedding key counts:")
    print(emb_key_counter)

    print("\nid value types:")
    print(id_type_counter)

    print("\ntitle value types:")
    print(title_type_counter)

    print("\nembedding value types:")
    print(emb_type_counter)

    print("\nembedding lengths:")
    print(emb_len_counter)

    print("\nmissing counts:")
    print({
        "missing_id": missing_id,
        "missing_title": missing_title,
        "missing_emb": missing_emb,
    })


def main():
    files = sorted(glob.glob(file_glob))
    if not files:
        print(f"No files matched: {file_glob}")
        return

    print(f"Found {len(files)} files")
    for p in files:
        size_gb = os.path.getsize(p) / 1024 / 1024 / 1024
        print(f"- {p} ({size_gb:.3f} GB)")

    for p in files:
        check_one_file(p)


if __name__ == "__main__":
    main()