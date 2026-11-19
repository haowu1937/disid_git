import pandas as pd


PATH = "/mnt/data_2/data_hao/DisenID/data/Amazon/FHCKM/FHCKM.train.inter"
NUM_USERS = 3


def main():
    df = pd.read_csv(PATH, sep="\t")

    user_col = "user_id:token"
    hist_col = "item_id_list:token_seq"
    target_col = "item_id:token"

    first_users = df[user_col].drop_duplicates().head(NUM_USERS).tolist()

    for user in first_users:
        user_df = df[df[user_col] == user]

        print("\n" + "=" * 120)
        print(f"USER {user} | rows={len(user_df)}")
        print("=" * 120)

        for i, (_, row) in enumerate(user_df.iterrows(), 1):
            hist_list = str(row[hist_col]).split() if pd.notna(row[hist_col]) else []
            full_seq = hist_list + [str(row[target_col])]
            print(f"[{i}] {' '.join(full_seq)}")


if __name__ == "__main__":
    main()