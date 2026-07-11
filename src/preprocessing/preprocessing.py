import os
import math
import pandas as pd

# Global config
MAX_SEQ_LEN = 50
K_CORE = 5
MIN_RATING = 4          # Amazon and ML-1M treat ratings >= 4 as positive
OUTPUT_DIR = "src/datasets/preprocessed"


# Utilities
def k_core_filtering(df: pd.DataFrame, user_col: str, item_col: str, k: int = 5) -> pd.DataFrame:
    """Iterative k-core filtering until the graph is stable."""
    while True:
        before = len(df)

        user_counts = df[user_col].value_counts()
        df = df[df[user_col].isin(user_counts[user_counts >= k].index)]

        item_counts = df[item_col].value_counts()
        df = df[df[item_col].isin(item_counts[item_counts >= k].index)]

        if len(df) == before:
            break

    return df.reset_index(drop=True)


def truncate_sequences(df: pd.DataFrame, user_col: str, timestamp_col: str,
                        max_len: int = MAX_SEQ_LEN) -> pd.DataFrame:
    """Keep only the most recent `max_len` interactions per user."""
    df = df.sort_values([user_col, timestamp_col])
    df = df.groupby(user_col).tail(max_len)
    return df.reset_index(drop=True)


def print_stats(df: pd.DataFrame, user_col: str, item_col: str, name: str) -> None:
    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    n_inter = len(df)
    avg_len = df.groupby(user_col).size().mean()
    sparsity = 1 - n_inter / (n_users * n_items)
    print(f"Dataset : {name}")
    print(f"  Interactions : {n_inter:,}")
    print(f"  Users        : {n_users:,}")
    print(f"  Items        : {n_items:,}")
    print(f"  Avg seq len  : {avg_len:.1f}")
    print(f"  Sparsity     : {sparsity:.4%}")


def save_recbole_format(df: pd.DataFrame, dataset_name: str,
                         user_col: str, item_col: str,
                         timestamp_col: str) -> None:
    """
    Save dataset in RecBole atomic .inter format:
        user_id:token  item_id:token  timestamp:float
    """
    os.makedirs(os.path.join(OUTPUT_DIR, dataset_name), exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, dataset_name, f"{dataset_name}.inter")

    out_df = df[[user_col, item_col, timestamp_col]].copy()
    out_df.columns = ["user_id:token", "item_id:token", "timestamp:float"]

    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"  Saved → {out_path}")


# Amazon (Beauty & Video Games)
def preprocess_amazon(input_path: str, dataset_name: str) -> None:
    """
    Shared preprocessing for Amazon ratings-only CSV files.
    Expected columns : item_id, user_id, rating, timestamp
    """
    print(f"\n[Amazon – {dataset_name}] Loading...")
    df = pd.read_csv(
        input_path,
        header=None,
        names=["item_id", "user_id", "rating", "timestamp"],
    )

    print(f"  Raw interactions : {len(df):,}")

    # 1. Keep positive feedback only
    df = df[df["rating"] >= MIN_RATING].copy()
    print(f"  After rating >= {MIN_RATING} filter : {len(df):,}")

    # 2. Drop duplicates (same user–item pair, keep earliest)
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["user_id", "item_id"], keep="first"
    )

    # 3. K-core filtering
    df = k_core_filtering(df, user_col="user_id", item_col="item_id", k=K_CORE)
    print(f"  After {K_CORE}-core filtering : {len(df):,}")

    # 4. Truncate to max sequence length (keep most recent)
    df = truncate_sequences(df, user_col="user_id", timestamp_col="timestamp")

    # 5. Stats
    print_stats(df, user_col="user_id", item_col="item_id", name=dataset_name)

    # 6. Save
    save_recbole_format(df, dataset_name,
                        user_col="user_id", item_col="item_id",
                        timestamp_col="timestamp")


def preprocess_amazon_beauty(input_path: str) -> None:
    preprocess_amazon(input_path, dataset_name="amazon_beauty")


def preprocess_amazon_videogames(input_path: str) -> None:
    preprocess_amazon(input_path, dataset_name="amazon_videogames")



# MovieLens-1M
def preprocess_movielens(input_path: str) -> None:
    """
    MovieLens-1M ratings.dat
    Format: UserID::MovieID::Rating::Timestamp
    No k-core needed.
    """
    print("\n[MovieLens-1M] Loading...")
    df = pd.read_csv(
        input_path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    print(f"  Raw interactions : {len(df):,}")

    # 1. Keep positive feedback only
    df = df[df["rating"] >= MIN_RATING].copy()
    print(f"  After rating >= {MIN_RATING} filter : {len(df):,}")

    # 2. Drop duplicates
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["user_id", "item_id"], keep="first"
    )

    # 3. Truncate to max sequence length
    df = truncate_sequences(df, user_col="user_id", timestamp_col="timestamp")

    # 4. Stats
    print_stats(df, user_col="user_id", item_col="item_id", name="MovieLens-1M")

    # 5. Save
    save_recbole_format(df, "movielens_1m",
                        user_col="user_id", item_col="item_id",
                        timestamp_col="timestamp")



# LastFM-1K
def preprocess_lastfm(input_path: str) -> None:
    """
    LastFM-1K TSV file.
    Format: userid  timestamp  artid  artname  traid  traname
    Uses artist-level interactions (traid has ~11% nulls).
    """
    print("\n[LastFM-1K] Loading...")
    df = pd.read_csv(
        input_path,
        sep="\t",
        names=["user_id", "timestamp", "artist_id", "artist_name", "track_id", "track_name"],
        on_bad_lines="skip",
        engine="python",
        usecols=["user_id", "timestamp", "artist_id"],
    )

    print(f"  Raw interactions : {len(df):,}")

    # 1. Drop missing artist IDs
    df = df.dropna(subset=["artist_id"])
    print(f"  After dropping null artists : {len(df):,}")

    # 2. Collapse to unique user-artist pairs (keep earliest timestamp)
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["user_id", "artist_id"], keep="first")
    print(f"  After deduplication : {len(df):,}")

    # 3. Parse timestamps — LastFM uses ISO 8601 strings, convert to unix
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].astype("int64") // 10**9  # seconds

    # 4. K-core filtering
    df = k_core_filtering(df, user_col="user_id", item_col="artist_id", k=K_CORE)
    print(f"  After {K_CORE}-core filtering : {len(df):,}")

    # 5. Truncate to max sequence length
    df = truncate_sequences(df, user_col="user_id", timestamp_col="timestamp")

    # 6. Stats
    print_stats(df, user_col="user_id", item_col="artist_id", name="LastFM-1K")

    # 7. Save
    save_recbole_format(df, "lastfm_1k",
                        user_col="user_id", item_col="artist_id",
                        timestamp_col="timestamp")



if __name__ == "__main__":

    BEAUTY_PATH      = "src/datasets/raw/All_Beauty.csv"
    VIDEOGAMES_PATH  = "src/datasets/raw/Video_Games.csv"
    ML1M_PATH        = "src/datasets/raw/ratings.dat"
    LASTFM_PATH      = "src/datasets/raw/userid-timestamp-artid-artname-traid-traname.tsv"

    preprocess_amazon_beauty(BEAUTY_PATH)
    preprocess_amazon_videogames(VIDEOGAMES_PATH)
    preprocess_movielens(ML1M_PATH)
    preprocess_lastfm(LASTFM_PATH)