import os
import pandas as pd
from src.utils import path_builder

# Global config
K_CORE = 5
OUTPUT_DIR = path_builder('src/datasets/preprocessed')


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

    Note: sequences are NOT truncated here. RecBole's sequential
    dataloader handles windowing/padding via MAX_ITEM_LIST_LENGTH
    in the model config, and generates leave-one-out training
    instances from the FULL history. Truncating upstream would
    throw away real training signal (especially for long-sequence
    datasets like ML-1M) and reduce the number of sliding-window
    instances RecBole derives per user.
    """
    os.makedirs(os.path.join(OUTPUT_DIR, dataset_name), exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, dataset_name, f"{dataset_name}.inter")

    out_df = df[[user_col, item_col, timestamp_col]].copy()
    out_df.columns = ["user_id:token", "item_id:token", "timestamp:float"]

    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"  Saved -> {out_path}")


# Amazon (Beauty & Video Games)
def preprocess_amazon(input_path: str, dataset_name: str) -> None:
    """
    Shared preprocessing for Amazon ratings-only CSV files.
    Expected columns : item_id, user_id, rating, timestamp

    All interactions are treated as implicit positive feedback,
    regardless of rating value this matches standard practice
    in sequential rec papers (SASRec, S3-Rec, CL4SRec, DuoRec):
    the fact a user interacted with an item is the signal, not
    the rating magnitude. Filtering by rating threshold would
    shrink sequences, drop users entirely, and break comparability
    with published benchmark numbers.
    """
    print(f"\n[Amazon - {dataset_name}] Loading...")
    df = pd.read_csv(
        input_path,
        header=None,
        names=["item_id", "user_id", "rating", "timestamp"],
    )
    print(f"  Raw interactions : {len(df):,}")

    # 1. Drop duplicates (same user-item pair, keep earliest)
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["user_id", "item_id"], keep="first"
    )
    print(f"  After dedup : {len(df):,}")

    # 2. K-core filtering
    df = k_core_filtering(df, user_col="user_id", item_col="item_id", k=K_CORE)
    print(f"  After {K_CORE}-core filtering : {len(df):,}")

    # 3. Stats
    print_stats(df, user_col="user_id", item_col="item_id", name=dataset_name)

    # 4. Save
    save_recbole_format(df, dataset_name,
                         user_col="user_id", item_col="item_id",
                         timestamp_col="timestamp")


def preprocess_amazon_musical_instruments(input_path: str) -> None:
    preprocess_amazon(input_path, dataset_name="amazon_musical_instruments")


def preprocess_amazon_videogames(input_path: str) -> None:
    preprocess_amazon(input_path, dataset_name="amazon_videogames")


# MovieLens-1M
def preprocess_movielens(input_path: str) -> None:
    """
    MovieLens-1M ratings.dat
    Format: UserID::MovieID::Rating::Timestamp

    All ratings kept as implicit feedback (no MIN_RATING filter),
    consistent with the Amazon preprocessing above. No k-core
    needed since ML-1M is already densely rated.
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

    # 1. Drop duplicates
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["user_id", "item_id"], keep="first"
    )
    print(f"  After dedup : {len(df):,}")

    # 2. Stats
    print_stats(df, user_col="user_id", item_col="item_id", name="MovieLens-1M")

    # 3. Save
    save_recbole_format(df, "movielens_1m",
                         user_col="user_id", item_col="item_id",
                         timestamp_col="timestamp")


# LastFM-1K
def preprocess_lastfm(input_path: str) -> None:
    """
    LastFM-1K TSV file.
    Format: userid  timestamp  artid  artname  traid  traname
    Uses artist-level interactions (traid has approx. 11% nulls, and
    artist-level aggregation is what gets item counts in line
    with the published benchmark stats for this dataset).
    """
    print("\n[LastFM-1K] Loading...")

    # Count raw lines up front so we can report how many get
    # silently skipped by on_bad_lines="skip" below.
    with open(input_path, encoding="utf-8", errors="ignore") as f:
        n_raw_lines = sum(1 for _ in f)

    df = pd.read_csv(
        input_path,
        sep="\t",
        names=["user_id", "timestamp", "artist_id", "artist_name", "track_id", "track_name"],
        on_bad_lines="skip",
        engine="python",
        usecols=["user_id", "timestamp", "artist_id"],
    )
    print(f"  Raw lines    : {n_raw_lines:,}")
    print(f"  Parsed rows  : {len(df):,} ({n_raw_lines - len(df):,} malformed lines skipped)")

    # 1. Drop missing artist IDs
    df = df.dropna(subset=["artist_id"])
    print(f"  After dropping null artists : {len(df):,}")

    # 2. Collapse to unique user-artist pairs (keep earliest timestamp)
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["user_id", "artist_id"], keep="first")
    print(f"  After deduplication : {len(df):,}")

    # 3. Parse timestamps -- LastFM uses ISO 8601 strings, convert to unix
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].astype("int64") // 10**9  # seconds

    # 4. K-core filtering
    df = k_core_filtering(df, user_col="user_id", item_col="artist_id", k=K_CORE)
    print(f"  After {K_CORE}-core filtering : {len(df):,}")

    # 5. Stats
    print_stats(df, user_col="user_id", item_col="artist_id", name="LastFM-1K")

    # 6. Save
    save_recbole_format(df, "lastfm_1k",
                         user_col="user_id", item_col="artist_id",
                         timestamp_col="timestamp")


if __name__ == "__main__":

    INSTRUMENTS_PATH = path_builder('src/datasets/raw/Musical_Instruments.csv')
    VIDEOGAMES_PATH  = path_builder('src/datasets/raw/Video_Games.csv')
    ML1M_PATH        = path_builder('src/datasets/raw/ratings.dat')
    LASTFM_PATH      = path_builder('src/datasets/raw/userid-timestamp-artid-artname-traid-traname.tsv')

    preprocess_amazon_musical_instruments(INSTRUMENTS_PATH)
    preprocess_amazon_videogames(VIDEOGAMES_PATH)
    preprocess_movielens(ML1M_PATH)
    preprocess_lastfm(LASTFM_PATH)