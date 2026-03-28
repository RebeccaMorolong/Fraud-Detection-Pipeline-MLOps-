"""
Generated a realistic synthetic fraud dataset for training and evaluation.

Real fraud datasets are confidential and cannot be published to GitHub.
This generator creates statistically realistic data that mimics the
class imbalance (0.5% fraud rate) and feature distributions of real datasets
like the Kaggle Credit Card Fraud dataset.

Run:
    python scripts/generate_data.py

Output:
    data/raw/transactions.csv        — full dataset (100,000 rows)
    data/processed/train.csv         — 80% split
    data/processed/val.csv           — 10% split
    data/processed/test.csv          — 10% split (held-out evaluation)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURE_COLUMNS = [
    "amount", "is_online", "hour_of_day", "day_of_week",
    "is_weekend", "is_night", "is_high_risk_merchant", "is_medium_risk_merchant",
    "amount_z_score", "amount_vs_user_max", "days_since_last_txn",
    "txn_count_1h", "txn_count_6h", "txn_count_24h",
    "total_amount_1h", "total_amount_24h",
    "is_new_country", "is_new_merchant_category",
]


def generate_legitimate_transactions(n: int) -> pd.DataFrame:
    """
    Generate realistic legitimate transaction records.

    Legitimate transactions:
    - Small to moderate amounts (mean $45, right-skewed)
    - Low z-scores (normal for this user)
    - Low velocity
    - Known country and merchant category
    """
    return pd.DataFrame({
        "amount":                 np.abs(np.random.lognormal(mean=3.5, sigma=1.2, size=n)).clip(1, 2000),
        "is_online":              np.random.binomial(1, 0.45, n),
        "hour_of_day":            np.random.choice(range(24), n, p=_hour_distribution()),
        "day_of_week":            np.random.randint(0, 7, n),
        "is_weekend":             np.random.binomial(1, 0.28, n),
        "is_night":               np.random.binomial(1, 0.08, n),
        "is_high_risk_merchant":  np.random.binomial(1, 0.03, n),
        "is_medium_risk_merchant":np.random.binomial(1, 0.12, n),
        "amount_z_score":         np.random.normal(0.1, 0.8, n),          # Low z-scores
        "amount_vs_user_max":     np.random.uniform(0.05, 0.8, n),
        "days_since_last_txn":    np.abs(np.random.exponential(2.5, n)),
        "txn_count_1h":           np.random.poisson(0.3, n),
        "txn_count_6h":           np.random.poisson(1.2, n),
        "txn_count_24h":          np.random.poisson(3.0, n),
        "total_amount_1h":        np.abs(np.random.exponential(20, n)),
        "total_amount_24h":       np.abs(np.random.exponential(80, n)),
        "is_new_country":         np.random.binomial(1, 0.04, n),
        "is_new_merchant_category": np.random.binomial(1, 0.06, n),
        "is_fraud":               np.zeros(n, dtype=int),
    })


def generate_fraudulent_transactions(n: int) -> pd.DataFrame:
    """
    Generate realistic fraudulent transaction records.

    Fraudulent transactions have distinct patterns:
    - High amounts OR very low amounts (card testing)
    - High z-scores (unusual for this user)
    - High velocity (multiple rapid transactions)
    - Often new country or high-risk merchant
    - Skewed towards night-time and online
    """
    # Mix of fraud patterns: testing, takeover, new account
    n_high_value = int(n * 0.55)   # High-value account takeover
    n_card_test = int(n * 0.25)    # Card testing (small amounts first)
    n_mixed = n - n_high_value - n_card_test

    def _base(size):
        return {
            "hour_of_day":            np.random.choice(range(24), size, p=_hour_distribution(fraud=True)),
            "day_of_week":            np.random.randint(0, 7, size),
            "is_weekend":             np.random.binomial(1, 0.35, size),
        }

    # High-value takeover
    hv = _base(n_high_value)
    hv.update({
        "amount":                 np.abs(np.random.lognormal(mean=6.5, sigma=1.0, size=n_high_value)).clip(500, 10000),
        "is_online":              np.random.binomial(1, 0.82, n_high_value),
        "is_night":               np.random.binomial(1, 0.45, n_high_value),
        "is_high_risk_merchant":  np.random.binomial(1, 0.38, n_high_value),
        "is_medium_risk_merchant":np.random.binomial(1, 0.25, n_high_value),
        "amount_z_score":         np.abs(np.random.normal(6.0, 2.5, n_high_value)).clip(2, 15),
        "amount_vs_user_max":     np.random.uniform(1.2, 5.0, n_high_value),
        "days_since_last_txn":    np.abs(np.random.exponential(0.5, n_high_value)),
        "txn_count_1h":           np.random.poisson(4.5, n_high_value),
        "txn_count_6h":           np.random.poisson(8.0, n_high_value),
        "txn_count_24h":          np.random.poisson(12.0, n_high_value),
        "total_amount_1h":        np.abs(np.random.exponential(1500, n_high_value)),
        "total_amount_24h":       np.abs(np.random.exponential(3000, n_high_value)),
        "is_new_country":         np.random.binomial(1, 0.72, n_high_value),
        "is_new_merchant_category": np.random.binomial(1, 0.55, n_high_value),
    })

    # Card testing (small amounts)
    ct = _base(n_card_test)
    ct.update({
        "amount":                 np.abs(np.random.uniform(0.5, 15, n_card_test)),
        "is_online":              np.ones(n_card_test, dtype=int),
        "is_night":               np.random.binomial(1, 0.35, n_card_test),
        "is_high_risk_merchant":  np.random.binomial(1, 0.15, n_card_test),
        "is_medium_risk_merchant":np.random.binomial(1, 0.20, n_card_test),
        "amount_z_score":         np.random.normal(-1.5, 0.8, n_card_test),   # Low amount
        "amount_vs_user_max":     np.random.uniform(0.01, 0.1, n_card_test),
        "days_since_last_txn":    np.abs(np.random.exponential(0.1, n_card_test)),
        "txn_count_1h":           np.random.poisson(8.0, n_card_test),        # High velocity!
        "txn_count_6h":           np.random.poisson(15.0, n_card_test),
        "txn_count_24h":          np.random.poisson(20.0, n_card_test),
        "total_amount_1h":        np.abs(np.random.uniform(1, 50, n_card_test)),
        "total_amount_24h":       np.abs(np.random.uniform(5, 200, n_card_test)),
        "is_new_country":         np.random.binomial(1, 0.60, n_card_test),
        "is_new_merchant_category": np.random.binomial(1, 0.45, n_card_test),
    })

    # Mixed pattern fraud
    mx = _base(n_mixed)
    mx.update({
        "amount":                 np.abs(np.random.lognormal(mean=5.0, sigma=1.5, size=n_mixed)).clip(10, 8000),
        "is_online":              np.random.binomial(1, 0.70, n_mixed),
        "is_night":               np.random.binomial(1, 0.35, n_mixed),
        "is_high_risk_merchant":  np.random.binomial(1, 0.25, n_mixed),
        "is_medium_risk_merchant":np.random.binomial(1, 0.20, n_mixed),
        "amount_z_score":         np.abs(np.random.normal(4.0, 2.0, n_mixed)).clip(1, 12),
        "amount_vs_user_max":     np.random.uniform(0.5, 3.0, n_mixed),
        "days_since_last_txn":    np.abs(np.random.exponential(1.0, n_mixed)),
        "txn_count_1h":           np.random.poisson(3.0, n_mixed),
        "txn_count_6h":           np.random.poisson(6.0, n_mixed),
        "txn_count_24h":          np.random.poisson(10.0, n_mixed),
        "total_amount_1h":        np.abs(np.random.exponential(500, n_mixed)),
        "total_amount_24h":       np.abs(np.random.exponential(1000, n_mixed)),
        "is_new_country":         np.random.binomial(1, 0.55, n_mixed),
        "is_new_merchant_category": np.random.binomial(1, 0.40, n_mixed),
    })

    frames = []
    for d in [hv, ct, mx]:
        d["is_fraud"] = 1
        frames.append(pd.DataFrame(d))

    return pd.concat(frames, ignore_index=True)


def _hour_distribution(fraud: bool = False) -> list[float]:
    """
    Probability distribution over hours 0–23.

    Legitimate: peaks during business hours (9am–8pm).
    Fraudulent: skewed towards night (10pm–4am when humans aren't watching).
    """
    if fraud:
        weights = np.array([
            3, 3, 4, 5, 4, 3,    # 0–5am: elevated for fraud
            2, 2, 2, 2, 2, 2,    # 6–11am
            2, 2, 2, 2, 2, 2,    # 12–5pm
            2, 3, 3, 4, 4, 4,    # 6–11pm: also elevated
        ], dtype=float)
    else:
        weights = np.array([
            1, 1, 1, 1, 1, 1,    # 0–5am: low
            2, 3, 4, 5, 5, 5,    # 6–11am: rising
            5, 5, 5, 5, 5, 5,    # 12–5pm: peak
            5, 5, 4, 3, 2, 2,    # 6–11pm: tapering
        ], dtype=float)
    return (weights / weights.sum()).tolist()


def generate_dataset(n_legitimate: int = 99_500, n_fraud: int = 500) -> pd.DataFrame:
    """
    Generate and combine legitimate + fraudulent transactions.

    Args:
        n_legitimate: Number of legitimate transactions
        n_fraud:      Number of fraudulent transactions

    Returns:
        Shuffled DataFrame with is_fraud label column.
    """
    logger.info(f"Generating {n_legitimate:,} legitimate transactions...")
    legit = generate_legitimate_transactions(n_legitimate)

    logger.info(f"Generating {n_fraud:,} fraudulent transactions...")
    fraud = generate_fraudulent_transactions(n_fraud)

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    fraud_rate = df["is_fraud"].mean()
    logger.info(
        f"Dataset: {len(df):,} total rows | "
        f"Fraud rate: {fraud_rate:.2%} ({df['is_fraud'].sum():,} fraudulent)"
    )
    return df


def save_splits(df: pd.DataFrame, output_dir: str = "data") -> None:
    """Save raw data and train/val/test splits."""
    Path(f"{output_dir}/raw").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/processed").mkdir(parents=True, exist_ok=True)

    # Save full raw dataset
    raw_path = f"{output_dir}/raw/transactions.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Saved raw dataset: {raw_path}")

    # Stratified split: preserves fraud rate in each split
    train_val, test = train_test_split(df, test_size=0.10, random_state=RANDOM_SEED, stratify=df["is_fraud"])
    train, val = train_test_split(train_val, test_size=0.111, random_state=RANDOM_SEED, stratify=train_val["is_fraud"])

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = f"{output_dir}/processed/{split_name}.csv"
        split_df.to_csv(path, index=False)
        fraud_count = split_df["is_fraud"].sum()
        logger.info(
            f"Saved {split_name}: {len(split_df):,} rows | "
            f"fraud={fraud_count} ({fraud_count/len(split_df):.2%})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic fraud detection dataset")
    parser.add_argument("--n-legit", type=int, default=99_500, help="Number of legitimate transactions")
    parser.add_argument("--n-fraud", type=int, default=500, help="Number of fraudulent transactions")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    df = generate_dataset(n_legitimate=args.n_legit, n_fraud=args.n_fraud)
    save_splits(df, output_dir=args.output_dir)
    logger.info("Dataset generation complete.")