"""
# the script for Feature Engineering for Fraud Detection project
Transforms raw transaction records into model-ready features.

# Key insight: raw transaction data alone is weak for fraud detection.
The most predictive features are DEVIATIONS from a user's normal behaviour.
A $5,000 transaction is suspicious for someone who normally spends $30/day,
but normal for someone who regularly buys wholesale goods.

Feature categories:
    1. Transaction features   - amount, channel, time of day
    2. Merchant risk features - category risk tier
    3. Behavioural deviation  - how unusual is this vs user history?
    4. Velocity features      - transaction count and amount in past N hours
    5. Context features       - new country, night-time, weekend
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd



# High-risk merchant categories (known fraud hotspots)

HIGH_RISK_MERCHANT_CATEGORIES = {
    "gambling",
    "crypto_exchange",
    "wire_transfer",
    "gift_cards",
    "money_order",
    "prepaid_cards",
    "foreign_currency",
}

MEDIUM_RISK_MERCHANT_CATEGORIES = {
    "electronics",
    "jewelry",
    "luxury_goods",
    "online_marketplace",
    "travel",
}


@dataclass
class TransactionRecord:
    """
    Raw transaction data as received from the payment system.

    These are the fields available BEFORE feature engineering.
    Real systems receive these in real-time (Kafka stream, REST webhook, etc).
    """

    transaction_id: str
    user_id: str
    amount: float                        #usd
    merchant_category: str               
    country: str                         
    is_online: bool                     
    timestamp: float                     
    hour_of_day: int                   
    day_of_week: int                     


@dataclass
class EngineeredFeatures:
    """
    Model-ready feature vector produced by engineer_features().

    All features are numeric. Categorical variables are encoded.
    This is what the GradientBoostingClassifier receives.
    """

    #Transaction-level features
    amount: float
    is_online: int                       
    hour_of_day: int
    day_of_week: int
    is_weekend: int                      
    is_night: int                        

    #Merchant risk
    is_high_risk_merchant: int           #if in HIGH_RISK set
    is_medium_risk_merchant: int         #if in MEDIUM_RISK set

    #Behavioural deviation features
    # These required user history. Very predictive but need a lookup layer.
    amount_z_score: float                
    amount_vs_user_max: float            
    days_since_last_txn: float           

    # --- Velocity features ---
    txn_count_1h: int                    # Transactions in last 1 hour
    txn_count_6h: int                    # Transactions in last 6 hours
    txn_count_24h: int                   # Transactions in last 24 hours
    total_amount_1h: float               # Total spent in last 1 hour
    total_amount_24h: float              # Total spent in last 24 hours

    # --- Context features ---
    is_new_country: int                  # 1 if country not in user history
    is_new_merchant_category: int        # 1 if category not in user history


# The feature column order MUST match what the model was trained on.
# This list is the single source of truth used by both training and inference.
FEATURE_COLUMNS = [
    "amount",
    "is_online",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
    "is_high_risk_merchant",
    "is_medium_risk_merchant",
    "amount_z_score",
    "amount_vs_user_max",
    "days_since_last_txn",
    "txn_count_1h",
    "txn_count_6h",
    "txn_count_24h",
    "total_amount_1h",
    "total_amount_24h",
    "is_new_country",
    "is_new_merchant_category",
]


def engineer_features(
    txn: TransactionRecord,
    user_history: Optional[pd.DataFrame] = None,
) -> EngineeredFeatures:
    """
    Convert a raw transaction record into model-ready features.

    Args:
        txn:          The incoming transaction to score.
        user_history: DataFrame of this user's past transactions.
                      Columns: [amount, country, merchant_category, timestamp]
                      If None (new user), behavioural features default to 0.

    Returns:
        EngineeredFeatures dataclass with all numeric features populated.

    Design note:
        We compute behavioural deviation features here because they are
        the most predictive signals. A naive model using only transaction
        features (amount, category) is easily beaten by testing small amounts
        first. Behavioural features catch velocity attacks and unusual patterns.
    """
    now = txn.timestamp

    
    # 1. Time-based features
    is_weekend = int(txn.day_of_week >= 5)          # Sat=5, Sun=6
    is_night = int(txn.hour_of_day < 6 or txn.hour_of_day >= 22)

    
    # 2. Merchant risk features
    
    is_high_risk = int(txn.merchant_category.lower() in HIGH_RISK_MERCHANT_CATEGORIES)
    is_medium_risk = int(txn.merchant_category.lower() in MEDIUM_RISK_MERCHANT_CATEGORIES)

    
    # 3. Behavioural deviation features (require user history)
    
    if user_history is not None and len(user_history) > 0:
        user_amounts = user_history["amount"].values
        user_mean = user_amounts.mean()
        user_std = user_amounts.std() + 1e-8        
        user_max = user_amounts.max()

        # Z-score: how many standard deviations above/below the user's mean?
        # z=0: normal for this user, z=3: very unusual, z=8: almost certainly abnormal
        amount_z_score = (txn.amount - user_mean) / user_std
        amount_vs_max = txn.amount / (user_max + 1e-8)

        # Time since last transaction
        last_txn_time = user_history["timestamp"].max()
        days_since_last = (now - last_txn_time) / 86400.0

        # Velocity: count transactions in rolling windows
        cutoff_1h = now - 3600
        cutoff_6h = now - 21600
        cutoff_24h = now - 86400

        recent_1h = user_history[user_history["timestamp"] >= cutoff_1h]
        recent_6h = user_history[user_history["timestamp"] >= cutoff_6h]
        recent_24h = user_history[user_history["timestamp"] >= cutoff_24h]

        txn_count_1h = len(recent_1h)
        txn_count_6h = len(recent_6h)
        txn_count_24h = len(recent_24h)
        total_amount_1h = float(recent_1h["amount"].sum()) if len(recent_1h) > 0 else 0.0
        total_amount_24h = float(recent_24h["amount"].sum()) if len(recent_24h) > 0 else 0.0

        # New country / category flags
        known_countries = set(user_history["country"].unique())
        known_categories = set(user_history["merchant_category"].unique())
        is_new_country = int(txn.country not in known_countries)
        is_new_category = int(txn.merchant_category not in known_categories)

    else:
        # New user: no historical baseline available.
        # Default z_score=0, velocity=0, new_country=1 (conservative assumption).
        amount_z_score = 0.0
        amount_vs_max = 1.0
        days_since_last = 999.0    
        txn_count_1h = 0
        txn_count_6h = 0
        txn_count_24h = 0
        total_amount_1h = 0.0
        total_amount_24h = 0.0
        is_new_country = 1         
        is_new_category = 1

    return EngineeredFeatures(
        amount=txn.amount,
        is_online=int(txn.is_online),
        hour_of_day=txn.hour_of_day,
        day_of_week=txn.day_of_week,
        is_weekend=is_weekend,
        is_night=is_night,
        is_high_risk_merchant=is_high_risk,
        is_medium_risk_merchant=is_medium_risk,
        amount_z_score=round(amount_z_score, 4),
        amount_vs_user_max=round(amount_vs_max, 4),
        days_since_last_txn=round(days_since_last, 2),
        txn_count_1h=txn_count_1h,
        txn_count_6h=txn_count_6h,
        txn_count_24h=txn_count_24h,
        total_amount_1h=round(total_amount_1h, 2),
        total_amount_24h=round(total_amount_24h, 2),
        is_new_country=is_new_country,
        is_new_merchant_category=is_new_category,
    )


def features_to_array(features: EngineeredFeatures) -> list[float]:
    """
    Convert EngineeredFeatures dataclass to a flat list in FEATURE_COLUMNS order.

    This is what gets passed to model.predict_proba().
    Keeping FEATURE_COLUMNS as the single source of truth prevents
    the classic bug where training and inference use different column orders.
    """
    return [getattr(features, col) for col in FEATURE_COLUMNS]


def dataframe_from_features(features: EngineeredFeatures) -> pd.DataFrame:
    """Wrap features in a DataFrame with correct column names for sklearn."""
    return pd.DataFrame([features_to_array(features)], columns=FEATURE_COLUMNS)