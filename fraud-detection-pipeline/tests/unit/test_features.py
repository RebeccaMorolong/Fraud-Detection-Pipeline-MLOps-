"""
Unit Tests — Feature Engineering
===================================
Tests the feature engineering module in isolation.
No model or API required — just pure Python functions.

Run:
    pytest tests/unit/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    FEATURE_COLUMNS,
    TransactionRecord,
    engineer_features,
    features_to_array,
    HIGH_RISK_MERCHANT_CATEGORIES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def normal_transaction():
    return TransactionRecord(
        transaction_id="txn_001",
        user_id="user_001",
        amount=45.00,
        merchant_category="grocery",
        country="US",
        is_online=False,
        timestamp=1_700_000_000.0,
        hour_of_day=14,
        day_of_week=2,
    )


@pytest.fixture
def suspicious_transaction():
    return TransactionRecord(
        transaction_id="txn_002",
        user_id="user_002",
        amount=5000.00,
        merchant_category="crypto_exchange",
        country="RO",                      # New country
        is_online=True,
        timestamp=1_700_000_000.0,
        hour_of_day=2,                     # Night
        day_of_week=6,                     # Sunday
    )


@pytest.fixture
def rich_user_history():
    """User with 30 days of history for behavioural feature testing."""
    now = 1_700_000_000.0
    records = []
    for i in range(60):
        records.append({
            "amount": np.random.uniform(20, 80),
            "country": "US",
            "merchant_category": "grocery",
            "timestamp": now - (i * 86400 / 2),   # Every 12 hours for 30 days
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Tests: feature engineering
# ---------------------------------------------------------------------------
class TestEngineerFeatures:
    def test_returns_correct_number_of_features(self, normal_transaction):
        features = engineer_features(normal_transaction)
        feature_array = features_to_array(features)
        assert len(feature_array) == len(FEATURE_COLUMNS)

    def test_normal_transaction_produces_low_risk_features(self, normal_transaction, rich_user_history):
        features = engineer_features(normal_transaction, user_history=rich_user_history)
        assert features.is_high_risk_merchant == 0
        assert features.is_night == 0
        assert features.is_new_country == 0
        assert features.txn_count_1h == 0    # No recent transactions

    def test_suspicious_transaction_produces_high_risk_features(self, suspicious_transaction, rich_user_history):
        features = engineer_features(suspicious_transaction, user_history=rich_user_history)
        assert features.is_high_risk_merchant == 1    # crypto_exchange is high risk
        assert features.is_night == 1                 # 2am is night
        assert features.is_new_country == 1           # RO not in history (history is US only)

    def test_high_amount_produces_high_z_score(self, rich_user_history):
        """A $5000 transaction for a user who normally spends $45 should have high z-score."""
        txn = TransactionRecord(
            transaction_id="txn_003",
            user_id="user_003",
            amount=5000.00,            # Way above normal
            merchant_category="grocery",
            country="US",
            is_online=False,
            timestamp=1_700_000_000.0,
            hour_of_day=14,
            day_of_week=2,
        )
        features = engineer_features(txn, user_history=rich_user_history)
        assert features.amount_z_score > 5.0   # Should be very high

    def test_new_user_has_safe_defaults(self, normal_transaction):
        """New users (no history) should have conservative default features."""
        features = engineer_features(normal_transaction, user_history=None)
        assert features.txn_count_1h == 0
        assert features.txn_count_24h == 0
        assert features.days_since_last_txn == 999.0   # Signals new account
        assert features.is_new_country == 1             # Conservative assumption

    def test_weekend_detection(self):
        """Saturday (day_of_week=5) should produce is_weekend=1."""
        txn = TransactionRecord(
            transaction_id="txn_004",
            user_id="user_004",
            amount=50.0,
            merchant_category="grocery",
            country="US",
            is_online=False,
            timestamp=1_700_000_000.0,
            hour_of_day=12,
            day_of_week=5,   # Saturday
        )
        features = engineer_features(txn)
        assert features.is_weekend == 1

    def test_night_detection(self):
        """2am should produce is_night=1."""
        txn = TransactionRecord(
            transaction_id="txn_005",
            user_id="user_005",
            amount=50.0,
            merchant_category="grocery",
            country="US",
            is_online=True,
            timestamp=1_700_000_000.0,
            hour_of_day=2,
            day_of_week=1,
        )
        features = engineer_features(txn)
        assert features.is_night == 1

    def test_high_risk_merchant_categories(self):
        """Each high-risk category should produce is_high_risk_merchant=1."""
        for category in HIGH_RISK_MERCHANT_CATEGORIES:
            txn = TransactionRecord(
                transaction_id="txn_006",
                user_id="user_006",
                amount=100.0,
                merchant_category=category,
                country="US",
                is_online=True,
                timestamp=1_700_000_000.0,
                hour_of_day=12,
                day_of_week=1,
            )
            features = engineer_features(txn)
            assert features.is_high_risk_merchant == 1, f"Failed for category: {category}"

    def test_velocity_features_computed_correctly(self):
        """Velocity features should count transactions in correct time windows."""
        now = 1_700_000_000.0
        user_history = pd.DataFrame({
            "amount": [50, 50, 50, 50, 50],
            "country": ["US"] * 5,
            "merchant_category": ["grocery"] * 5,
            "timestamp": [
                now - 600,       # 10 min ago   — within 1h
                now - 1800,      # 30 min ago   — within 1h
                now - 7200,      # 2 hours ago  — within 6h, NOT 1h
                now - 18000,     # 5 hours ago  — within 6h
                now - 90000,     # 25 hours ago — outside all windows
            ],
        })
        txn = TransactionRecord(
            transaction_id="txn_007",
            user_id="user_007",
            amount=50.0,
            merchant_category="grocery",
            country="US",
            is_online=False,
            timestamp=now,
            hour_of_day=12,
            day_of_week=1,
        )
        features = engineer_features(txn, user_history=user_history)
        assert features.txn_count_1h == 2    # Only the first two
        assert features.txn_count_6h == 4    # First four
        assert features.txn_count_24h == 4   # Same (fifth is outside 24h)


class TestFeatureColumns:
    def test_feature_columns_are_consistent(self, normal_transaction):
        """features_to_array should produce values in FEATURE_COLUMNS order."""
        features = engineer_features(normal_transaction)
        array = features_to_array(features)

        # Each value should match the corresponding attribute
        for i, col in enumerate(FEATURE_COLUMNS):
            expected = getattr(features, col)
            assert array[i] == expected, f"Mismatch at column {col}: {array[i]} != {expected}"

    def test_no_nan_values(self, normal_transaction, rich_user_history):
        """Feature array should never contain NaN values."""
        features = engineer_features(normal_transaction, user_history=rich_user_history)
        array = features_to_array(features)
        assert not any(np.isnan(v) for v in array), "NaN found in feature vector"

    def test_no_inf_values(self, suspicious_transaction):
        """Feature array should never contain infinite values."""
        features = engineer_features(suspicious_transaction)
        array = features_to_array(features)
        assert not any(np.isinf(v) for v in array), "Inf found in feature vector"