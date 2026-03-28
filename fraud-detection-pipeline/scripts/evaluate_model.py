"""
Model Evaluation Script
========================
Loads the trained model and evaluates it on the held-out test set.
Produces a detailed evaluation report saved to JSON.

This script is also used as the CI/CD evaluation gate:
    pytest tests/evaluation/test_model_gate.py

Which fails the build if AUC-ROC < 0.90.

Run:
    python scripts/evaluate_model.py

Output:
    models/saved/evaluation_report.json
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "amount", "is_online", "hour_of_day", "day_of_week",
    "is_weekend", "is_night", "is_high_risk_merchant", "is_medium_risk_merchant",
    "amount_z_score", "amount_vs_user_max", "days_since_last_txn",
    "txn_count_1h", "txn_count_6h", "txn_count_24h",
    "total_amount_1h", "total_amount_24h",
    "is_new_country", "is_new_merchant_category",
]


def evaluate(
    model_path: str = "models/saved/fraud_detector_latest.joblib",
    test_data_path: str = "data/processed/test.csv",
    output_path: str = "models/saved/evaluation_report.json",
) -> dict:
    """
    Full evaluation on held-out test set.

    Why a separate evaluation script?
    - The test set is truly held-out (never seen during training or threshold tuning)
    - Provides unbiased performance estimates
    - The report is logged as a CI artifact
    """
    logger.info(f"Loading model: {model_path}")
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]

    logger.info(f"Loading test data: {test_data_path}")
    test_df = pd.read_csv(test_data_path)

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["is_fraud"].values

    logger.info(f"Test set: {len(test_df):,} rows | Fraud: {y_test.sum()} ({y_test.mean():.3%})")

    # Predict
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Core metrics
    auc_roc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Calibration: accuracy within confidence bands
    calibration = {}
    for threshold_pct in [0.5, 0.7, 0.9, 0.95]:
        mask = y_prob >= threshold_pct
        if mask.sum() > 0:
            precision_at = y_test[mask].mean()
            calibration[f"precision_at_prob_{threshold_pct:.0%}"] = {
                "count": int(mask.sum()),
                "fraud_rate": round(float(precision_at), 4),
                "coverage": round(float(mask.mean()), 4),
            }

    evaluation_report = {
        "model_path": model_path,
        "test_set_size": len(test_df),
        "test_fraud_rate": float(y_test.mean()),
        "threshold_used": threshold,
        "metrics": {
            "auc_roc": round(auc_roc, 4),
            "pr_auc": round(pr_auc, 4),
            "fraud_precision": round(report["1"]["precision"], 4),
            "fraud_recall": round(report["1"]["recall"], 4),
            "fraud_f1": round(report["1"]["f1-score"], 4),
            "accuracy": round(report["accuracy"], 4),
        },
        "confusion_matrix": {
            "true_negatives":  int(cm[0][0]),
            "false_positives": int(cm[0][1]),
            "false_negatives": int(cm[1][0]),
            "true_positives":  int(cm[1][1]),
        },
        "calibration": calibration,
    }

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS (held-out test set)")
    logger.info("=" * 60)
    logger.info(f"  AUC-ROC:         {auc_roc:.4f}")
    logger.info(f"  PR-AUC:          {pr_auc:.4f}")
    logger.info(f"  Fraud Precision: {report['1']['precision']:.4f}")
    logger.info(f"  Fraud Recall:    {report['1']['recall']:.4f}")
    logger.info(f"  Fraud F1:        {report['1']['f1-score']:.4f}")
    logger.info(f"  True Positives:  {int(cm[1][1])} (fraud correctly caught)")
    logger.info(f"  False Negatives: {int(cm[1][0])} (fraud missed)")
    logger.info(f"  False Positives: {int(cm[0][1])} (legitimate declined)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(evaluation_report, f, indent=2)
    logger.info(f"\nEvaluation report saved: {output_path}")

    return evaluation_report


if __name__ == "__main__":
    report = evaluate()