"""
# Model Training Pipeline

Trains a GradientBoostingClassifier on the synthetic fraud dataset,
handles class imbalance with SMOTE, finds the optimal decision threshold
using Precision-Recall curves, and logs everything to MLflow.

Run:
    python scripts/train_model.py

Output:
    models/saved/fraud_detector_v{timestamp}.joblib   — trained pipeline
    models/saved/fraud_detector_latest.joblib          — symlink / copy to latest
    mlruns/                                            — MLflow experiment logs
"""

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler

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

LABEL_COLUMN = "is_fraud"


def load_data(data_dir: str = "data/processed") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation sets from processed CSV files."""
    train = pd.read_csv(f"{data_dir}/train.csv")
    val = pd.read_csv(f"{data_dir}/val.csv")
    logger.info(f"Train: {len(train):,} rows | Val: {len(val):,} rows")
    logger.info(f"Train fraud rate: {train[LABEL_COLUMN].mean():.3%}")
    return train, val


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Find the classification threshold that maximises F1 score on the validation set.

    Why not use 0.5?
    With severe class imbalance (0.5% fraud), the model's calibration makes
    0.5 a terrible threshold. Almost no fraud predictions would exceed 0.5.
    We use Precision-Recall curve to find the threshold that maximises F1,
    balancing catching fraud (recall) vs false alarms (precision).

    Args:
        y_true: True binary labels
        y_prob: Predicted fraud probabilities

    Returns:
        (optimal_threshold, best_f1)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # F1 = 2 * precision * recall / (precision + recall)
    # The arrays have one more element than thresholds (sklearn quirk), so we slice
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = f1_scores.argmax()
    optimal_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    logger.info(f"Optimal threshold: {optimal_threshold:.4f} | Best F1: {best_f1:.4f}")
    return optimal_threshold, best_f1


def train(
    data_dir: str = "data/processed",
    model_dir: str = "models/saved",
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    smote_k_neighbors: int = 5,
    experiment_name: str = "fraud-detection",
) -> dict:
    """
    Full training pipeline with MLflow tracking.

    Architecture decisions:
    - GradientBoostingClassifier: strong baseline for tabular fraud data.
      Handles non-linear interactions between velocity and amount features.
      XGBoost would also work well but GBM is stdlib scikit-learn.
    - SMOTE oversampling: creates synthetic minority (fraud) examples by
      interpolating between existing fraud cases in feature space.
      Preferred over simple duplication because it adds diversity.
    - StandardScaler: GBM is not sensitive to scale but good practice.
      Required if you later add logistic regression to the ensemble.
    - Pipeline: chains SMOTE + scaler + model into one object.
      Ensures SMOTE only sees training data (no data leakage).

    Args:
        data_dir:        Path to processed CSV files
        model_dir:       Where to save trained model artifacts
        n_estimators:    Number of boosting trees
        max_depth:       Maximum tree depth (higher = more capacity = more overfit risk)
        learning_rate:   Step size shrinkage (lower = slower convergence, better generalisation)
        subsample:       Fraction of training samples used per tree (reduces overfitting)
        smote_k_neighbors: k for SMOTE synthetic sample generation
        experiment_name: MLflow experiment name

    Returns:
        Dict with model path, metrics, and threshold.
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(experiment_name)

    train_df, val_df = load_data(data_dir)

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[LABEL_COLUMN].values
    X_val = val_df[FEATURE_COLUMNS].values
    y_val = val_df[LABEL_COLUMN].values

    logger.info(
        f"Class distribution — Train fraud: {y_train.sum():,} / {len(y_train):,} "
        f"({y_train.mean():.3%})"
    )

    run_name = f"gbt-n{n_estimators}-d{max_depth}-lr{learning_rate}"

    with mlflow.start_run(run_name=run_name) as run:
        # -----------------------------------------------------------------------
        # Log hyperparameters
        # -----------------------------------------------------------------------
        params = {
            "model_type": "GradientBoostingClassifier",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "smote_k_neighbors": smote_k_neighbors,
            "n_features": len(FEATURE_COLUMNS),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "train_fraud_rate": float(y_train.mean()),
        }
        mlflow.log_params(params)

        # -----------------------------------------------------------------------
        # Build pipeline: SMOTE -> StandardScaler -> GBM
        # SMOTE is applied INSIDE the pipeline so it only touches training data.
        # -----------------------------------------------------------------------
        logger.info("Building pipeline: SMOTE + StandardScaler + GradientBoostingClassifier")
        t0 = time.time()

        pipeline = ImbPipeline([
            ("smote", SMOTE(
                k_neighbors=smote_k_neighbors,
                random_state=42,
                sampling_strategy="auto",   # Oversample fraud until balanced
            )),
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                min_samples_leaf=20,        # Prevents overfitting on rare fraud patterns
                random_state=42,
                verbose=1,
            )),
        ])

        logger.info("Training... (this may take 1–3 minutes)")
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0
        logger.info(f"Training complete in {train_time:.1f}s")

        # -----------------------------------------------------------------------
        # Evaluate on validation set
        # -----------------------------------------------------------------------
        y_prob = pipeline.predict_proba(X_val)[:, 1]   # Fraud probability

        # Core metrics
        auc_roc = roc_auc_score(y_val, y_prob)
        avg_precision = average_precision_score(y_val, y_prob)   # Area under PR curve

        logger.info(f"AUC-ROC: {auc_roc:.4f}")
        logger.info(f"Average Precision (PR-AUC): {avg_precision:.4f}")

        # Find optimal threshold
        optimal_threshold, best_f1 = find_optimal_threshold(y_val, y_prob)

        # Evaluate at optimal threshold
        y_pred = (y_prob >= optimal_threshold).astype(int)
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

        fraud_metrics = report.get("1", {})
        precision_fraud = fraud_metrics.get("precision", 0)
        recall_fraud = fraud_metrics.get("recall", 0)
        f1_fraud = fraud_metrics.get("f1-score", 0)

        logger.info(
            f"At threshold={optimal_threshold:.4f}: "
            f"Precision={precision_fraud:.4f} | "
            f"Recall={recall_fraud:.4f} | "
            f"F1={f1_fraud:.4f}"
        )

        # -----------------------------------------------------------------------
        # Log metrics to MLflow
        # -----------------------------------------------------------------------
        mlflow.log_metrics({
            "auc_roc": round(auc_roc, 4),
            "pr_auc": round(avg_precision, 4),
            "fraud_precision": round(precision_fraud, 4),
            "fraud_recall": round(recall_fraud, 4),
            "fraud_f1": round(f1_fraud, 4),
            "optimal_threshold": round(optimal_threshold, 4),
            "train_time_seconds": round(train_time, 1),
        })

        # -----------------------------------------------------------------------
        # Feature importances — log as artifact
        # -----------------------------------------------------------------------
        gbt_model = pipeline.named_steps["model"]
        importances = gbt_model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        logger.info("\nTop 10 Feature Importances:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']:35s}: {row['importance']:.4f}")

        importance_path = f"{model_dir}/feature_importances.json"
        importance_df.to_json(importance_path, orient="records", indent=2)
        mlflow.log_artifact(importance_path)

        # -----------------------------------------------------------------------
        # Save model artifact
        # -----------------------------------------------------------------------
        timestamp = int(time.time())
        model_filename = f"fraud_detector_v{timestamp}.joblib"
        model_path = f"{model_dir}/{model_filename}"
        latest_path = f"{model_dir}/fraud_detector_latest.joblib"

        # Bundle: model + threshold + feature columns + metadata
        model_bundle = {
            "pipeline": pipeline,
            "threshold": optimal_threshold,
            "feature_columns": FEATURE_COLUMNS,
            "feature_importances": importance_df.to_dict("records"),
            "metrics": {
                "auc_roc": auc_roc,
                "pr_auc": avg_precision,
                "fraud_f1": f1_fraud,
                "fraud_precision": precision_fraud,
                "fraud_recall": recall_fraud,
            },
            "run_id": run.info.run_id,
            "trained_at": timestamp,
        }

        joblib.dump(model_bundle, model_path)
        joblib.dump(model_bundle, latest_path)
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Latest link: {latest_path}")

        # Log model to MLflow model registry
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="fraud-detector",
        )

        run_id = run.info.run_id
        logger.info(f"\nMLflow Run ID: {run_id}")
        logger.info("View in MLflow UI: mlflow ui --host 0.0.0.0 --port 5000")

    return {
        "model_path": model_path,
        "run_id": run_id,
        "threshold": optimal_threshold,
        "metrics": {
            "auc_roc": auc_roc,
            "pr_auc": avg_precision,
            "fraud_f1": f1_fraud,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--model-dir", default="models/saved")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--experiment", default="fraud-detection")
    args = parser.parse_args()

    result = train(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        experiment_name=args.experiment,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model: {result['model_path']}")
    print(f"  Threshold: {result['threshold']:.4f}")
    print(f"  AUC-ROC: {result['metrics']['auc_roc']:.4f}")
    print(f"  PR-AUC: {result['metrics']['pr_auc']:.4f}")
    print(f"  Fraud F1: {result['metrics']['fraud_f1']:.4f}")
    print(f"  MLflow Run: {result['run_id']}")