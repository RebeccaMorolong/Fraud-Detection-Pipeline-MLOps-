"""
Production Prediction Logger
==============================
Logs every scoring request to SQLite for monitoring, drift detection,
and delayed label reconciliation.

In production, this data feeds:
- Real-time dashboards (fraud rate, latency, decision distribution)
- Drift detection (comparing current vs training distributions)
- Model retraining triggers (when accuracy degrades below threshold)
- Chargeback reconciliation (linking fraud labels back to predictions)
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    request_id: str
    timestamp: float
    fraud_probability: float
    decision: str           # APPROVE / REVIEW / DECLINE
    latency_ms: float
    true_label: Optional[int] = None   # Added later via label reconciliation


class PredictionLogger:
    """
    Logs fraud scoring predictions to a SQLite database.

    Design:
        SQLite is used for simplicity (zero infrastructure overhead).
        For high-volume production (>1000 req/s): replace with ClickHouse,
        BigQuery, or a managed time-series store.
    """

    def __init__(self, db_path: str = "monitoring/predictions.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()
        logger.info(f"Prediction logger initialised: {db_path}")

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    request_id    TEXT PRIMARY KEY,
                    timestamp     REAL NOT NULL,
                    fraud_prob    REAL NOT NULL,
                    decision      TEXT NOT NULL,
                    latency_ms    REAL NOT NULL,
                    true_label    INTEGER     -- NULL until label is received
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_decision ON predictions(decision)")

    def log(
        self,
        request_id: str,
        fraud_probability: float,
        decision: str,
        latency_ms: float,
    ) -> None:
        """Log a single prediction. Called after every /score request."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO predictions VALUES (?,?,?,?,?,?)",
                (request_id, time.time(), fraud_probability, decision, latency_ms, None),
            )

    def add_label(self, request_id: str, true_label: int) -> None:
        """
        Add ground-truth label to an existing prediction.

        Called when a chargeback is received (confirmed fraud) or
        when a transaction is verified as legitimate.
        This enables delayed accuracy evaluation without needing
        labels at inference time.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE predictions SET true_label=? WHERE request_id=?",
                (true_label, request_id),
            )

    def get_metrics(self, hours: int = 24) -> dict:
        """
        Compute monitoring metrics for the last N hours.

        Key metrics:
        - Average fraud probability (proxy for model confidence)
        - Decision distribution (are we declining too many / too few?)
        - P95/P99 latency (SLA compliance)
        - Labeled accuracy (when ground truth is available)
        """
        cutoff = time.time() - hours * 3600

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM predictions WHERE timestamp >= ?",
                (cutoff,),
            ).fetchall()

        if not rows:
            return {"error": "No predictions in time window", "hours": hours}

        probs = [r["fraud_prob"] for r in rows]
        latencies = [r["latency_ms"] for r in rows]
        decisions = [r["decision"] for r in rows]

        # Accuracy on labeled subset
        labeled = [r for r in rows if r["true_label"] is not None]
        accuracy = None
        if labeled:
            correct = sum(
                1 for r in labeled
                if (r["decision"] != "APPROVE") == bool(r["true_label"])
            )
            accuracy = correct / len(labeled)

        lat_arr = np.array(latencies)

        metrics = {
            "period_hours": hours,
            "total_predictions": len(rows),
            "decision_distribution": {
                "APPROVE": decisions.count("APPROVE"),
                "REVIEW": decisions.count("REVIEW"),
                "DECLINE": decisions.count("DECLINE"),
            },
            "fraud_probability": {
                "mean": round(float(np.mean(probs)), 4),
                "median": round(float(np.median(probs)), 4),
                "p95": round(float(np.percentile(probs, 95)), 4),
            },
            "latency_ms": {
                "mean": round(float(lat_arr.mean()), 2),
                "p95": round(float(np.percentile(lat_arr, 95)), 2),
                "p99": round(float(np.percentile(lat_arr, 99)), 2),
            },
            "labeled_accuracy": round(accuracy, 4) if accuracy is not None else None,
            "alerts": self._check_alerts(probs, latencies, decisions),
        }

        return metrics

    def _check_alerts(
        self, probs: list, latencies: list, decisions: list
    ) -> list[str]:
        """Return triggered alert conditions."""
        alerts = []

        decline_rate = decisions.count("DECLINE") / len(decisions)
        if decline_rate > 0.05:
            alerts.append(f"HIGH_DECLINE_RATE:{decline_rate:.1%} (threshold: 5%)")

        avg_prob = np.mean(probs)
        if avg_prob > 0.15:
            alerts.append(f"HIGH_AVG_FRAUD_PROB:{avg_prob:.3f}")

        p99_latency = np.percentile(latencies, 99)
        if p99_latency > 500:
            alerts.append(f"HIGH_P99_LATENCY:{p99_latency:.0f}ms (threshold: 500ms)")

        return alerts