"""
Fraud Detection Scoring API
============================
FastAPI application that serves the trained fraud detection model.

Endpoints:
    GET  /health          — Liveness / readiness check
    GET  /model/info      — Model metadata and current threshold
    POST /score           — Score a single transaction
    POST /score/batch     — Score multiple transactions (up to 100)

Design decisions:
    - Model loaded at startup, not per-request (prevents 2s latency per call)
    - Pydantic validation rejects malformed inputs before they reach the model
    - Response includes top risk factors for explainability (regulatory requirement)
    - Confidence bands: APPROVE / REVIEW / DECLINE based on fraud probability
    - All predictions logged to monitoring module
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.monitoring.logger import PredictionLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global app state — model lives here, shared across all requests
# ---------------------------------------------------------------------------
APP_STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load model once.
    Shutdown: release resources.

    The @asynccontextmanager pattern is FastAPI's modern approach.
    Code before 'yield' = startup.
    Code after 'yield' = shutdown.
    Model loading happens ONCE when the server starts, not per-request.
    """
    logger.info("Starting Fraud Detection API...")

    try:
        import joblib
        from pathlib import Path

        model_path = "models/saved/fraud_detector_latest.joblib"

        if not Path(model_path).exists():
            logger.warning(f"Model not found at {model_path}. Run: python scripts/train_model.py")
            APP_STATE["model_bundle"] = None
            APP_STATE["ready"] = False
        else:
            logger.info(f"Loading model from {model_path}...")
            bundle = joblib.load(model_path)
            APP_STATE["model_bundle"] = bundle
            APP_STATE["ready"] = True
            logger.info(
                f"Model loaded | "
                f"Threshold: {bundle['threshold']:.4f} | "
                f"AUC-ROC: {bundle['metrics'].get('auc_roc', 'N/A'):.4f}"
            )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        APP_STATE["ready"] = False

    APP_STATE["prediction_logger"] = PredictionLogger()
    APP_STATE["request_count"] = 0

    logger.info("API ready. Visit http://localhost:8000/docs for Swagger UI")

    yield  # App runs here

    logger.info("Shutting down...")
    APP_STATE.clear()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time transaction fraud scoring using a GradientBoostingClassifier. "
        "Returns fraud probability, decision, and top risk factor explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class TransactionFeatures(BaseModel):
    """
    Input features for a single transaction.

    All features must be numeric. Categorical fields have already been
    encoded by the feature engineering layer (is_online, is_new_country, etc).

    In a production system, the feature engineering layer would compute
    these from the raw transaction event. Here we accept pre-computed
    features directly to keep the API simple and testable.
    """

    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    is_online: int = Field(..., ge=0, le=1, description="1=online, 0=in-person")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day (0=Mon, 6=Sun)")
    is_weekend: int = Field(0, ge=0, le=1)
    is_night: int = Field(0, ge=0, le=1, description="1 if between 22:00–05:59")
    is_high_risk_merchant: int = Field(0, ge=0, le=1)
    is_medium_risk_merchant: int = Field(0, ge=0, le=1)
    amount_z_score: float = Field(0.0, description="(amount - user_mean) / user_std")
    amount_vs_user_max: float = Field(1.0, ge=0, description="amount / user_historical_max")
    days_since_last_txn: float = Field(1.0, ge=0, description="Days since user's last transaction")
    txn_count_1h: int = Field(0, ge=0, description="User's transactions in last hour")
    txn_count_6h: int = Field(0, ge=0, description="User's transactions in last 6 hours")
    txn_count_24h: int = Field(0, ge=0, description="User's transactions in last 24 hours")
    total_amount_1h: float = Field(0.0, ge=0, description="Total USD spent in last 1 hour")
    total_amount_24h: float = Field(0.0, ge=0, description="Total USD spent in last 24 hours")
    is_new_country: int = Field(0, ge=0, le=1, description="1 if country not in user history")
    is_new_merchant_category: int = Field(0, ge=0, le=1)

    @field_validator("amount")
    @classmethod
    def amount_reasonable(cls, v):
        if v > 1_000_000:
            raise ValueError("amount exceeds $1,000,000 maximum")
        return round(v, 2)


class RiskFactor(BaseModel):
    """A single factor contributing to the fraud risk score."""
    feature: str
    value: float
    importance: float
    contribution: str   # Human-readable explanation


class ScoreResponse(BaseModel):
    """
    Full response from the /score endpoint.

    Includes fraud probability, decision, and top risk factor explanations.
    Explainability is critical for compliance — you can't simply decline
    a transaction without being able to state why.
    """
    request_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    decision: str           # APPROVE / REVIEW / DECLINE
    threshold: float
    top_risk_factors: list[RiskFactor]
    processing_time_ms: float
    model_version: str


class BatchTransactionRequest(BaseModel):
    transactions: list[TransactionFeatures] = Field(..., min_length=1, max_length=100)


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]
    total_processed: int
    declined_count: int
    review_count: int
    approved_count: int
    total_processing_time_ms: float


# ---------------------------------------------------------------------------
# Feature column order — must match training
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "amount", "is_online", "hour_of_day", "day_of_week",
    "is_weekend", "is_night", "is_high_risk_merchant", "is_medium_risk_merchant",
    "amount_z_score", "amount_vs_user_max", "days_since_last_txn",
    "txn_count_1h", "txn_count_6h", "txn_count_24h",
    "total_amount_1h", "total_amount_24h",
    "is_new_country", "is_new_merchant_category",
]

# Human-readable descriptions for each feature (for risk factor explanations)
FEATURE_DESCRIPTIONS = {
    "amount": "Transaction amount",
    "is_online": "Online transaction",
    "amount_z_score": "Amount deviation from user's typical spending",
    "txn_count_1h": "Transactions in the last hour",
    "txn_count_6h": "Transactions in the last 6 hours",
    "txn_count_24h": "Transactions in the last 24 hours",
    "total_amount_1h": "Total spent in the last hour",
    "total_amount_24h": "Total spent in the last 24 hours",
    "is_new_country": "Transaction from new country",
    "is_new_merchant_category": "New merchant category",
    "is_high_risk_merchant": "High-risk merchant category",
    "is_night": "Night-time transaction",
    "days_since_last_txn": "Days since last transaction",
    "amount_vs_user_max": "Amount relative to user maximum",
    "is_medium_risk_merchant": "Medium-risk merchant category",
    "is_weekend": "Weekend transaction",
    "hour_of_day": "Hour of day",
    "day_of_week": "Day of week",
}


def _score_transaction(features: TransactionFeatures) -> ScoreResponse:
    """
    Core scoring logic. Called by both /score and /score/batch.

    Steps:
    1. Extract feature vector in the correct column order
    2. Pass through the trained pipeline (scaler + model)
    3. Get fraud probability from predict_proba
    4. Apply decision threshold: APPROVE / REVIEW / DECLINE
    5. Compute top risk factors using feature importances
    """
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]

    bundle = APP_STATE["model_bundle"]
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]
    importances = {item["feature"]: item["importance"] for item in bundle["feature_importances"]}
    trained_at = bundle.get("trained_at", "unknown")

    # Build feature vector in training order
    feature_values = [getattr(features, col) for col in FEATURE_COLUMNS]
    X = np.array(feature_values).reshape(1, -1)

    # Get fraud probability
    fraud_prob = float(pipeline.predict_proba(X)[0][1])

    # Decision logic with confidence bands:
    #   DECLINE: above threshold — block transaction
    #   REVIEW:  in the grey zone (70–100% of threshold) — flag for human review
    #   APPROVE: clearly below threshold
    review_zone = threshold * 0.70
    if fraud_prob >= threshold:
        decision = "DECLINE"
    elif fraud_prob >= review_zone:
        decision = "REVIEW"
    else:
        decision = "APPROVE"

    # -----------------------------------------------------------------------
    # Compute top risk factors
    # Rank features by: importance * |feature_value| (contribution magnitude)
    # This gives an interpretable explanation of WHY this transaction scored high.
    # -----------------------------------------------------------------------
    risk_factors = []
    for col, value in zip(FEATURE_COLUMNS, feature_values):
        importance = importances.get(col, 0.0)
        contribution_score = importance * abs(float(value))
        if importance > 0.01:  # Skip near-zero importance features
            risk_factors.append(RiskFactor(
                feature=col,
                value=round(float(value), 4),
                importance=round(importance, 4),
                contribution=_explain_factor(col, float(value), importance),
            ))

    # Sort by contribution magnitude (importance × |value|)
    risk_factors.sort(
        key=lambda rf: rf.importance * abs(rf.value),
        reverse=True,
    )
    top_risk_factors = risk_factors[:5]

    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        f"[{request_id}] fraud_prob={fraud_prob:.4f} "
        f"decision={decision} "
        f"time={elapsed_ms:.1f}ms"
    )

    # Log to monitoring
    APP_STATE["prediction_logger"].log(
        request_id=request_id,
        fraud_probability=fraud_prob,
        decision=decision,
        latency_ms=elapsed_ms,
    )
    APP_STATE["request_count"] += 1

    return ScoreResponse(
        request_id=request_id,
        fraud_probability=round(fraud_prob, 6),
        decision=decision,
        threshold=round(threshold, 4),
        top_risk_factors=top_risk_factors,
        processing_time_ms=round(elapsed_ms, 2),
        model_version=f"trained_at_{trained_at}",
    )


def _explain_factor(feature: str, value: float, importance: float) -> str:
    """
    Generate a human-readable explanation for a risk factor.

    This is basic rule-based explainability. In production you'd use
    SHAP values for more accurate per-prediction explanations.
    """
    desc = FEATURE_DESCRIPTIONS.get(feature, feature)

    if feature == "amount_z_score":
        if value > 5:
            return f"{desc} is extremely high ({value:.1f} std deviations above normal)"
        elif value > 3:
            return f"{desc} is significantly elevated ({value:.1f} std deviations)"
        elif value > 1.5:
            return f"{desc} is moderately above normal ({value:.1f} std deviations)"
        else:
            return f"{desc}: {value:.2f} (within normal range)"

    elif feature == "txn_count_1h":
        if value >= 10:
            return f"Very high velocity: {int(value)} transactions in last hour (card testing pattern)"
        elif value >= 5:
            return f"Elevated velocity: {int(value)} transactions in last hour"
        else:
            return f"{desc}: {int(value)}"

    elif feature == "is_new_country":
        return "Transaction from a country not previously used by this cardholder" if value == 1 else "Known country"

    elif feature == "is_high_risk_merchant":
        return "High-risk merchant category (crypto, gift cards, wire transfer)" if value == 1 else "Standard merchant"

    elif feature == "amount":
        return f"Large transaction amount: ${value:,.2f}" if value > 1000 else f"Transaction amount: ${value:,.2f}"

    elif feature == "total_amount_1h":
        return f"${value:,.0f} total spent in last hour (unusual spending velocity)" if value > 500 else f"${value:,.0f} in last hour"

    elif feature == "is_night":
        return "Transaction occurred at night (22:00–05:59)" if value == 1 else "Daytime transaction"

    else:
        return f"{desc}: {value}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Operations"])
async def health():
    """
    Kubernetes liveness probe.
    Returns 200 if healthy, 503 if model not loaded.
    """
    if not APP_STATE.get("ready"):
        raise HTTPException(503, "Model not loaded. Run: python scripts/train_model.py")
    return {
        "status": "healthy",
        "model_loaded": True,
        "requests_served": APP_STATE.get("request_count", 0),
    }


@app.get("/model/info", tags=["Operations"])
async def model_info():
    """Return model metadata, threshold, and performance metrics."""
    if not APP_STATE.get("ready"):
        raise HTTPException(503, "Model not loaded")
    bundle = APP_STATE["model_bundle"]
    return {
        "threshold": bundle["threshold"],
        "feature_count": len(bundle["feature_columns"]),
        "metrics": bundle["metrics"],
        "run_id": bundle.get("run_id"),
        "trained_at": bundle.get("trained_at"),
        "top_features": bundle["feature_importances"][:5],
    }


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
async def score_transaction(transaction: TransactionFeatures):
    """
    Score a single transaction for fraud risk.

    Returns:
        - **fraud_probability**: Model's estimate of fraud probability (0–1)
        - **decision**: APPROVE / REVIEW / DECLINE
        - **threshold**: The classification threshold used
        - **top_risk_factors**: Top 5 features driving the score with explanations
        - **processing_time_ms**: Inference latency
    """
    if not APP_STATE.get("ready"):
        raise HTTPException(503, "Model not loaded. Run: python scripts/train_model.py")
    return _score_transaction(transaction)


@app.post("/score/batch", response_model=BatchScoreResponse, tags=["Scoring"])
async def score_batch(request: BatchTransactionRequest):
    """
    Score multiple transactions in one API call.
    More efficient than N sequential /score calls.
    Maximum 100 transactions per batch.
    """
    if not APP_STATE.get("ready"):
        raise HTTPException(503, "Model not loaded")

    t_start = time.perf_counter()
    results = [_score_transaction(txn) for txn in request.transactions]
    total_ms = (time.perf_counter() - t_start) * 1000

    return BatchScoreResponse(
        results=results,
        total_processed=len(results),
        declined_count=sum(1 for r in results if r.decision == "DECLINE"),
        review_count=sum(1 for r in results if r.decision == "REVIEW"),
        approved_count=sum(1 for r in results if r.decision == "APPROVE"),
        total_processing_time_ms=round(total_ms, 2),
    )