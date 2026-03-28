# Real Time-Fraud-Detection Pipeline (mlops)

# The Business Problem

Payment fraud costs global businesses over $40 billion annually. Banks and fintech companies need to score every transaction in under 100ms to approve or flag it. The challenge: fraud patterns change weekly. Models must be retrained frequently and deployed with zero downtime.

**The AI solution:** I Built a real-time scoring API that classifies transactions as fraudulent or legitimate, with a full MLOps pipeline: automated retraining when performance drops, model versioning, drift detection, A/B testing between model versions.

**Companies building this:** Stripe Radar, PayPal fraud team, Mastercard Decision Intelligence, Feedzai, every bank's fraud analytics department.

---

# Architecture
---
    Transaction Event (JSON)
          |
          v
    [Feature Engineering Service]
      - velocity features (txns in last 1/5/24 hours)
      - behavioral deviation (is amount unusual for this user?)
      - merchant category encoding
           |
           v
    [Fraud Scoring API] ——> Score (0.0 - 1.0) + Explanation
           |
           v
    [Decision: Approve / Flag / Decline]
           |
           v
    [Feedback Loop] ——> Confirmed fraud labels -> retrain trigger
           |
           v
    [Monitoring] ——> Drift detection + accuracy tracking


    [Feedback Loop] ——> Confirmed fraud labels -> retrain trigger
           |
           v
    [Monitoring] ——> Drift detection + accuracy tracking

---
