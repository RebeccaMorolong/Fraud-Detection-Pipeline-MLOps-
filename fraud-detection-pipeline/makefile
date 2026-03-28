.PHONY: install data train evaluate serve test lint format docker-build clean

# ============================================================
# Setup
# ============================================================
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt ruff black mypy pre-commit
	pre-commit install

# ============================================================
# Data & Training Pipeline
# ============================================================
data:
	python scripts/generate_data.py

train:
	python scripts/train_model.py

evaluate:
	python scripts/evaluate_model.py

# Run full pipeline: data -> train -> evaluate
pipeline: data train evaluate
	@echo "Pipeline complete."

# ============================================================
# API Server
# ============================================================
serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2

# ============================================================
# MLflow UI
# ============================================================
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# ============================================================
# Testing
# ============================================================
test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --cov=src --cov-report=term-missing

test-integration:
	pytest tests/integration/ -v --tb=short

# ============================================================
# Code Quality
# ============================================================
lint:
	ruff check src/ tests/ scripts/

format:
	black src/ tests/ scripts/
	ruff check src/ tests/ scripts/ --fix

# ============================================================
# Docker
# ============================================================
docker-build:
	docker build -t fraud-detection-api:latest .

docker-run:
	docker run -p 8000:8000 \
	  -v $(PWD)/models:/app/models \
	  -v $(PWD)/monitoring:/app/monitoring \
	  fraud-detection-api:latest

# ============================================================
# Demo: send test transactions
# ============================================================
demo-normal:
	curl -s -X POST http://localhost:8000/score \
	  -H "Content-Type: application/json" \
	  -d '{"amount":25,"is_online":1,"hour_of_day":14,"day_of_week":2,"is_weekend":0,"is_night":0,"is_high_risk_merchant":0,"is_medium_risk_merchant":0,"amount_z_score":0.3,"amount_vs_user_max":0.4,"days_since_last_txn":1.2,"txn_count_1h":0,"txn_count_6h":1,"txn_count_24h":2,"total_amount_1h":0,"total_amount_24h":45,"is_new_country":0,"is_new_merchant_category":0}' \
	  | python -m json.tool

demo-fraud:
	curl -s -X POST http://localhost:8000/score \
	  -H "Content-Type: application/json" \
	  -d '{"amount":5000,"is_online":1,"hour_of_day":2,"day_of_week":6,"is_weekend":1,"is_night":1,"is_high_risk_merchant":1,"is_medium_risk_merchant":0,"amount_z_score":8.5,"amount_vs_user_max":4.2,"days_since_last_txn":0.1,"txn_count_1h":15,"txn_count_6h":20,"txn_count_24h":25,"total_amount_1h":8500,"total_amount_24h":12000,"is_new_country":1,"is_new_merchant_category":1}' \
	  | python -m json.tool

# ============================================================
# Cleanup
# ============================================================
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf monitoring/predictions.db

clean-all: clean
	rm -rf data/raw/* data/processed/* models/saved/* mlruns/