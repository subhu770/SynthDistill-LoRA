# SynthDistill-LoRA Makefile
.PHONY: help install test lint run benchmark docker-build docker-up docker-down clean

help:
	@echo "SynthDistill-LoRA Build & Execution Commands:"
	@echo "  make install      - Install local dependencies"
	@echo "  make test         - Run full pytest test suite"
	@echo "  make benchmark    - Run 1,000-sample end-to-end distillation benchmark"
	@echo "  make run          - Start FastAPI orchestration server"
	@echo "  make docker-build - Build Docker container image"
	@echo "  make docker-up    - Run Docker Compose stack"
	@echo "  make docker-down  - Stop Docker Compose stack"
	@echo "  make clean        - Remove caches, temporary files, and test checkpoints"

install:
	pip install -e ".[dev]"

test:
	pytest -v

benchmark:
	python benchmarks/eval_distillation.py

run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf data/*.jsonl checkpoints/*
