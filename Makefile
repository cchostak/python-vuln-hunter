PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
DATA_DIR ?= data/raw/codexglue_defect
UVICORN ?= uvicorn vuln_hunter.inference.api:app --host 0.0.0.0 --port 8000
SPLIT ?= test
THRESHOLD ?= 0.2

.PHONY: install venv test train evaluate api scan download-data docker-build docker-up docker-down clean

install:
	$(PIP) install -r requirements.txt

venv:
	$(PYTHON) -m venv .venv

test:
	$(PYTHON) -m pytest

train:
	$(PYTHON) -m vuln_hunter.training.train_han --data-dir $(DATA_DIR) --epochs 5 --splits train,validation --batch-size 256

evaluate:
	$(PYTHON) -m vuln_hunter.training.evaluate --data-dir $(DATA_DIR) --split $(SPLIT) --threshold $(THRESHOLD)

api:
	$(UVICORN)

scan:
	$(PYTHON) -m vuln_hunter.cli.scan $(path)

download-data:
	$(PYTHON) scripts/download_data.py --target $(DATA_DIR)

download-kaggle:
	scripts/download_kaggle.sh $(DATA_DIR)

docker-build:
	docker-compose build

docker-up:
	docker-compose up

docker-down:
	docker-compose down

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache
