PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
DATA_DIR ?= data/raw/rules_dataset
UVICORN ?= uvicorn vuln_hunter.inference.api:app --host 0.0.0.0 --port 8000
SPLIT ?= test
THRESHOLD ?= 0.78
SAFE_DIR ?= safe_examples

.PHONY: install venv test train evaluate api scan scan-live-repo download-data download-kaggle download-rules thresholds docker-build docker-up docker-down clean

install:
	$(PIP) install -r requirements.txt

venv:
	$(PYTHON) -m venv .venv

test:
	$(PYTHON) -m pytest

train:
	$(PYTHON) -m vuln_hunter.training.train_han --data-dir $(DATA_DIR) --pos-weight 1.0 --epochs 10 --splits train,validation --batch-size 64

evaluate:
	$(PYTHON) -m vuln_hunter.training.evaluate --data-dir $(DATA_DIR) --split $(SPLIT) --threshold $(THRESHOLD)

api:
	$(UVICORN)

scan:
	$(PYTHON) -m vuln_hunter.cli.scan $(path)

scan-live-repo:
	$(PYTHON) -m vuln_hunter.cli.scan /home/cchostak/Desktop/vulpy --threshold $(THRESHOLD)

download-data:
	$(PYTHON) scripts/download_data.py --target $(DATA_DIR)

download-kaggle:
	scripts/download_kaggle.sh $(DATA_DIR)

download-rules:
	$(PYTHON) scripts/build_rule_dataset.py --output data/raw/rules_dataset $(if $(SAFE_DIR),--safe-dir $(SAFE_DIR))

thresholds:
	$(PYTHON) scripts/threshold_search.py --data-dir $(DATA_DIR) --split $(SPLIT) --steps 101

docker-build:
	docker-compose build

docker-up:
	docker-compose up

docker-down:
	docker-compose down

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache
