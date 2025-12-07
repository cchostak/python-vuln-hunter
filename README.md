# python-vuln-hunter

Dual-modality vulnerability detector for Python using FastText embeddings and a hierarchical attention network (HAN). Supports training, inference, FastAPI serving, and a CLI scanner.

References:

- https://link.springer.com/article/10.1186/s42400-024-00332-7
- https://github.com/microsoft/CodeXGLUE/tree/main
- https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection

## Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training
Default training uses the locally built rule-based dataset (`data/raw/rules_dataset`), composed of Bandit examples, Semgrep Python rules, and bundled safe examples. If a Kaggle CSV (`vulnerability_fix_dataset.csv`) is present in the data dir, it will be used; Hugging Face PySecDB is only used when `VULN_HUNTER_USE_PYSEC` is set (it is gated).
```
make download-rules     # build rule dataset (add SAFE_DIR=/path/to/clean/python/code to add more negatives)
python -m vuln_hunter.training.train_han --data-dir data/raw/rules_dataset --epochs 3 --splits train,validation
```
Common targets:
```
make download-data      # fetch HF dataset (jsonl splits)
make download-rules     # build rule dataset (Bandit/Semgrep + safe examples)
make train              # train using DATA_DIR (defaults to rules_dataset)
make evaluate           # eval split/test with THRESHOLD (defaults: split=test, threshold=0.74)
make thresholds         # grid-search thresholds to maximize F1
```

### Optional: Kaggle vulnerability-fix dataset
Requires Kaggle API credentials (`KAGGLE_USERNAME`/`KAGGLE_KEY` set, or `~/.kaggle/kaggle.json`).
```
make download-kaggle
```
Data will be placed under `data/raw/codexglue_defect` by default; adjust `DATA_DIR` if desired.

### Optional: Bandit/Semgrep rule-based dataset
Downloads Bandit examples and Semgrep python rules, builds a simple labeled jsonl split under `data/raw/rules_dataset`.
```
make download-rules
```
You can add your own known-safe Python files to balance negatives:
```
make download-rules SAFE_DIR=/path/to/clean/python/code
```

### Training hyperparameters
`train_han.py` flags (also passable via `make train` overrides): `--batch-size`, `--epochs`, `--lr`, `--pos-weight`, `--max-segments`, `--embedding-dim`, `--hidden-size`, `--dropout`, `--splits`.

### Threshold selection
Use the search helper to pick a decision cutoff:
```
make thresholds DATA_DIR=data/raw/rules_dataset SPLIT=test
```
Scan with the chosen threshold and `--explain` to print the top-attended source segment.

## Running the API
```
uvicorn vuln_hunter.inference.api:app --host 0.0.0.0 --port 8000
```
or
```
make api
```

## CLI scanning
```
python -m vuln_hunter.cli.scan /path/to/repo
```
or
```
make scan path=/path/to/repo
make scan-live-repo THRESHOLD=0.8
```
Use `--explain` with the CLI to print the top-attended source segment for each file.

## Tests
```
pytest
```
or
```
make test
```

## Docker
Build and run the stack:
```
docker-compose up --build
```
- `train`: trains the HAN model
- `api`: runs FastAPI server on port 8000
- `worker`: runs the CLI scanner against `/workspace/target`
