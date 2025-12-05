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
Training uses the Hugging Face-hosted Python vulnerability dataset (`maddyrucos/code_vulnerability_python`) by default; if a Kaggle CSV (`vulnerability_fix_dataset.csv`) is present in the data dir, it will be used instead.
```
make download-data      # prefetches the HF dataset
python -m vuln_hunter.training.train_han --data-dir data/raw/codexglue_defect --epochs 3 --splits train,validation
```
Most commands can be driven via `make`:
```
make download-data      # fetch HF dataset (jsonl splits)
make train              # train using HF dataset (defaults to splits train,validation)
make evaluate           # quick eval run (defaults to --split test; override with SPLIT=validation and THRESHOLD=0.3)
```

### Optional: Kaggle vulnerability-fix dataset
Requires Kaggle API credentials (`KAGGLE_USERNAME`/`KAGGLE_KEY` set, or `~/.kaggle/kaggle.json`).
```
make download-kaggle
```
Data will be placed under `data/raw/codexglue_defect` by default; adjust `DATA_DIR` if desired.

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
```

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
