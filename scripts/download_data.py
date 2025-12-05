"""Download Python vulnerability dataset from Hugging Face."""
import argparse
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit("datasets package not installed; run `pip install -r requirements.txt`") from exc


def download(target: Path, dataset: str = "maddyrucos/code_vulnerability_python"):
    ds = load_dataset(dataset)
    target.mkdir(parents=True, exist_ok=True)
    # Dataset may expose different split names; iterate over available ones.
    for split in ds.keys():
        if split not in ds:
            continue
        out = target / f"{split}.jsonl"
        ds[split].to_json(out, orient="records", lines=True)
        print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description="Download CodeXGLUE defect detection dataset")
    parser.add_argument("--target", default="data/raw/codexglue_defect", help="output directory for splits")
    parser.add_argument("--dataset", default="google/code_x_glue_cc_defect_detection", help="HF dataset name")
    args = parser.parse_args()
    download(Path(args.target), args.dataset)


if __name__ == "__main__":
    main()
