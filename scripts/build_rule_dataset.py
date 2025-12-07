#!/usr/bin/env python3
"""Download Bandit examples and Semgrep python rules to build a small labeled jsonl dataset."""
import argparse
import json
import os
import random
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

BAD_HINTS = ("bad", "insecure", "vuln", "weak", "unsafe", "bandit-main/examples", "semgrep-rules-develop", "_bad")
GOOD_HINTS = ("ok", "good", "safe", "secure", "good", "_good")


def download_and_extract(url: str, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "archive.tgz"
        urlretrieve(url, tmp_path)
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(dest)


def collect_py_files(root: Path):
    for path in root.rglob("*.py"):
        if path.is_file():
            yield path


def label_from_path(path: Path) -> int:
    lower = str(path).lower()
    if any(h in lower for h in GOOD_HINTS):
        return 0
    if any(h in lower for h in BAD_HINTS):
        return 1
    # default to positive for these curated sources
    return 1


def build_examples(source_roots, labeler=label_from_path):
    examples = []
    for root in source_roots:
        for py in collect_py_files(root):
            try:
                code = py.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            label = labeler(py)
            examples.append({"code": code, "label": label, "path": str(py)})
    return examples


def split_and_write(examples, out_dir: Path):
    random.shuffle(examples)
    n = len(examples)
    train_end = int(0.8 * n)
    valid_end = int(0.9 * n)
    splits = {
        "train": examples[:train_end],
        "validation": examples[train_end:valid_end],
        "test": examples[valid_end:],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        out_path = out_dir / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"wrote {out_path} ({len(rows)} examples)")


def main():
    parser = argparse.ArgumentParser(description="Build rule-based vuln dataset from Bandit and Semgrep sources")
    parser.add_argument("--output", default="data/raw/rules_dataset", help="Output directory for jsonl splits")
    parser.add_argument("--safe-dir", default=None, help="Optional directory of known-safe Python files to add as negatives")
    args = parser.parse_args()
    default_safe = Path(__file__).resolve().parent.parent / "safe_examples"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        bandit_dest = tmp_root / "bandit"
        semgrep_dest = tmp_root / "semgrep"

        print("Downloading Bandit examples...")
        download_and_extract(
            "https://github.com/PyCQA/bandit/archive/refs/heads/main.tar.gz", bandit_dest
        )
        print("Downloading Semgrep rules...")
        download_and_extract(
            "https://github.com/semgrep/semgrep-rules/archive/refs/heads/develop.tar.gz", semgrep_dest
        )

        bandit_examples_root = next(bandit_dest.glob("bandit-main/examples"), None)
        semgrep_py_root = next(semgrep_dest.glob("semgrep-rules-develop/python"), None)
        roots = [p for p in (bandit_examples_root, semgrep_py_root) if p and p.exists()]
        if not roots:
            raise SystemExit("Could not locate downloaded sources; layout may have changed.")

        examples = build_examples(roots)

        # Append additional safe code as negatives if provided
        safe_root = Path(args.safe_dir) if args.safe_dir else default_safe
        if safe_root and safe_root.exists():
            for py in collect_py_files(safe_root):
                try:
                    code = py.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                examples.append({"code": code, "label": 0, "path": str(py)})

        if not examples:
            raise SystemExit("No examples collected; check download and paths.")
        split_and_write(examples, Path(args.output))


if __name__ == "__main__":
    main()
