"""Dataset preparation for HAN training."""
import os
import json
import glob
import csv
from typing import List, Tuple, Sequence, Optional, Dict
import torch
from torch.utils.data import Dataset

from vuln_hunter.utils.tokenizer import tokenize_source
from vuln_hunter.utils.bytecode_extractor import extract_bytecode_ops
from vuln_hunter.utils.preprocessing import build_parallel_segments, tokens_to_indices, normalize_segments
from vuln_hunter.models.han_model import build_vocab


VULN_KEYWORDS = {"eval", "exec", "pickle", "system"}


def label_code(code: str) -> int:
    return int(any(keyword in code for keyword in VULN_KEYWORDS))


def build_examples(files: List[str], segment_len: int = 100):
    examples = []
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        src_tokens = tokenize_source(code)
        bc_tokens = extract_bytecode_ops(code)
        src_segments, bc_segments = build_parallel_segments(src_tokens, bc_tokens, segment_len)
        examples.append({"src": src_segments, "bc": bc_segments, "label": label_code(code)})
    return examples


def build_vocabs(examples) -> Tuple[dict, dict]:
    src_flat = [tok for ex in examples for seg in ex["src"] for tok in seg]
    bc_flat = [tok for ex in examples for seg in ex["bc"] for tok in seg]
    return build_vocab([src_flat]), build_vocab([bc_flat])


class VulnDataset(Dataset):
    def __init__(self, examples, src_vocab: dict, bc_vocab: dict, segment_len: int = 100, max_segments: int = 8):
        self.examples = examples
        self.src_vocab = src_vocab
        self.bc_vocab = bc_vocab
        self.segment_len = segment_len
        self.max_segments = max_segments

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        src_segments = normalize_segments(ex["src"], self.max_segments)
        bc_segments = normalize_segments(ex["bc"], self.max_segments)
        src = tokens_to_indices(src_segments, self.src_vocab, self.segment_len)
        bc = tokens_to_indices(bc_segments, self.bc_vocab, self.segment_len)
        label = torch.tensor(ex["label"], dtype=torch.float32)
        return torch.tensor(src, dtype=torch.long), torch.tensor(bc, dtype=torch.long), label


def load_codexglue_jsonl(data_dir: str, segment_len: int, splits: Sequence[str]):
    """Load CodeXGLUE defect-detection jsonl files if present."""
    jsonl_files = []
    for split in splits:
        patterns = [f"{split}.jsonl", f"{split.capitalize()}.jsonl", f"*{split}*.jsonl"]
        for pattern in patterns:
            jsonl_files.extend(glob.glob(os.path.join(data_dir, pattern)))
    examples = []
    for path in jsonl_files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                code = sample.get("func") or sample.get("code") or ""
                label = int(sample.get("target", 0))
                src_tokens = tokenize_source(code)
                bc_tokens = extract_bytecode_ops(code)
                src_segments, bc_segments = build_parallel_segments(src_tokens, bc_tokens, segment_len)
                examples.append({"src": src_segments, "bc": bc_segments, "label": label})
    return examples


def class_counts_from_examples(examples) -> Dict[str, int]:
    counts = {"total": len(examples), "pos": 0, "neg": 0}
    for ex in examples:
        if ex.get("label", 0):
            counts["pos"] += 1
        else:
            counts["neg"] += 1
    return counts


def load_hf_codexglue(splits: Sequence[str], segment_len: int, dataset_name: str = "maddyrucos/code_vulnerability_python"):
    """Load Python vulnerability dataset from Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    split_map = {"valid": "validation", "validation": "validation", "train": "train", "test": "test"}
    examples = []
    hf_ds = load_dataset(dataset_name)
    for split in splits:
        hf_split = split_map.get(split, split)
        if hf_split not in hf_ds:
            continue
        for sample in hf_ds[hf_split]:
            code = sample.get("code") or sample.get("func") or ""
            label = int(sample.get("target", sample.get("label", 0)))
            src_tokens = tokenize_source(code)
            bc_tokens = extract_bytecode_ops(code)
            src_segments, bc_segments = build_parallel_segments(src_tokens, bc_tokens, segment_len)
            examples.append({"src": src_segments, "bc": bc_segments, "label": label})
    return examples


def load_kaggle_csv(data_dir: str, segment_len: int):
    """Load vulnerability_fix_dataset.csv with vulnerable/fixed code pairs."""
    csv_path = os.path.join(data_dir, "vulnerability_fix_dataset.csv")
    if not os.path.exists(csv_path):
        return []
    examples = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vuln_code = row.get("vulnerable_code", "")
            fixed_code = row.get("fixed_code", "")
            if vuln_code:
                src_tokens = tokenize_source(vuln_code)
                bc_tokens = extract_bytecode_ops(vuln_code)
                src_segments, bc_segments = build_parallel_segments(src_tokens, bc_tokens, segment_len)
                examples.append({"src": src_segments, "bc": bc_segments, "label": 1})
            if fixed_code:
                src_tokens = tokenize_source(fixed_code)
                bc_tokens = extract_bytecode_ops(fixed_code)
                src_segments, bc_segments = build_parallel_segments(src_tokens, bc_tokens, segment_len)
                examples.append({"src": src_segments, "bc": bc_segments, "label": 0})
    return examples


def load_txt_split(data_dir: str, split: str, segment_len: int):
    path = os.path.join(data_dir, f"{split}.txt")
    if not os.path.exists(path):
        return []
    examples = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                label_str, code = line.split("\t", 1)
                label = int(label_str)
            except ValueError:
                continue
            src_tokens = tokenize_source(code)
            bc_tokens = extract_bytecode_ops(code)
            src_segments, bc_segments = build_parallel_segments(src_tokens, bc_tokens, segment_len)
            examples.append({"src": src_segments, "bc": bc_segments, "label": label})
    return examples


def default_dataset(
    data_dir: str,
    segment_len: int = 100,
    splits: Optional[Sequence[str]] = None,
    max_segments: int = 8,
    src_vocab_override: Optional[dict] = None,
    bc_vocab_override: Optional[dict] = None,
):
    # Prefer Hugging Face CodeXGLUE; fallback to local jsonl/txt; lastly .py files.
    splits = splits or ("train",)
    examples: List[Dict] = []

    # Primary: Kaggle CSV if present
    examples = load_kaggle_csv(data_dir, segment_len)

    # Next: Hugging Face hosted dataset (requires network/cache)
    if not examples:
        examples = load_hf_codexglue(splits, segment_len)

    # Fallback: local jsonl
    if not examples:
        examples = load_codexglue_jsonl(data_dir, segment_len, splits)

    # Fallback: local txt (legacy CodeXGLUE format)
    if not examples:
        for split in splits:
            examples.extend(load_txt_split(data_dir, split, segment_len))

    # Fallback: any python files in data_dir
    if not examples:
        files = []
        for root, _, names in os.walk(data_dir):
            for name in names:
                if name.endswith(".py"):
                    files.append(os.path.join(root, name))
        if not files:
            sample_path = os.path.join(data_dir, "toy.py")
            os.makedirs(os.path.dirname(sample_path), exist_ok=True)
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write("import os\n\ndef run(cmd):\n    return os.system(cmd)\n")
            files.append(sample_path)
        examples = build_examples(files, segment_len)

    src_vocab = src_vocab_override or None
    bc_vocab = bc_vocab_override or None
    if src_vocab is None or bc_vocab is None:
        src_vocab, bc_vocab = build_vocabs(examples)
    dataset = VulnDataset(examples, src_vocab, bc_vocab, segment_len, max_segments=max_segments)
    stats = class_counts_from_examples(examples)
    return dataset, src_vocab, bc_vocab, stats
