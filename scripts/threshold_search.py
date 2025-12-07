#!/usr/bin/env python3
"""Grid search thresholds to maximize F1 on a given split."""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vuln_hunter.training.evaluate import load_model
from vuln_hunter.training.dataset_builder import default_dataset
from vuln_hunter.config import DEFAULT_SEQ_LEN


def evaluate_thresholds(logits, labels, thresholds):
    best = None
    for thr in thresholds:
        preds = (logits > thr).astype(np.int64)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        acc = (tp + tn) / max(1, tp + tn + fp + fn)
        result = {
            "threshold": thr,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "acc": acc,
        }
        if best is None or result["f1"] > best["f1"]:
            best = result
        yield result
    return best


def main():
    parser = argparse.ArgumentParser(description="Search for best decision threshold")
    parser.add_argument("--data-dir", default="data/raw/rules_dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (defaults to config)")
    parser.add_argument("--steps", type=int, default=21, help="Number of thresholds between 0 and 1 inclusive")
    args = parser.parse_args()

    model, payload = load_model(args.checkpoint) if args.checkpoint else load_model()
    dataset, _, _, stats = default_dataset(
        args.data_dir,
        DEFAULT_SEQ_LEN,
        splits=[args.split],
        src_vocab_override=payload.get("src_vocab"),
        bc_vocab_override=payload.get("bc_vocab"),
    )
    loader = DataLoader(dataset, batch_size=64)
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for src, bc, labels in loader:
            logits, _ = model(src, bc)
            all_logits.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)

    thresholds = np.linspace(0, 1, num=args.steps)
    best = None
    for res in evaluate_thresholds(logits_np, labels_np, thresholds):
        print(
            f"thr={res['threshold']:.2f} tp={res['tp']} fp={res['fp']} tn={res['tn']} fn={res['fn']} "
            f"prec={res['precision']:.2f} rec={res['recall']:.2f} f1={res['f1']:.2f} acc={res['acc']:.2f}"
        )
        if best is None or res["f1"] > best["f1"]:
            best = res

    print("\nBest threshold by F1:")
    print(best)
    print("Dataset stats:", stats)


if __name__ == "__main__":
    main()
