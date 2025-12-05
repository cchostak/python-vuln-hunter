import argparse
import torch
from torch.utils.data import DataLoader

from vuln_hunter.training.dataset_builder import default_dataset
from vuln_hunter.models.han_model import HANModel
from vuln_hunter.config import MODEL_CHECKPOINT, DEFAULT_SEQ_LEN


def load_model(checkpoint_path: str = MODEL_CHECKPOINT):
    # Allow numpy objects saved in older checkpoints
    import numpy as np

    torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = HANModel(len(payload["src_vocab"]), len(payload["bc_vocab"]))
    model.load_state_dict(payload["model_state"])
    return model, payload


def evaluate(data_dir: str, checkpoint: str = MODEL_CHECKPOINT, splits=None, threshold: float = 0.5):
    model, payload = load_model(checkpoint)
    dataset, _, _, stats = default_dataset(
        data_dir,
        DEFAULT_SEQ_LEN,
        splits=splits,
        src_vocab_override=payload.get("src_vocab"),
        bc_vocab_override=payload.get("bc_vocab"),
    )
    loader = DataLoader(dataset, batch_size=32)
    model.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for src, bc, labels in loader:
            logits, _ = model(src, bc)
            probs = torch.sigmoid(logits)
            preds = probs > threshold
            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    total = tp + tn + fp + fn
    acc = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    print(f"Dataset stats: {stats}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Confusion matrix: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Accuracy: {acc:.2f} Precision: {precision:.2f} Recall: {recall:.2f}")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "acc": acc, "precision": precision, "recall": recall}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--checkpoint", default=MODEL_CHECKPOINT)
    parser.add_argument("--split", default="test", help="CodeXGLUE split to evaluate (test/valid/train)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for positive")
    args = parser.parse_args()
    splits = [args.split] if args.split else None
    evaluate(args.data_dir, args.checkpoint, splits, threshold=args.threshold)
