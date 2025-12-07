import argparse
import torch

from vuln_hunter.training.dataset_builder import default_dataset
from vuln_hunter.models.trainer import train, save_checkpoint, initialize_model
from vuln_hunter.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_CHECKPOINT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--splits",
        default="train",
        help="Comma-separated CodeXGLUE splits to load (e.g. train,valid). Defaults to train only.",
    )
    parser.add_argument("--pos-weight", type=float, default=None, help="Positive class weight for imbalanced data")
    parser.add_argument("--max-segments", type=int, default=8, help="Number of code segments to pad/truncate to")
    parser.add_argument("--embedding-dim", type=int, default=50, help="Embedding dimension")
    parser.add_argument("--hidden-size", type=int, default=32, help="GRU hidden size")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout for classifier head")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    dataset, src_vocab, bc_vocab, stats = default_dataset(args.data_dir, splits=splits, max_segments=args.max_segments)
    pos_weight = args.pos_weight
    if pos_weight is None:
        pos = max(1, stats.get("pos", 0))
        neg = max(1, stats.get("neg", 0))
        pos_weight = neg / pos
    print(f"Class balance: pos={stats.get('pos',0)} neg={stats.get('neg',0)} pos_weight={pos_weight:.2f}")
    model, src_emb, bc_emb = initialize_model(src_vocab, bc_vocab, embedding_dim=args.embedding_dim, hidden_size=args.hidden_size, dropout=args.dropout)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(model, dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device, pos_weight=pos_weight)
    save_checkpoint(model, src_vocab, bc_vocab, src_emb, bc_emb, MODEL_CHECKPOINT)
    print(f"Saved checkpoint to {MODEL_CHECKPOINT}")


if __name__ == "__main__":
    main()
