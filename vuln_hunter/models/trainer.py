import torch
from torch.utils.data import DataLoader, random_split
from torch import nn

from vuln_hunter.models.han_model import HANModel, build_embedding_matrix
from vuln_hunter.config import MODEL_CHECKPOINT, DEFAULT_SEQ_LEN


def train(model: HANModel, dataset, epochs: int = 1, batch_size: int = 4, lr: float = 1e-3, device: str = "cpu", pos_weight: float | None = None):
    model.to(device)
    weight = torch.tensor([pos_weight], device=device) if pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_size = max(1, int(0.2 * len(dataset))) if len(dataset) > 1 else 0
    if val_size > 0:
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        val_loader = DataLoader(val_ds, batch_size=batch_size)
    else:
        train_ds = dataset
        val_loader = None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for src, bc, labels in train_loader:
            src, bc, labels = src.to(device), bc.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(src, bc)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}/{epochs} loss={avg_loss:.4f}")

        if val_loader:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for src, bc, labels in val_loader:
                    src, bc, labels = src.to(device), bc.to(device), labels.to(device)
                    logits, _ = model(src, bc)
                    preds = torch.sigmoid(logits) > 0.5
                    correct += (preds.float() == labels).sum().item()
                    total += labels.numel()
            acc = correct / max(1, total)
            print(f"Validation accuracy: {acc:.2f}")


def save_checkpoint(model: HANModel, src_vocab: dict, bc_vocab: dict, src_emb_matrix, bc_emb_matrix, path: str = MODEL_CHECKPOINT):
    payload = {
        "model_state": model.state_dict(),
        "src_vocab": src_vocab,
        "bc_vocab": bc_vocab,
        "src_emb": src_emb_matrix,
        "bc_emb": bc_emb_matrix,
    }
    torch.save(payload, path)


def initialize_model(src_vocab: dict, bc_vocab: dict, embedding_dim: int = 50, hidden_size: int = 32):
    src_emb = build_embedding_matrix(src_vocab, MODEL_CHECKPOINT.replace("han_model.pt", "fasttext_source.bin"), dim=embedding_dim)
    bc_emb = build_embedding_matrix(bc_vocab, MODEL_CHECKPOINT.replace("han_model.pt", "fasttext_bytecode.bin"), dim=embedding_dim)
    model = HANModel(len(src_vocab), len(bc_vocab), embedding_dim=embedding_dim, hidden_size=hidden_size,
                     src_embedding_matrix=src_emb, bc_embedding_matrix=bc_emb)
    return model, src_emb, bc_emb
