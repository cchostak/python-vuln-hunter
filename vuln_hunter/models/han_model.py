import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.context = nn.Parameter(torch.randn(hidden_size))

    def forward(self, rnn_outputs: torch.Tensor):
        # rnn_outputs: (batch, seq, hidden)
        scores = torch.tanh(rnn_outputs) @ self.context
        weights = torch.softmax(scores, dim=1)
        attended = (rnn_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return attended, weights


class HANStream(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, embedding_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
        self.word_gru = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.word_att = Attention(hidden_size * 2)
        self.sent_gru = nn.GRU(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.sent_att = Attention(hidden_size * 2)

    def forward(self, inputs: torch.Tensor):
        # inputs: (batch, segments, tokens)
        bsz, segs, toks = inputs.shape
        x = inputs.view(bsz * segs, toks)
        emb = self.embedding(x)
        word_out, _ = self.word_gru(emb)
        sent_vecs, word_weights = self.word_att(word_out)
        sent_vecs = sent_vecs.view(bsz, segs, -1)
        sent_out, _ = self.sent_gru(sent_vecs)
        doc_vec, sent_weights = self.sent_att(sent_out)
        return doc_vec, word_weights.view(bsz, segs, toks), sent_weights


class HANModel(nn.Module):
    def __init__(self, src_vocab_size: int, bc_vocab_size: int, embedding_dim: int = 50, hidden_size: int = 32,
                 src_embedding_matrix=None, bc_embedding_matrix=None):
        super().__init__()
        self.src_stream = HANStream(src_vocab_size, embedding_dim, hidden_size, src_embedding_matrix)
        self.bc_stream = HANStream(bc_vocab_size, embedding_dim, hidden_size, bc_embedding_matrix)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, src_inputs: torch.Tensor, bc_inputs: torch.Tensor):
        src_vec, src_word_att, src_sent_att = self.src_stream(src_inputs)
        bc_vec, bc_word_att, bc_sent_att = self.bc_stream(bc_inputs)
        fused = torch.cat([src_vec, bc_vec], dim=1)
        logits = self.classifier(fused).squeeze(-1)
        return logits, {"src_word": src_word_att, "src_sent": src_sent_att, "bc_word": bc_word_att, "bc_sent": bc_sent_att}


def build_vocab(token_sequences, min_freq: int = 1):
    from collections import Counter

    counter = Counter()
    for seq in token_sequences:
        counter.update(seq)
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def build_embedding_matrix(vocab, fasttext_path: str, dim: int = 50):
    """Attempt to load FastText vectors, fallback to random."""
    matrix = np.random.uniform(-0.05, 0.05, size=(len(vocab), dim)).astype("float32")
    try:
        import fasttext

        model = fasttext.load_model(fasttext_path)
        for token, idx in vocab.items():
            matrix[idx] = model.get_word_vector(token)[:dim]
    except Exception:
        # Random init is fine for minimal scaffolding
        pass
    return matrix
