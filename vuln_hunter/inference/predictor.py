import os
import torch
from typing import Dict
import numpy as np

from vuln_hunter.utils.tokenizer import tokenize_source
from vuln_hunter.utils.bytecode_extractor import extract_bytecode_ops
from vuln_hunter.utils.preprocessing import build_parallel_segments, tokens_to_indices
from vuln_hunter.models.han_model import HANModel, build_vocab
from vuln_hunter.config import MODEL_CHECKPOINT, DEFAULT_SEQ_LEN


class Predictor:
    def __init__(self, checkpoint: str = MODEL_CHECKPOINT):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if os.path.exists(checkpoint):
            torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
            payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
            self.src_vocab = payload["src_vocab"]
            self.bc_vocab = payload["bc_vocab"]
            self.model = HANModel(len(self.src_vocab), len(self.bc_vocab))
            self.model.load_state_dict(payload["model_state"])
        else:
            # Minimal fallback
            self.src_vocab = {"<pad>": 0, "<unk>": 1}
            self.bc_vocab = {"<pad>": 0, "<unk>": 1}
            self.model = HANModel(len(self.src_vocab), len(self.bc_vocab))
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, code: str):
        src_tokens = tokenize_source(code)
        bc_tokens = extract_bytecode_ops(code)
        src_segments, bc_segments = build_parallel_segments(src_tokens, bc_tokens, DEFAULT_SEQ_LEN)
        src_idx = tokens_to_indices(src_segments, self.src_vocab, DEFAULT_SEQ_LEN)
        bc_idx = tokens_to_indices(bc_segments, self.bc_vocab, DEFAULT_SEQ_LEN)
        src_tensor = torch.tensor(src_idx, dtype=torch.long).unsqueeze(0)
        bc_tensor = torch.tensor(bc_idx, dtype=torch.long).unsqueeze(0)
        return src_tensor.to(self.device), bc_tensor.to(self.device), src_segments, bc_segments

    def predict(self, code: str, threshold: float = 0.5) -> Dict:
        src_tensor, bc_tensor, src_segments, bc_segments = self.preprocess(code)
        with torch.no_grad():
            logits, attn = self.model(src_tensor, bc_tensor)
            prob = torch.sigmoid(logits)[0].item()
        top_src = int(torch.argmax(attn["src_sent"], dim=1)[0].item()) if attn.get("src_sent") is not None else 0
        top_bc = int(torch.argmax(attn["bc_sent"], dim=1)[0].item()) if attn.get("bc_sent") is not None else 0
        explanation = {
            "source_segment": src_segments[top_src] if src_segments else [],
            "bytecode_segment": bc_segments[top_bc] if bc_segments else [],
        }
        return {"probability": prob, "vulnerable": prob > threshold, "explanation": explanation, "threshold": threshold}
