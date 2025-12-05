"""Utility preprocessing helpers for building datasets."""
from typing import List, Tuple
import numpy as np


def segment_tokens(tokens: List[str], segment_len: int = 100) -> List[List[str]]:
    """Split tokens into near-equal chunks of length segment_len."""
    segments = []
    for i in range(0, len(tokens), segment_len):
        segments.append(tokens[i : i + segment_len])
    if not segments:
        segments = [[]]
    return segments


def pad_segment(segment: List[str], length: int) -> List[str]:
    if len(segment) >= length:
        return segment[:length]
    return segment + ["<pad>"] * (length - len(segment))


def tokens_to_indices(segments: List[List[str]], vocab: dict, pad_length: int) -> np.ndarray:
    """Convert segments of tokens to indexed numpy array with padding."""
    encoded = []
    for seg in segments:
        padded = pad_segment(seg, pad_length)
        encoded.append([vocab.get(tok, vocab.get("<unk>", 1)) for tok in padded])
    return np.array(encoded, dtype="int64")


def build_parallel_segments(source_tokens: List[str], bytecode_tokens: List[str], segment_len: int = 100) -> Tuple[List[List[str]], List[List[str]]]:
    """Create aligned segments for source and bytecode streams."""
    src_segments = segment_tokens(source_tokens, segment_len)
    byte_segments = segment_tokens(bytecode_tokens, segment_len)
    max_len = max(len(src_segments), len(byte_segments))
    while len(src_segments) < max_len:
        src_segments.append(["<pad>"])
    while len(byte_segments) < max_len:
        byte_segments.append(["<pad>"])
    return src_segments, byte_segments


def normalize_segments(segments: List[List[str]], max_segments: int) -> List[List[str]]:
    """Pad or truncate list of segments to fixed length for batching."""
    if len(segments) > max_segments:
        return segments[:max_segments]
    if len(segments) < max_segments:
        segments = segments + [["<pad>"]] * (max_segments - len(segments))
    return segments
