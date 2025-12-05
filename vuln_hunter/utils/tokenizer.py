"""Tokenizer for Python-like source code."""
import io
import re
import tokenize
from typing import List

SKIP_KINDS = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.NEWLINE}


def _fallback_tokens(code: str) -> List[str]:
    # Simple alphanumeric splitter for non-Python inputs
    return re.findall(r"\w+", code)


def tokenize_source(code: str) -> List[str]:
    """Return a list of token strings, excluding comments and non-semantic tokens.

    Falls back to a regex-based splitter if the code is not valid Python.
    """
    tokens: List[str] = []
    buffer = io.BytesIO(code.encode("utf-8"))
    try:
        for tok in tokenize.tokenize(buffer.readline):
            if tok.type in SKIP_KINDS:
                continue
            if tok.string.strip() == "":
                continue
            tokens.append(tok.string)
    except (tokenize.TokenError, SyntaxError):
        tokens = _fallback_tokens(code)
    return tokens
