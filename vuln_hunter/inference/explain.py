"""Explain predictions from the HAN model."""
from typing import Dict
from vuln_hunter.inference.predictor import Predictor


def explain_code(code: str) -> Dict:
    predictor = Predictor()
    return predictor.predict(code)
