from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vuln_hunter.inference.predictor import Predictor

app = FastAPI(title="python-vuln-hunter")
predictor = Predictor()


class ScanRequest(BaseModel):
    code: str
    threshold: float | None = None


@app.post("/scan")
def scan(req: ScanRequest):
    if not req.code:
        raise HTTPException(status_code=400, detail="code is required")
    threshold = req.threshold if req.threshold is not None else 0.5
    result = predictor.predict(req.code, threshold=threshold)
    return result
