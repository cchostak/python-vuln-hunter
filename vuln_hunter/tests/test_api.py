from fastapi.testclient import TestClient
from vuln_hunter.inference.api import app


def test_scan_endpoint_returns_json():
    client = TestClient(app)
    resp = client.post("/scan", json={"code": "print('hi')"})
    assert resp.status_code == 200
    data = resp.json()
    assert "probability" in data and "vulnerable" in data
