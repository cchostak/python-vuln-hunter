"""Safe HTTP example using requests with timeouts."""
import requests


def fetch_json(url: str):
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    return resp.json()

if __name__ == "__main__":
    try:
        print(fetch_json("https://httpbin.org/get"))
    except Exception as exc:
        print(f"Request failed: {exc}")
