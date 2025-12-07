import argparse
from vuln_hunter.utils.file_scanner import iter_python_files
from vuln_hunter.inference.predictor import Predictor


def main():
    parser = argparse.ArgumentParser(description="Scan a repository for potential vulnerabilities")
    parser.add_argument("path", help="Path to repository")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for positive classification")
    parser.add_argument("--explain", action="store_true", help="Show top-attended source segment for context")
    args = parser.parse_args()

    predictor = Predictor()
    for file_path in iter_python_files(args.path):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        result = predictor.predict(code, threshold=args.threshold)
        status = "vulnerable" if result["vulnerable"] else "clean"
        line = f"{file_path}: {status} (p={result['probability']:.2f})"
        print(line)
        if args.explain and result.get("explanation"):
            src_seg = result["explanation"].get("source_segment", [])
            if src_seg:
                snippet = " ".join(src_seg[:30])
                print(f"  top-segment: {snippet}")


if __name__ == "__main__":
    main()
