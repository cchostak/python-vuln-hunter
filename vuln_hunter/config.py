import os

DATA_DIR = os.environ.get("VULN_HUNTER_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))
FASTTEXT_SOURCE_PATH = os.environ.get(
    "VULN_HUNTER_FASTTEXT_SOURCE", os.path.join(os.path.dirname(__file__), "models", "fasttext_source.bin")
)
FASTTEXT_BYTECODE_PATH = os.environ.get(
    "VULN_HUNTER_FASTTEXT_BYTECODE", os.path.join(os.path.dirname(__file__), "models", "fasttext_bytecode.bin")
)
MODEL_CHECKPOINT = os.environ.get(
    "VULN_HUNTER_CHECKPOINT", os.path.join(os.path.dirname(__file__), "models", "han_model.pt")
)
DEFAULT_SEQ_LEN = 100
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 1
