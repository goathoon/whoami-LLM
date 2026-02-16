import os

# Default to offline mode for Hugging Face dependencies.
# Users can still override by setting these env vars explicitly.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
