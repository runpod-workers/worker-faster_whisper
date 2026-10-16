"""
Single source of truth for the model this worker serves.

Imported both at build time (builder/fetch_models.py) and at runtime
(src/predict.py), so the baked weights and the loaded weights can never drift.
"""

import os

MODEL_NAME = "kotoba-whisper-v2.0-faster"
MODEL_REPO = "kotoba-tech/kotoba-whisper-v2.0-faster"

# Where fetch_models.py bakes the weights during the image build.
MODEL_DIR = os.environ.get("MODEL_PATH", "/models/kotoba-whisper-v2.0-faster")

# kotoba-whisper is a Japanese-only distillation of large-v3; the language is
# never taken from the request.
LANGUAGE = "ja"

# Recommended by the model card for sequential long-form decoding.
DEFAULT_CHUNK_LENGTH = 15

ACCEPTED_MODEL_NAMES = {MODEL_NAME, MODEL_REPO}
