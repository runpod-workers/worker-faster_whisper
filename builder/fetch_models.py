"""Download Whisper model weights into the image at build time.

Set WHISPER_MODELS (space-separated) to bake only some models, for example
WHISPER_MODELS="turbo" for a CI image. Unset or empty bakes all of them.
"""

import os

ALL_MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "distil-large-v2",
    "distil-large-v3",
    "turbo",
]


def selected_models(value):
    if not value:
        return list(ALL_MODELS)
    models = value.split()
    unknown = [name for name in models if name not in ALL_MODELS]
    if unknown:
        raise ValueError(f"unknown WHISPER_MODELS: {', '.join(unknown)}")
    return models


def download_model_weights(selected_model):
    """
    Download model weights.
    """
    from faster_whisper.utils import download_model

    print(f"Downloading {selected_model}...")
    download_model(selected_model, cache_dir=None)
    print(f"Finished downloading {selected_model}.")


if __name__ == "__main__":
    for model_name in selected_models(os.environ.get("WHISPER_MODELS")):
        download_model_weights(model_name)

    print("Finished downloading models.")
