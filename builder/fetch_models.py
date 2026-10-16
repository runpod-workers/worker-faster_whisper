from faster_whisper.utils import download_model

from model_config import MODEL_DIR, MODEL_REPO


def download_model_weights():
    """
    Download the kotoba-whisper CTranslate2 weights into the image.
    """
    print(f"Downloading {MODEL_REPO} to {MODEL_DIR}...")
    download_model(MODEL_REPO, output_dir=MODEL_DIR)
    print(f"Finished downloading {MODEL_REPO}.")


if __name__ == "__main__":
    download_model_weights()
