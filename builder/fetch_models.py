from faster_whisper.utils import download_model

# Only download large-v2 model to save storage and ensure it's cached
model_name = "large-v2"

def download_model_weights(selected_model):
    """
    Download model weights.
    """
    print(f"Downloading {selected_model}...")
    download_model(selected_model, cache_dir=None)
    print(f"Finished downloading {selected_model}.")

# Download only large-v2
download_model_weights(model_name)
print(f"Finished downloading {model_name}.")
