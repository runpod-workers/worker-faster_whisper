from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel

model_names = ["tiny"] #, "base", "small", "medium", "large-v1", "large-v2", "large-v3"]


def load_model(selected_model):
    '''
    Load and cache models in parallel
    '''
    for _attempt in range(5):
        while True:
            try:
                loaded_model = WhisperModel(
                    selected_model, device="cpu", compute_type="int8")
            except (AttributeError, OSError):
                continue

            break

    return selected_model, loaded_model


models = {}

with ThreadPoolExecutor() as executor:
    for model_name, model in executor.map(load_model, model_names):
        if model_name is not None:
            models[model_name] = model
