from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
from runpod import RunPodLogger
from urllib.parse import urlparse

logger = RunPodLogger()
model_names = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large-v1",
    "openai/whisper-large-v2",
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
]

MODEL_CACHE_PATH_TEMPLATE = "/runpod/cache/model/{path}"


def topath(raw: str) -> str:
    raw = raw.strip()
    if ":" in raw:
        model, branch = raw.rsplit(":", maxsplit=1)
    else:
        model, branch = raw, "main"
    if "/" not in model:
        raise ValueError(
            f"invalid model: expected one in the form user/model[:path], but got {model}"
        )
    user, model = model.rsplit("/", maxsplit=1)
    return MODEL_CACHE_PATH_TEMPLATE.format(
        path="/".join(c.strip("/") for c in (user, model, branch))
    )


def modelpaths() -> list[str]:
    raw = os.environ.get("RUNPOD_HUGGINGFACE_MODEL")
    if not raw:
        return []
    return [topath(m) for m in raw.split(",")]


def load_model(selected_model):
    """
    Load and cache models in parallel
    """
    selected_model = modelpaths()
    if len(selected_model) == 0:
        logger.error(
            "You must provide a model in the RUNPOD_HUGGINGFACE_MODEL environment variable"
        )
        return None, None
    elif len(selected_model) > 1:
        logger.error(
            "Whisper only supports a single model at a time, but multiple were provided in the RUNPOD_HUGGINGFACE_MODEL environment variable"
        )
        return None, None
    selected_model = selected_model[0]
    # TODO: this seems like a hack?
    for _attempt in range(5):
        while True:
            try:
                loaded_model = WhisperModel(
                    selected_model, device="cpu", compute_type="int8"
                )
            except (AttributeError, OSError):
                # FIXME: should we be swallowing these errors?
                continue

            break

    return selected_model, loaded_model


models = {}

with ThreadPoolExecutor() as executor:
    for model_name, model in executor.map(load_model, model_names):
        if model_name is not None:
            models[model_name] = model
