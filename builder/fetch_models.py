from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel

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

MODEL_CACHE_PATH_TEMPLATE = "/runpod/cache/{model}/{revision}"


def resolve_model_cache_path(repositories: list[str]) -> list[str]:
    return [
        MODEL_CACHE_PATH_TEMPLATE.format(
            # the model is always the first element
            model=repository_and_revision[0],
            # the revision is the second element if it exists
            revision=repository_and_revision[1]
            if len(repository_and_revision) > 1
            else "main",
        )
        # the repository is split into the model and revision by the last colon
        for repository_and_revision in (
            repository.rsplit(":", 1)
            for repository in (
                *(os.environ.get("RUNPOD_HUGGINGFACE_MODEL", "").split(",")),
                *deprecated_model_names.split(";"),
            )
        )
    ]


def load_model(selected_model):
    """
    Load and cache models in parallel
    """
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
