# look for main() to see the actual entry point.
from typing import Any, Literal, Union
import base64
import faster_whisper
import numpy as np
import os
import requests
import runpod
import subprocess
import sys
import tempfile
import time


def cachedmodel2path(basedir="/runpod/cache/model") -> dict[str, str]:
    """look for directories in the form basedir/:user/:model/:revision
    and return a dictionary mapping user/model to path
    """

    def listsubdir(path: str) -> list[str]:
        """as listdir, but only returns directories"""
        return [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]

    """load cached models from basedir"""
    models: list[str] = [
        os.path.join(user, model)
        for user in listsubdir(basedir)
        for model in listsubdir(os.path.join(basedir, user))
    ]
    model2path = {}
    for model in models:
        for revision in listsubdir(os.path.join(basedir, model)):
            if len(revision) > 1:
                raise ValueError(
                    f"""load cached models: basedir={basedir} than one revision found: {revision}
    HELP: check your modelcache settings in your pod configuration"""
                )
            model2path[revision[0]] = os.path.join(basedir, revision[0])
    return model2path


def log(msg: str, **kwargs) -> None:
    print(f"runpod: {msg}", file=sys.stderr, **kwargs)


def format_transcript(
    format: Literal["plain_text", "vtt", "srt", "rich_text"],
    segments: list[faster_whisper.transcribe.Segment],
) -> str:
    """
    for 3a and 4a: format the the transcripts
    format the transcript in the specified format
    """

    if format == "plain_text":
        return " ".join(segment.text.lstrip() for segment in segments)
    elif format == "vtt":
        result = ""
        # TODO: this is mega inefficent; do a join rather than a zillion allocs
        for segment in segments:
            result += f"{faster_whisper.format_timestamp(segment.start)} --> {faster_whisper.format_timestamp(segment.end)}\n"
            result += f"{segment.text.strip().replace('-->', '->')}\n"
            result += "\n"
    elif format == "srt":
        result = ""
        # TODO: this is mega inefficent; do a join rather than a zillion allocs
        for i, segment in enumerate(segments, 1):
            result += f"{i}\n"
            result += f"{faster_whisper.format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
            result += f"{faster_whisper.format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
            result += f"{segment.text.strip().replace('-->', '->')}\n"
            result += "\n"
        return result
    elif format == "rich_text":
        return "\n".join(segment.text.lstrip() for segment in segments)
    else:
        raise ValueError(
            f"unknown transcription format: {format}: expected one of 'plain_text', 'vtt', 'srt', 'rich_text'"
        )
    return result


def handle(
    event: dict[str, Any],
    model2Whisper: dict[str, faster_whisper.WhisperModel],
    device: Literal["cuda", "cpu"],
) -> dict[str, Any]:
    """
    use event.input.model to do a task on the audio file specified by input.audio or input.audio_base64,
    using the rest of event.input as keyword arguments to the task
    # 1. validate the input, set defaults, and tell the user about problems
    # 2. download the audio file to a temporary file (or write the base64 audio to the temporary file)
    # 3. transcribe the audio file.
    #   3a: format the the transcript in the specified format
    # 4 (optional): translate the transcript
    #   4a: format the translation in the specified format

    """

    # 1. validate the input, set defaults, and tell the user about problems

    input: dict[str, Any] = event["input"]
    id: str = event["id"]
    modelname: str = input["model"]
    if modelname not in model2Whisper:
        raise KeyError(
            f"id {id}: bad request: model {modelname} not found: available models are {model2Whisper.keys()}"
        )
    model = model2Whisper[modelname]
    # temperature is expected to be an iterable.
    temperature: Any = input.get("temperature", 0)
    get_word_timestamps: bool = input.get("word_timestamps", False)
    assert isinstance(get_word_timestamps, bool), "bad request: key 'word_timestamps'"
    if (inc := input.get("temperature_increment_on_fallback", 0.2)) is not None:
        temperature = np.arange(temperature, 1.0 + 1e-6, inc)
    elif isinstance(temperature, (int, float)):
        temperature = [temperature]

    task = input.get("task", "transcribe")
    assert task == "transcribe" or task == "translate", "bad request: key 'task'"

    transcription_format = input.get("transcription", "plain_text")
    assert transcription_format in (
        "plain_text",
        "vtt",
        "srt",
        "rich_text",
    ), "bad request: key 'transcription'"

    translate = input.get("translate", False)
    assert isinstance(translate, bool), "bad request: key 'translate'"
    translation_format = input.get("translation", "plain_text")
    assert translation_format in (
        "plain_text",
        "vtt",
        "srt",
        "rich_text",
    ), "bad request: key 'translation'"

    # use the rest of the input as keyword arguments to the task, but remove the model and audio keys, since we're going to set those manually
    kwargs = {
        k: v
        for k, v in input.items()
        if k
        not in [
            "audio_base64",
            "audio",
            "model",
            "temperature",
            "transcription",
            "translate",
            "translation",
            "word_timestamps",
        ]
    }

    # set default values for the keyword arguments if and only if they're not already set
    for k, default in {
        "task": "transcribe",
        "language": None,
        "temperature": 0,
        "best_of": 5,
        "beam_size": 5,
        "patience": 1,
        "length_penalty": None,
        "initial_prompt": None,
        "condition_on_previous_text": True,
        "temperature_increment_on_fallback": 0.2,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "enable_vad": False,
    }.items():
        if k not in kwargs:
            kwargs[k] = default

    if (inc := kwargs.get("temperature_increment_on_fallback", None)) is not None:
        kwargs["temperature"] = np.arange(kwargs["temperature"], 1.0 + 1e-6, inc)
    elif isinstance(kwargs["temperature"], (int, float)):
        kwargs["temperature"] = [
            kwargs["temperature"]
        ]  # turn it into a list so we can iterate over it

    # 2. download the audio file to a temporary file (or write the base64 audio to the temporary file)
    log(f"handling request {id}: model={modelname}")
    if "audio" in input and "audio_base64" in input:
        raise KeyError(
            f"id {id}: bad request: expected key 'audio' or 'audio_base64', but got both"
        )
    if "audio" not in input and "audio_base64" not in input:
        raise KeyError(
            f"id {id}: bad request: expected key 'audio' or 'audio_base64', but got neither"
        )
    log(f"id {id}: opening temporary file for audio")

    with tempfile.NamedTemporaryFile() as f:
        if "audio" in input:
            log(f"downloading audio from {input['audio']} to temporary file")
            assert isinstance(input["audio"], str)
            for chunk in requests.get(input["audio"], stream=True).iter_content(
                chunk_size=32 * 1024
            ):
                f.write(chunk)
        else:
            log("decoding base64 audio to temporary file")
            assert isinstance(input["audio_base64"], str)
            f.write(base64.b64decode(input["audio_base64"]))
        f.flush()  # flush the buffer to disk
        f.seek(0)  # seek back to the beginning of the file so we can read it later

        # 3. transcribe the audio file.
        log("transcribing audio...")
        start = time.time()
        _transcribe_segements, transcribe_info = model.transcribe(
            audio=f.name, **kwargs
        )
        transcribe_segments: list[faster_whisper.transcribe.Segment] = list(
            _transcribe_segements
        )
        log(f"transcribed audio in {time.time()-start:.2f}")

        # 4 (optiona): translate.

        translation_segments: list[faster_whisper.transcribe.Segment] | None = None
        if translate:
            start = time.time()
            log("translating...")
            translation_segments, _ = model.transcribe(
                audio=f.name, task="translate", temperature=temperature, **kwargs
            )
            log(f"translated in {time.time()-start:.2f}")

        def format_segment(segment):
            return {
                "id": segment.id,
                "seek": segment.seek,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "tokens": segment.tokens,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
            }

        results = {
            "detected_language": transcribe_info.language,
            "device": device,
            "model": modelname,
            "segments": [format_segment(segment) for segment in transcribe_segments],
            "transcription": format_transcript(
                format=transcription_format, segments=transcribe_segments
            ),
            "word_timestamps": (
                [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for s in transcribe_segments
                    for w in s.words
                ]
                if get_word_timestamps
                else None
            ),
            "translation": (
                format_transcript(
                    format=translation_format, segments=translation_segments
                )
                if translation_segments is not None
                else None
            ),
            "translation_segments": (
                [format_segment(segment) for segment in translation_segments]
                if translation_segments is not None
                else None
            ),
        }
        # drop None values
        # drop None values
        return {k: v for k, v in results.items() if v is not None}


def main() -> None:
    """
    run the serverless worker, loading models from the filesystem (specified by RUNPOD_HUGGINGFACE_MODEL via the modelcache)
    exactly once on boot, then delegate to the handle function
    """
    device: Literal["cuda", "cpu"]
    compute_type: Literal["float16", "int8"]
    log("checking gpu settings")
    device_index: Union[list[int], int]
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu", "count", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
        )
        p.check_returncode()
        device_index = list(
            range(int(p.stdout.decode().strip()))
        )  # i.e, " 4 " -> [0,1,2,3]
        log(f"gpu: found {len(device_index)} GPUs")
        compute_type = "float16"
        device = "cuda"

    except Exception as e:
        log(f"failed to get number of GPUs: {e}: falling back to CPU mode")
        device_index = 0
        compute_type = "int8"
        device = "cpu"

    print(f"runpod: loading models from filesystem...", file=sys.stderr)
    model2path = cachedmodel2path()
    model2Whisper = {}

    for model, path in model2path.items():
        log(f"runpod: loading model {model} from {path}")
        start = time.time()
        model2Whisper[model] = faster_whisper.WhisperModel(
            model_size_or_path=path,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            local_files_only=True,
            num_workers=8,
        )
        log(f"loaded model {model} in {time.time()-start:.2f}")

    friendlyModel2Whisper = {}
    # let's make things easier for our users.
    # we specify models by user/model/revision, but they might not know about git revisions -
    # ML customers are not experienced software developers.
    # we'll allow them to specify models by user/model, model, user/model/revision, and user/model:revision
    for path, whisper in model2Whisper.items():
        *_, user, model, revision = path.split("/")
        friendlyModel2Whisper[f"{user}/{model}"] = whisper
        friendlyModel2Whisper[f"{model}"] = whisper
        friendlyModel2Whisper[f"{user}/{model}/{revision}"] = whisper
        friendlyModel2Whisper[f"{user}/{model}:{revision}"] = whisper

    log(f"loaded {len(model2Whisper)} models: {model2Whisper.keys()}")
    log(f"starting serverless worker...")
    runpod.start(
        {
            "handler": lambda event: handle(
                event=event, model2Whisper=friendlyModel2Whisper, device=device
            ),
        }
    )


if __name__ == "__main__":
    main()
