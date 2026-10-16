"""
This file contains the Predictor class, which runs transcription with
kotoba-whisper-v2.0-faster. The worker serves exactly one model and always
transcribes in Japanese.
"""

import os

import numpy as np

from runpod.serverless.utils import rp_cuda

from faster_whisper import WhisperModel
from faster_whisper.utils import format_timestamp

from model_config import (
    ACCEPTED_MODEL_NAMES,
    DEFAULT_CHUNK_LENGTH,
    LANGUAGE,
    MODEL_DIR,
    MODEL_NAME,
    MODEL_REPO,
)


class Predictor:
    """A Predictor class for kotoba-whisper, loaded once at startup."""

    def __init__(self):
        self.model = None

    def setup(self):
        """Load the single model into memory."""
        # Baked into the image by builder/fetch_models.py; fall back to the Hub
        # so the handler stays runnable outside the container (CI, local runs).
        source = MODEL_DIR if os.path.isdir(MODEL_DIR) else MODEL_REPO

        print(f"Loading {MODEL_NAME} from {source}...")
        self.model = WhisperModel(
            source,
            device="cuda" if rp_cuda.is_available() else "cpu",
            compute_type="float16" if rp_cuda.is_available() else "int8",
        )
        print(f"Model {MODEL_NAME} loaded successfully.")

    def predict(
        self,
        audio,
        model_name=MODEL_NAME,
        transcription="plain_text",
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=1,
        length_penalty=1,
        suppress_tokens="-1",
        initial_prompt=None,
        condition_on_previous_text=False,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        enable_vad=False,
        word_timestamps=False,
        chunk_length=DEFAULT_CHUNK_LENGTH,
    ):
        """
        Run a single transcription. The language is always Japanese.
        """
        if model_name not in ACCEPTED_MODEL_NAMES:
            raise ValueError(
                f"Invalid model '{model_name}'. This worker only serves "
                f"{MODEL_NAME} (alias: {MODEL_REPO})."
            )

        if self.model is None:
            raise RuntimeError("Predictor.setup() must be called before predict().")

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        segments, info = self.model.transcribe(
            str(audio),
            language=LANGUAGE,
            task="transcribe",
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=parse_suppress_tokens(suppress_tokens),
            without_timestamps=False,
            max_initial_timestamp=1.0,
            word_timestamps=word_timestamps,
            vad_filter=enable_vad,
            chunk_length=chunk_length,
        )

        segments = list(segments)

        results = {
            "segments": serialize_segments(segments),
            "detected_language": info.language,
            "transcription": format_segments(transcription, segments),
            "device": "cuda" if rp_cuda.is_available() else "cpu",
            "model": MODEL_NAME,
        }

        if word_timestamps:
            word_timestamps_list = []
            for segment in segments:
                for word in segment.words:
                    word_timestamps_list.append(
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                        }
                    )
            results["word_timestamps"] = word_timestamps_list

        return results


def parse_suppress_tokens(suppress_tokens):
    """
    Turn the comma-separated `suppress_tokens` input into the list of ints that
    faster-whisper expects.
    """
    if suppress_tokens is None or suppress_tokens == "":
        return [-1]

    if isinstance(suppress_tokens, (list, tuple)):
        candidates = suppress_tokens
    else:
        candidates = [
            token for token in str(suppress_tokens).split(",") if token.strip()
        ]

    try:
        return [int(token) for token in candidates]
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid suppress_tokens: {suppress_tokens!r}. "
            "Expected a comma-separated list of integers."
        ) from e


def serialize_segments(transcript):
    """
    Serialize the segments to be returned in the API response.
    """
    return [
        {
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
        for segment in transcript
    ]


def format_segments(format_type, segments):
    """
    Format the segments to the desired format
    """

    if format_type == "plain_text":
        return " ".join([segment.text.lstrip() for segment in segments])
    elif format_type == "formatted_text":
        return "\n".join([segment.text.lstrip() for segment in segments])
    elif format_type == "srt":
        return write_srt(segments)
    elif format_type == "vtt":
        return write_vtt(segments)
    else:
        print(f"Warning: Unknown format '{format_type}', defaulting to plain text.")
        return " ".join([segment.text.lstrip() for segment in segments])


def write_vtt(transcript):
    """
    Write the transcript in VTT format.
    """
    result = ""

    for segment in transcript:
        result += f"{format_timestamp(segment.start, always_include_hours=True)} --> {format_timestamp(segment.end, always_include_hours=True)}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result


def write_srt(transcript):
    """
    Write the transcript in SRT format.
    """
    result = ""

    for i, segment in enumerate(transcript, start=1):
        result += f"{i}\n"
        result += f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result
