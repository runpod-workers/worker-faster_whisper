"""
This file contains the Predictor class, which is used to run predictions on the
Whisper model. It is based on the Predictor class from the original Whisper
repository, with some modifications to make it work with the RP platform.
"""


from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np

from faster_whisper import WhisperModel
from faster_whisper.utils import format_timestamp


class Predictor:
    ''' A Predictor class for the Whisper model '''

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.models = {}

        def load_model(model_name):
            '''
            Load the model from the weights folder.
            '''
            loaded_model = WhisperModel(
                model_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
                compute_type="float16" if torch.cuda.is_available() else "int8")

            return model_name, loaded_model

        model_names = ["tiny", "base", "small", "medium", "large-v1", "large-v2"]
        with ThreadPoolExecutor() as executor:
            for model_name, model in executor.map(load_model, model_names):
                if model_name is not None:
                    self.models[model_name] = model

    def predict(
        self,
        audio,
        model_name="base",
        transcription="plain text",
        translate=False,
        language=None,
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=1,
        length_penalty=None,
        suppress_tokens="-1",
        initial_prompt=None,
        condition_on_previous_text=True,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    ):
        """
        Run a single prediction on the model
        """
        model = self.models[model_name]

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        segments, info = list(model.transcribe(str(audio),
                                               language=language,
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
                                               suppress_tokens=[-1],
                                               without_timestamps=False,
                                               max_initial_timestamp=1.0,
                                               word_timestamps=False
                                               ))

        segments = list(segments)

        # if transcription == "plain_text":
        #     transcription = result["text"]
        # elif transcription == "srt":
        #     transcription = write_srt(result["segments"])
        # else:
        #     transcription = write_vtt(result["segments"])

        if transcription == "srt":
            transcription = write_srt(segments)
        else:
            transcription = write_vtt(segments)

        if translate:
            translation_segments, translation_info = model.transcribe(str(audio), task="translate", temperature=temperature
                                                                      )

        return {
            "segments": format_segments(segments),
            "detected_language": info.language,
            "transcription": transcription,
            "translation": write_srt(translation_segments) if translate else None,
        }


def format_segments(transcript):
    '''
    Format the segments to be returned in the API response.
    '''
    return [{
        "id": segment.id,
        "seek": segment.seek,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "tokens": segment.tokens,
        "temperature": segment.temperature,
        "avg_logprob": segment.avg_logprob,
        "compression_ratio": segment.compression_ratio,
        "no_speech_prob": segment.no_speech_prob
    } for segment in transcript]


def write_vtt(transcript):
    '''
    Write the transcript in VTT format.
    '''
    result = ""

    for segment in transcript:
        result += f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result


def write_srt(transcript):
    '''
    Write the transcript in SRT format.
    '''
    result = ""

    for i, segment in enumerate(transcript, start=1):
        result += f"{i}\n"
        result += f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result
