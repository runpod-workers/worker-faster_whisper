"""
rp_handler.py for runpod worker

rp_debugger:
- Utility that provides additional debugging information.
The handler must be called with --rp_debugger flag to enable it.
"""
import base64
import os
import tempfile

from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict


MODEL = predict.Predictor()
MODEL.setup()


def base64_to_tempfile(base64_file: str) -> str:
    '''
    Convert base64 file to tempfile.

    Parameters:
    base64_file (str): Base64 file

    Returns:
    str: Path to tempfile
    '''
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))

    return temp_file.name


@rp_debugger.FunctionTimer
def run_whisper_job(job):
    '''
    Run inference on the model.

    Parameters:
    job (dict): Input job containing the model parameters

    Returns:
    dict: The result of the prediction
    '''
    job_input = job['input']

    with rp_debugger.LineTimer('validation_step'):
        input_validation = validate(job_input, INPUT_VALIDATIONS)

        if 'errors' in input_validation:
            return {"error": input_validation['errors']}
        job_input = input_validation['validated_input']

    if not job_input.get('audio', False) and not job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64'}

    if job_input.get('audio', False) and job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64, not both'}

    audio_input = None

    if job_input.get('audio', False):
        with rp_debugger.LineTimer('download_step'):
            downloaded = download_files_from_urls(job['id'], [job_input['audio']])
            audio_input = downloaded[0] if downloaded else None

        # Guard: download_files_from_urls returns None / "None" for failed
        # fetches (e.g. expired presigned URL → 403). Without this check the
        # handler would call MODEL.predict(audio=None) and crash inside PyAV
        # with FileNotFoundError 'None', leaving the caller without a usable
        # error signal and the credits already debited.
        if not audio_input or audio_input == "None" or not os.path.exists(audio_input):
            audio_url = job_input.get('audio') or ''
            audio_url_host = audio_url.split('/')[2] if audio_url.count('/') >= 2 else None
            return {
                "error": "DOWNLOAD_FAILED",
                "message": (
                    "Failed to download audio from the provided URL. "
                    "The URL may have expired, been revoked, or returned a non-2xx response."
                ),
                "audio_url_host": audio_url_host,
            }

    if job_input.get('audio_base64', False):
        audio_input = base64_to_tempfile(job_input['audio_base64'])
        if not audio_input or not os.path.exists(audio_input):
            return {
                "error": "BASE64_DECODE_FAILED",
                "message": "Failed to decode audio_base64 to a temp file.",
            }

    with rp_debugger.LineTimer('prediction_step'):
        whisper_results = MODEL.predict(
            audio=audio_input,
            model_name=job_input["model"],
            transcription=job_input["transcription"],
            translation=job_input["translation"],
            translate=job_input["translate"],
            language=job_input["language"],
            temperature=job_input["temperature"],
            best_of=job_input["best_of"],
            beam_size=job_input["beam_size"],
            patience=job_input["patience"],
            length_penalty=job_input["length_penalty"],
            suppress_tokens=job_input.get("suppress_tokens", "-1"),
            initial_prompt=job_input["initial_prompt"],
            condition_on_previous_text=job_input["condition_on_previous_text"],
            temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
            compression_ratio_threshold=job_input["compression_ratio_threshold"],
            logprob_threshold=job_input["logprob_threshold"],
            no_speech_threshold=job_input["no_speech_threshold"],
            enable_vad=job_input["enable_vad"],
            vad_parameters=job_input["vad_parameters"],
            word_timestamps=job_input["word_timestamps"]
        )

    with rp_debugger.LineTimer('cleanup_step'):
        rp_cleanup.clean(['input_objects'])

    return whisper_results


runpod.serverless.start({"handler": run_whisper_job})
