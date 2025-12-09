"""
rp_handler.py for runpod worker

rp_debugger:
- Utility that provides additional debugging information.
The handler must be called with --rp_debugger flag to enable it.
"""
import base64
import tempfile
import json
import os
from io import BytesIO

from supabase import create_client, Client

from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict


MODEL = predict.Predictor()
MODEL.setup()


def upload_to_supabase(job_id: str, results: dict) -> str:
    """
    Connects to Supabase and uploads the transcription results.

    Parameters:
    job_id (str): RunPod job ID, used as the filename prefix.
    results (dict): The dictionary containing the transcription output.

    Returns:
    str: The publicly accessible URL of the uploaded file.
    """
    supabase_url: str = os.environ.get("SUPABASE_URL")
    supabase_key: str = os.environ.get("SUPABASE_KEY")
    supabase_bucket: str = os.environ.get("SUPABASE_BUCKET_NAME", "transcriptions")

    if not supabase_url or not supabase_key:
        raise EnvironmentError("SUPABASE_URL or SUPABASE_KEY is missing from environment variables.")

    supabase: Client = create_client(supabase_url, supabase_key)

    file_name = f"{job_id}_transcription.json"
    file_path = f"transcripts/{file_name}"

    transcription_json = json.dumps(results, indent=2)
    data = BytesIO(transcription_json.encode('utf-8'))

    supabase.storage.from_(supabase_bucket).upload(
        file_path,
        data.getvalue(),
        file_options={"content-type": "application/json"}
    )

    public_url = supabase.storage.from_(supabase_bucket).get_public_url(file_path)
    return public_url


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

    if job_input.get('audio', False):
        with rp_debugger.LineTimer('download_step'):
            audio_input = download_files_from_urls(job['id'], [job_input['audio']])[0]

    if job_input.get('audio_base64', False):
        audio_input = base64_to_tempfile(job_input['audio_base64'])

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
            word_timestamps=job_input["word_timestamps"]
        )

    # Upload results to Supabase
    with rp_debugger.LineTimer('supabase_upload_step'):
        try:
            supabase_url = upload_to_supabase(job['id'], whisper_results)
        except Exception as e:
            return {"error": f"Failed to upload to Supabase: {str(e)}"}

    with rp_debugger.LineTimer('cleanup_step'):
        rp_cleanup.clean(['input_objects'])

    return {
        "transcription_url": supabase_url,
        "detected_language": whisper_results.get("detected_language"),
        "model": whisper_results.get("model"),
    }


runpod.serverless.start({"handler": run_whisper_job})
