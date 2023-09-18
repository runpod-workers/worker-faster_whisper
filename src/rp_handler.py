'''
rp_handler.py for runpod worker

rp_debugger:
- Utility that provides additional debugging information.
The handler must be called with --rp_debugger flag to enable it.
'''

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
    import base64
    import tempfile

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

    if job_input['audio'].startswith('http'):
        with rp_debugger.LineTimer('download_step'):
            job_input['audio'] = download_files_from_urls(job['id'], [job_input['audio']])[0]
    else:
        job_input['audio'] = base64_to_tempfile(job_input['audio'])

    with rp_debugger.LineTimer('prediction_step'):
        whisper_results = MODEL.predict(
            audio=job_input["audio"],
            model_name=job_input["model"],
            transcription=job_input["transcription"],
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
        )

    with rp_debugger.LineTimer('cleanup_step'):
        rp_cleanup.clean(['input_objects'])

    return whisper_results


runpod.serverless.start({"handler": run_whisper_job})
