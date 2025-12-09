"""
rp_handler.py for runpod worker
"""
import base64
import tempfile
import json  # Import JSON to handle results formatting
import os    # Import OS to read environment variables
from io import BytesIO # To handle binary data in memory

from supabase import create_client, Client 

from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict


MODEL = predict.Predictor()
MODEL.setup()


# 2. NEW FUNCTION: Supabase Upload
def upload_to_supabase(job_id: str, results: dict) -> str:
    """
    Connects to Supabase and uploads the transcription results.
    
    Parameters:
    job_id (str): RunPod job ID, used as the filename prefix.
    results (dict): The dictionary containing the transcription output.
    
    Returns:
    str: The publicly accessible URL of the uploaded file.
    """
    # 2a. Securely retrieve credentials from environment variables
    supabase_url: str = os.environ.get("SUPABASE_URL")
    supabase_key: str = os.environ.get("SUPABASE_KEY")
    supabase_bucket: str = os.environ.get("SUPABASE_BUCKET_NAME", "transcriptions")

    if not supabase_url or not supabase_key:
        print("ERROR: SUPABASE_URL or SUPABASE_KEY is missing from environment variables.")
        raise EnvironmentError("Supabase credentials not configured.")

    # 2b. Initialize Supabase Client
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # 2c. Prepare the data for upload (convert dict results to JSON bytes)
    file_name = f"{job_id}_transcription.json"
    file_path = f"transcripts/{file_name}" # Folder path in the bucket
    
    transcription_json = json.dumps(results, indent=4)
    data = BytesIO(transcription_json.encode('utf-8'))
    
    try:
        # 2d. Upload the file
        upload_response = supabase.storage.from_(supabase_bucket).upload(
            file_path, 
            data.getvalue(),
            file_options={"content-type": "application/json"}
        )

        # 2e. Get the public URL
        public_url_response = supabase.storage.from_(supabase_bucket).get_public_url(file_path)

        print(f"File uploaded successfully to: {public_url_response}")
        return public_url_response

    except Exception as e:
        print(f"SUPABASE UPLOAD FAILED: {e}")
        raise RuntimeError(f"Failed to upload transcription results to Supabase: {e}")


def base64_to_tempfile(base64_file: str) -> str:
    '''
    Convert base64 file to tempfile.
    ... (no change) ...
    '''
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))

    return temp_file.name


@rp_debugger.FunctionTimer
def run_whisper_job(job):
    '''
    Run inference on the model and handle file upload.
    '''
    job_input = job['input']
    
    # ... (validation and audio download steps remain the same) ...

    # Check for callback_job_id (Your internal NestJS ID)
    if 'callback_job_id' not in job_input:
        return {"error": "callback_job_id is required in the input for webhook updates."}

    # ... (prediction step remains the same) ...
    with rp_debugger.LineTimer('prediction_step'):
        whisper_results = MODEL.predict(
            # ... (all model parameters remain the same) ...
            enable_vad=job_input["enable_vad"],
            word_timestamps=job_input["word_timestamps"]
        )

    # 3. NEW STEP: Upload results to Supabase and get URL
    with rp_debugger.LineTimer('supabase_upload_step'):
        try:
            supabase_url = upload_to_supabase(job['id'], whisper_results)
            
            # 4. Final Output Construction
            # This structured output is what RunPod will send to your NestJS webhook.
            final_output = {
                "transcriptionPath": supabase_url,
                "word_count": len(whisper_results.get('text', '').split()) # Example metric
            }

        except (EnvironmentError, RuntimeError) as e:
            # If upload fails, return an error to RunPod so the webhook reports FAILED
            return {"error": str(e)}

    with rp_debugger.LineTimer('cleanup_step'):
        rp_cleanup.clean(['input_objects'])

    # Return the structured final_output object
    return final_output


runpod.serverless.start({"handler": run_whisper_job})
