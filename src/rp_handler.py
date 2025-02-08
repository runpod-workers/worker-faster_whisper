from typing import Any
import base64
import predict
import requests
import runpod
import tempfile
def run(model: predict.Predictor, job: dict[str, Any]) -> dict[str, Any]:
    input: dict[str, Any] = (job['input'])
    id: str = job['id']

    # open a temporary file to write the audio content
    # either we'll download the audio from a URL ("audio" key)
    # or just dump base64 content to the file ("audio_base64" key)
    with tempfile.NamedTemporaryFile(suffix=id) as tmp:
        if "audio" in input and "audio_base64" in input: # both keys present: don't guess
            raise ValueError("bad request: expected keys 'audio' or 'audio_base64', not both")
        if (audiopath := input.get('audio', None)) is not None and (resp := requests.get(audiopath, stream=True)).status_code == 200: # URL
            for chunk in resp.iter_content(1024*32): # 32KiB chunks
                tmp.write(chunk)   
        elif (audiopath := input.get('audio_base64', None)) is not None: # base64 encoded audio
            tmp.write(base64.b64decode(audiopath))
        else:
            raise ValueError("bad request: expected keys 'audio' or 'audio_base64', not both")
        tmp.seek(0)
        # pass the rest of the input in as kwargs; remove the audio and model keys
        kwargs = {k: v for k, v in input.items() if k not in {"audio", "audio_base64", "model", "model_name"}} 

        return model.predict(
            audio=tmp.name,
            model_name=input.get("model", input.get("model_name", "base")),
            **kwargs
        )

if __name__ == "__main__":
    MODEL = predict.Predictor()
    MODEL.setup()
    runpod.serverless.start({"handler": lambda job: run(MODEL, job)})
