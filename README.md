<div align="center">

<h1>Faster Whisper | Worker</h1>

This repository contains the [Faster Whisper](https://github.com/guillaumekln/faster-whisper) Worker for RunPod. The Whisper Worker is designed to process audio files using various Whisper models, with options for transcription formatting, language translation, and more. It's part of the RunPod Workers collection aimed at providing diverse functionality for endpoint processing.

[Endpoint Docs](https://docs.runpod.io/reference/faster-whisper)

[Docker Image](https://hub.docker.com/r/runpod/ai-api-faster-whisper)

</div>

## Model Inputs

| Input                               | Type  | Description                                                                                                                                              |
|-------------------------------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `audio`                             | Path  | Audio file                                                                                                                                               |
| `audio_base64`                      | str   | Base64-encoded audio file                                                                                                                                |
| `model`                             | str   | Choose a Whisper model. Choices: "tiny", "base", "small", "medium", "large-v1", "large-v2". Default: "base"                                              |
| `transcription`                     | str   | Choose the format for the transcription. Choices: "plain_text", "formatted_text", "srt", "vtt". Default: "plain_text"                                    |
| `translate`                         | bool  | Translate the text to English when set to True. Default: False                                                                                           |
| `language`                          | str   | Language spoken in the audio, specify None to perform language detection. Default: None                                                                  |
| `temperature`                       | float | Temperature to use for sampling. Default: 0                                                                                                              |
| `best_of`                           | int   | Number of candidates when sampling with non-zero temperature. Default: 5                                                                                 |
| `beam_size`                         | int   | Number of beams in beam search, only applicable when temperature is zero. Default: 5                                                                     |
| `patience`                          | float | Optional patience value to use in beam decoding. Default: None                                                                                           |
| `length_penalty`                    | float | Optional token length penalty coefficient (alpha). Default: None                                                                                         |
| `suppress_tokens`                   | str   | Comma-separated list of token ids to suppress during sampling. Default: "-1"                                                                             |
| `initial_prompt`                    | str   | Optional text to provide as a prompt for the first window. Default: None                                                                                 |
| `condition_on_previous_text`        | bool  | If True, provide the previous output of the model as a prompt for the next window. Default: True                                                         |
| `temperature_increment_on_fallback` | float | Temperature to increase when falling back when the decoding fails. Default: 0.2                                                                          |
| `compression_ratio_threshold`       | float | If the gzip compression ratio is higher than this value, treat the decoding as failed. Default: 2.4                                                      |
| `logprob_threshold`                 | float | If the average log probability is lower than this value, treat the decoding as failed. Default: -1.0                                                     |
| `no_speech_threshold`               | float | If the probability of the token is higher than this value, consider the segment as silence. Default: 0.6                                                 |
| `enable_vad`                        | bool  | If True, use the voice activity detection (VAD) to filter out parts of the audio without speech. This step is using the Silero VAD model. Default: False |

## Test Inputs

The following inputs can be used for testing the model:

```json
{
    "input": {
        "audio": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav"
    }
}
```

## Sample output
```json
{
    "segments": [
        {
            "id": 1,
            "seek": 106,
            "start": 0.11,
            "end": 3.11,
            "text": " Hola mundo.",
            "tokens": [
                50364,
                22637,
                7968,
                13,
                50514
            ],
            "temperature": 0.1,
            "avg_logprob": -0.8348079785480325,
            "compression_ratio": 0.5789473684210527,
            "no_speech_prob": 0.1453857421875
        }
    ],
    "detected_language": "es",
    "transcription": "Hola mundo.",
    "translation": null,
    "device": "cuda",
    "model": "large-v2",
    "translation_time": 0.7796223163604736
}
````