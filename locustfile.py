import io

import numpy as np
from locust import HttpUser, task
import base64
from pydub import AudioSegment


def generate_random_audio(duration_ms):
    # Generate random data
    samples = np.random.normal(0, 1, int(44100 * duration_ms / 1000.0))

    # Convert to int16 array so we can make use of the pydub package
    samples = (samples * np.iinfo(np.int16).max).astype(np.int16)

    # Create an audio segment
    audio_segment = AudioSegment(
        samples.tobytes(),
        frame_rate=44100,
        sample_width=samples.dtype.itemsize,
        channels=1
    )

    # Convert the audio segment to a base64 string
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="wav")
    base64_audio = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return base64_audio

class ApiUser(HttpUser):
    @task
    def send_audio_request(self):
        headers = {
            'Content-Type': 'application/json',
        }
        audio_data = generate_random_audio(1000)    # 1 second audio
        
        data = {
            "input": {
                "audio": audio_data
            }
        }
        
        self.client.post("/v2/xxxxx/runsync", json=data, headers=headers)  # Replace with your endpoint ID

if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py")
