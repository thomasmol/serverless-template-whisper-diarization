# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests

model_payload = {
    "file":"somebase64string",
    "filename":"somefile.mp3",
    "prompt":"some audio",
    "num_speakers": 2
}

res = requests.post("http://localhost:8000/",json=model_payload)

print(res.text)
