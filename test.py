# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import os

# the file is a base64 encoded audio file of a few people talking about the moonlanding
# the morning after it happened (1969)

# open uploads/textbase64audio.txt and save to a var
with open('uploads/testbase64audio.txt', 'r') as f:
    base64file = f.read()

model_payload = {
    "file":base64file,
    "filename":"somefile.mp3",
    "prompt":"people talking about the moonlanding",
    "num_speakers": 3
}

res = requests.post("http://localhost:8000/",json=model_payload)

print(res.text)
