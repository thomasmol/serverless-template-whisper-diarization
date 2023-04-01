import os
import time
import json
import wave
import torch
import base64
import whisper
import datetime
import contextlib
import numpy as np
import pandas as pd
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global model_name
    global embedding_model
    # medium, large-v1, large-v2
    model_name = "large-v2"
    model = whisper.load_model(model_name)
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))


def speech_to_text(filepath, whisper_model, num_speakers, prompt):
    model = whisper.load_model(whisper_model)
    time_start = time.time()

    try:
        _, file_ending = os.path.splitext(f'{filepath}')
        print(f'file enging is {file_ending}')
        audio_file_wav = filepath.replace(file_ending, ".wav")
        print("-----starting conversion to wav-----")
        os.system(
            f'ffmpeg -i "{filepath}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file_wav}"')
    except Exception as e:
        raise RuntimeError("Error converting audio")

    # Get duration
    with contextlib.closing(wave.open(audio_file_wav, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    print(f"conversion to wav ready, duration of audio file: {duration}")

    # Transcribe audio
    print("starting whisper")
    options = dict(beam_size=5, best_of=5)
    transcribe_options = dict(task="transcribe", **options)
    result = model.transcribe(
        audio_file_wav, **transcribe_options, initial_prompt=prompt)
    segments = result["segments"]
    print("done with whisper")

    try:
        # Create embedding
        def segment_embedding(segment):
            audio = Audio()
            start = segment["start"]
            # Whisper overshoots the end timestamp in the last segment
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(audio_file_wav, clip)
            return embedding_model(waveform[None])

        print("starting embedding")
        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)
        print(f'Embedding shape: {embeddings.shape}')

        # Assign speaker label
        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        # Make output
        output = []  # Initialize an empty list for the output
        for segment in segments:
            # Append the segment to the output list
            output.append({
                'start': str(convert_time(segment["start"])),
                'end': str(convert_time(segment["end"])),
                'speaker': segment["speaker"],
                'text': segment["text"]
            })

        print("done with embedding")
        time_end = time.time()
        time_diff = time_end - time_start

        system_info = f"""-----Processing time: {time_diff:.5} seconds-----"""
        print(system_info)
        os.remove(audio_file_wav)
        return output

    except Exception as e:
        os.remove(audio_file_wav)
        raise RuntimeError("Error Running inference with local model", e)


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model
    global model_name
    global embedding_model

    # Parse out your arguments
    filename = model_inputs.get('filename', '')
    prompt = model_inputs.get('prompt', 'an audio')
    base64file = model_inputs.get('file', None)
    number_speakers = model_inputs.get('num_speakers', 2)

    if base64file == None or base64file == '' or filename == '':
        return {'message': "No correct input provided"}

    # TODO: check if file is right format, can also be done in frontend
    base64file = base64file.split(',')[1] if ',' in base64file else base64file

    file_data = base64.b64decode(base64file)
    file_start, file_ending = os.path.splitext(f'{filename}')

    ts = time.time()
    ts = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    filename = f'{ts}-{file_start}{file_ending}'
    with open(filename, 'wb') as f:
        f.write(file_data)

    # filepath = f'uploads/{filename}'
    filepath = filename

    transcription = speech_to_text(
        filepath, model_name, number_speakers, prompt)
    # print for testing
    print(transcription)

    os.remove(filepath)
    print(f'{filepath} removed, done with inference')
    return transcription
