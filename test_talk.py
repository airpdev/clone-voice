import soundfile as sf
import parselmouth
import librosa
import numpy as np
import time
import torchaudio
import torch
import boto3
import ffmpeg
import os
import asyncio
import requests
import json
import uuid
import threading
import subprocess
import speech_recognition as sr
import rvc.engine.rvc_workspace as rvc_ws
from rvc.engine.rvc import gen
import nltk
import shutil
from pydub import AudioSegment
from pydub.silence import split_on_silence
# from sklearn.metrics.pairwise import cosine_similarity
from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer
from bark.api import generate_audio_new, generate_audio
from bark.generation import load_codec_model, SAMPLE_RATE, preload_models_new
from scipy.io.wavfile import write as write_wav
from fastapi import FastAPI, Query, Request, File, Form, UploadFile, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from hubert.hubert_manager import HuBERTManager
from Levenshtein import distance
from pydantic import BaseModel
from urllib.parse import urlparse
import openai
from gtts import gTTS

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

openai.api_key = ""

@app.get("/")
def root():
    return time.time()

@app.post("/generate_answer")
@limiter.limit("500/minute")
async def generate_answer(request: Request, user_id: str = Form(...), file: UploadFile = File(...)):
    try:
        # Read the file content
        file_content = await file.read()
        file_location = "output/" + str(uuid.uuid4()) + ".wav"    
        # Save the uploaded file to the specified location
        with open(file_location, "wb") as file_object:
            file_object.write(file_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, 
                                        process_generate_answer, 
                                        user_id,
                                        file_location)
    return response

def process_generate_answer(user_id: str, file_location: str):
    output_path = "output/" + str(uuid.uuid4()) + ".wav"
    stream = ffmpeg.input(file_location)
    stream = ffmpeg.output(stream, output_path, acodec='pcm_s16le', ar="48000", ac="1")
    ffmpeg.run(stream)
    
    try:
        with sr.AudioFile(output_path) as source:
            r = sr.Recognizer()
            audio = r.record(source)
        transcript = r.recognize_google(audio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    answer = get_answer(transcript)
    answer_voice_location = generate_answer_voice(answer)
    return {"status" : "success", "transcript" : transcript, "answer" : answer, "answer_voice" : answer_voice_location}

def get_answer(question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    )
    answer = response['choices'][0]['message']['content']
    return answer

def generate_answer_voice(transcript):
    answer_path = "output/" + str(uuid.uuid4()) + ".wav"
    # audio_array = generate_audio_new(transcript, history_prompt="v2/en_speaker_0")
    # write_wav(output_path, SAMPLE_RATE, audio_array)
    tts = gTTS(text=transcript, lang='en')
    tts.save(answer_path)
    return answer_path