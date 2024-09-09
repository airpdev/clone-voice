import time
import asyncio
import requests
import json
import uuid
from fastapi import FastAPI, Query, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
cleanvoice_url = "https://api.cleanvoice.ai/v1/edits"
elevenlabs_url = "https://api.elevenlabs.io/v1/voices"

CLEAN_VOICE_KEY = "Sr7d2GRA42LJNjVj7cWNjKqhPiX2vLAi"
ELEVENLABS_API_KEY = "sk_ea133ae12a60b1acb72896e7e315f2ea7454ae178cac0a05" 

@app.get("/")
def root():
    return time.time()

class CleanInfo(BaseModel):
    voice_url: str
    
@app.post("/clean_voice")
@limiter.limit("500/minute")
async def clean_voice(request: Request, payload: CleanInfo):
    loop = asyncio.get_event_loop()
    voice_response = await loop.run_in_executor(None, cleanvoice, payload.voice_url)
    return {"voice_clean_data" : voice_response}

class CloneInfo(BaseModel):
    transcript: str

@app.post("/clone_voice")
@limiter.limit("500/minute")
async def clone_voice(request: Request, payload: CloneInfo):
    loop = asyncio.get_event_loop()
    voice_response = await loop.run_in_executor(None, clonevoice, payload.transcript)
    return {"voice_clean_data" : voice_response}

def clonevoice(transcript: str):
    CHUNK_SIZE = 1024  # Size of chunks to read/write at a time
    VOICE_ID = "Q8GD5kDYg05ZJaHrM5fA"  # ID of the voice model to use
    TEXT_TO_SPEAK = transcript  # Text you want to convert to speech
    OUTPUT_PATH = "output/" + str(uuid.uuid4()) + ".mp3" # Path to save the output audio file
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
    
    headers = {
        "Accept": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": TEXT_TO_SPEAK,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }
    response = requests.post(tts_url, headers=headers, json=data, stream=True)
    
    # Check if the request was successful
    if response.ok:
        # Open the output file in write-binary mode
        with open(OUTPUT_PATH, "wb") as f:
            # Read the response in chunks and write to the file
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
        # Inform the user of success
        return OUTPUT_PATH
    else:
        # Print the error message if the request was not successful
        return response.text
        
  
def cleanvoice(voice_url: str):
    # upload audio to cleanvoice
    cleanvoice_headers = {"Content-Type": "application/json", "X-API-Key": CLEAN_VOICE_KEY}
    data  = {
                "input": {
                    "files": [voice_url],
                    "config": {}
                }
            }
    response = requests.post(cleanvoice_url, headers=cleanvoice_headers, json=data)
    response_json = response.json()
    
    cleanvoice_get_url = cleanvoice_url + "/" + response_json["id"]
    response_json = get_voice(cleanvoice_get_url)

    if response_json["status"] == "SUCCESS":
        return response_json["url"]
    else:
        return ""
    
# get denoised audio from cleanvoice
def get_voice(url: str):
    cleanvoice_headers = {"Content-Type": "application/json", "X-API-Key": CLEAN_VOICE_KEY}
    response = requests.get(url, headers=cleanvoice_headers)
    response_json = response.json()
    if response_json["status"] == "SUCCESS":
        return {"status" : response_json["status"], "url" : response_json["result"]["download_url"]}
    if response_json["status"] == "FAILURE":
        return {"status" : response_json["status"], "url" : ""}
    
    # delay 3 seconds
    time.sleep(3)
    return get_voice(url)