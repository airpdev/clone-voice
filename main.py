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
from fastapi import FastAPI, Query, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from hubert.hubert_manager import HuBERTManager
from Levenshtein import distance
from pydantic import BaseModel
from urllib.parse import urlparse

class TranscriptInfo:
    def __init__(self, file_path, transcript_similarity):
        self.file_path = file_path
        self.transcript_similarity = transcript_similarity
        
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", default=None)
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", default=None)
CLEAN_VOICE_KEY = os.getenv("CLEAN_VOICE_KEY", default=None)

device = 'cpu' # or 'cuda'
if torch.cuda.is_available():
    device = 'cuda'
hubert_model = HuBERTManager()
hubert_model.make_sure_hubert_installed()
hubert_model.make_sure_tokenizer_installed()
hubert_model = CustomHubert(checkpoint_path='data/models/hubert/hubert.pt')
tokenizer = CustomTokenizer.load_from_checkpoint('data/models/hubert/tokenizer_large.pth', map_location=torch.device(device))
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
preload_models_new()
model = load_codec_model(use_gpu=torch.cuda.is_available())
cleanvoice_url = "https://api.cleanvoice.ai/v1/edits"
nltk.download('punkt')
lock = asyncio.Lock()

@app.get("/")
def root():
    return time.time()

class GenInfo(BaseModel):
    voice_id: str
    transcript: str
    prefix: str

@app.post("/generate_voice")
@limiter.limit("500/minute")
async def generate_voice(request: Request, payload: GenInfo):
    loop = asyncio.get_event_loop()
    voice_response = await loop.run_in_executor(None, 
                                                process_generate_voice, 
                                                payload.voice_id, 
                                                payload.transcript, 
                                                payload.prefix)
    return voice_response

@app.post("/clean_voice")
@limiter.limit("500/minute")
async def clean_voice(request: Request, s3_url: str = Query(None)):
    loop = asyncio.get_event_loop()
    voice_response = await loop.run_in_executor(None, cleanvoice, s3_url)
    return {"voice_clean_data" : voice_response}

class AddVoiceInfo(BaseModel):
    dataset_region: str
    dataset_bucket: str
    dataset_key: str

@app.post("/add_voice")
@limiter.limit("500/minute")
async def add_voice(request: Request, payload: AddVoiceInfo):
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, 
                                        process_add_voice, 
                                        payload.dataset_region, 
                                        payload.dataset_bucket, 
                                        payload.dataset_key)
    return response

class CloneInfo(BaseModel):
    voice_id: str
    transcript: str
    total_try_count: int

class ConvertInfo(BaseModel):
    voice_id: str
    voice_b_url: str

def merge_audio_files(input_files, output_file):
    # Create the ffmpeg command
    command = ['ffmpeg']
    
    # Add input files to the command
    for file in input_files:
        command.extend(['-i', file])
    
    # Add the filter_complex flag and the concat filter
    filter_complex = f'concat=n={len(input_files)}:v=0:a=1'
    command.extend(['-filter_complex', filter_complex])
    
    # Specify the output file
    command.append(output_file)
    
    try:
        # Execute the ffmpeg command
        subprocess.run(command, check=True)
        print(f"Audio files merged successfully into {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while merging the audio files: {e}")

@app.post("/convert_voice")
@limiter.limit("500/minute")
async def convert_voice(request: Request, payload: ConvertInfo):
    b_file_path = "output/" + str(uuid.uuid4()) + ".wav"
    b_file_path = download_file_url(payload.voice_b_url, b_file_path)
    voices = []
    voices.append(b_file_path)
    converted_array = process_rvc_model(payload.voice_id, voices)
    
    audio_name = str(uuid.uuid4())
    output_path = "output/" + audio_name + ".wav"
    stream = ffmpeg.input(converted_array[0])
    stream = ffmpeg.output(stream, output_path, acodec='pcm_s24le', ar="48000", ac="1")
    ffmpeg.run(stream)

    bucket_name = 'voice-dev1'
    object_name = 'tmp_audio/' + audio_name + ".wav"
    extra_args = {'ACL': 'public-read'}
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    s3 = session.client('s3')
    s3.upload_file(output_path, bucket_name, object_name, ExtraArgs=extra_args)
    s3_url = "https://{}.s3.us-east-2.amazonaws.com/{}".format(bucket_name, object_name)
    return {"status" : "success", "voice_id" : payload.voice_id, "voice_url" : s3_url}

@app.post("/clone_voice")
@limiter.limit("500/minute")
async def clone_voice(request: Request, payload: CloneInfo):
    voice_output_path = 'bark/assets/prompts/' + payload.voice_id + '.npz'
    model_path = payload.voice_id + "/" + payload.voice_id + ".pth"
    if os.path.exists(voice_output_path) == False:
        return {"status" : "failed", "voice_id" : payload.voice_id, "msg" : "npz file for voice isn't existed."}
    if os.path.exists("./data/models/rvc/" + model_path) == False:
        return {"status" : "failed", "voice_id" : payload.voice_id, "msg" : "pth file for voice isn't existed."}

    loop = asyncio.get_event_loop()
    sentences = nltk.sent_tokenize(payload.transcript)
    print(sentences)
    silence = np.zeros(int(0.03 * SAMPLE_RATE))  
    selected_files = []
    temp_files = []
    for sentence in sentences:
        cadidate_array = await loop.run_in_executor(None, 
                                        process_clone_voice, 
                                        payload.voice_id, 
                                        sentence, 
                                        payload.total_try_count)
        selected_array = filter_audio_array(cadidate_array, 1, get_reference("", payload.voice_id), sentence)
        selected_files.append(selected_array[0])
        for path in cadidate_array:
            temp_files.append(path)
    
    audio_name = uuid.uuid4()
    output_path = "output/" + str(audio_name) + ".wav"
    merge_audio_files(selected_files, output_path)
    bucket_name = 'voice-dev1'
    object_name = 'tmp_audio/' + str(audio_name) + ".wav"
    extra_args = {'ACL': 'public-read'}
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    s3 = session.client('s3')
    s3.upload_file(output_path, bucket_name, object_name, ExtraArgs=extra_args)
    s3_url = "https://{}.s3.us-east-2.amazonaws.com/{}".format(bucket_name, object_name)
    
    for temp in temp_files:
        os.remove(temp)
    print("finished cloning with rvc model")
    print(s3_url)

    if payload.gen_prefix == "music":
        return {"status" : "success", "voice_id" : payload.voice_id, "voice_url" : s3_url, "voice_clean_data" : ""}
    response = await loop.run_in_executor(None, cleanvoice, s3_url)
    return {"status" : "success", "voice_id" : payload.voice_id, "voice_url" : s3_url, "voice_clean_data" : response}

    # response_arry = []
    # for s3_url in s3_urls:
    #     response = await loop.run_in_executor(None, cleanvoice, s3_url)
    #     response_arry.append(response)
    # return {"status" : "success", "voice_id" : payload.voice_id, "voice_urls" : s3_urls, "voice_clean_data" : response_arry}

class ProsodyInfo(BaseModel):
    voice_id: str
    transcript: str
    urls: list[str]
    candidate_count: int

@app.post("/prosody_select")
@limiter.limit("500/minute")
async def prosody_select(request: Request, payload: ProsodyInfo):
    reference_path = get_reference("", payload.voice_id)   
    if os.path.exists(reference_path) == False:
        return {"status" : "failed", "voice_id" : payload.voice_id, "msg" : "reference file for voice isn't existed."}

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, 
                                        process_prosody_select, 
                                        payload.voice_id,
                                        payload.transcript,
                                        payload.urls, 
                                        payload.candidate_count)
    return response

def process_prosody_select(voice_id: str,
                      transcript: str, 
                      urls: list[str], 
                      candidate_count: int):
    reference_path = get_reference("", voice_id)   
    audio_array = []
    for url in urls:
        tmp_path = "output/" + str(uuid.uuid4()) + ".wav"
        tmp_path = download_file_url(url, tmp_path)
        if len(tmp_path) > 0:
            audio_array.append(tmp_path)
    selected_array = []
    for bark_voice in filter_audio_array(audio_array, candidate_count, reference_path, transcript):
        print(bark_voice)
        bucket_name = 'voice-dev1'
        audio_name = uuid.uuid4()
        object_name = 'tmp_audio/' + str(audio_name) + ".wav"
        extra_args = {'ACL': 'public-read'}
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        s3 = session.client('s3')
        s3.upload_file(bark_voice, bucket_name, object_name, ExtraArgs=extra_args)
        s3_url = "https://{}.s3.us-east-2.amazonaws.com/{}".format(bucket_name, object_name)
        selected_array.append(s3_url)
    for temp in audio_array:
        os.remove(temp)
    return {"status" : "success", "voice_id" : voice_id, "msg" : "", "voice_urls": selected_array}

def remove_silence(audio, silence_threshold=-50, min_silence_duration=500):
    non_silent_audio = AudioSegment.empty()
    chunks = split_on_silence(audio, silence_thresh=silence_threshold, min_silence_len=min_silence_duration)
    for chunk in chunks:
        non_silent_audio += chunk
    return non_silent_audio

def get_reference(file_path, voice_id):
    reference_path = "references/reference-rob1.wav" #"output/" + voice_id + "_ref.wav"
    if os.path.exists(reference_path):
        return reference_path
    audio = AudioSegment.from_file(file_path)
    modified_audio = remove_silence(audio)
    adjusted_audio = modified_audio.apply_gain(4)
    desilenced_path = str(uuid.uuid4()) + ".wav"
    adjusted_audio.export(desilenced_path, format="wav")

    audio_duration = get_audio_duration(desilenced_path)
    if audio_duration < 13.0:
        return None
        
    command = [
        "ffmpeg",
        "-i", desilenced_path,
        "-ss", str(0.0),
        "-to", str(13.0),
        "-vn",
        "-c:a", "pcm_s16le",
        "-ar", str(model.sample_rate),
        "-ac", str(model.channels),
        reference_path
    ]
    subprocess.run(command)
    os.remove(desilenced_path)
    return reference_path

def process_add_voice(dataset_region: str, 
                      dataset_bucket: str, 
                      dataset_key: str):
    voice_id = str(uuid.uuid4())
    # download dataset and extract reference for using bark model.
    reference_path = None
    current_directory = os.path.abspath(os.getcwd())
    dataset_path = current_directory + "/dataset/" + voice_id + "/"
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(dataset_path, exist_ok=True)
    try:
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        s3 = session.client('s3', region_name=dataset_region)
        objects = s3.list_objects_v2(Bucket=dataset_bucket, Prefix=dataset_key)
        for obj in objects['Contents']:
            key = obj['Key']
            filename = key.split("/")[-1]
            file_path = dataset_path + filename
            if ".wav" in filename:
                s3.download_file(dataset_bucket, key, file_path)
                if reference_path == None:
                    reference_path = get_reference(file_path, voice_id)
    except Exception:
        return {"status" : "failed", "msg" : "failed to download dataset for training rvc model."}
    if reference_path == None:
        return {"status" : "failed", "msg" : "failed to get reference audio for using in bark model."}
    # generate voice.npz for bark model ###########################################################################################
    bark_model_path = 'bark/assets/prompts/' + voice_id + '.npz'
    if os.path.exists(bark_model_path):
        os.remove(bark_model_path)
    # bark_ref_name = uuid.uuid4()
    # bark_ref_path = "output/" + str(bark_ref_name) + ".wav"
    # bark_ref_path = download_s3_file(bark_ref_region, bark_ref_bucket, bark_ref_key, bark_ref_path)
    # bark_ref_output = bark_ref_path + ".wav"
    # # getting semantic_tokens 
    bark_ref_output = reference_path + ".wav"
    stream = ffmpeg.input(reference_path)
    stream = ffmpeg.output(stream, bark_ref_output, acodec='pcm_s16le', ar=model.sample_rate, ac=model.channels)
    ffmpeg.run(stream)
    wav, sr = torchaudio.load(bark_ref_output)
    semantic_vectors = hubert_model.forward(wav, input_sample_hz=sr)
    semantic_tokens = tokenizer.get_token(semantic_vectors)
    # getting npz 
    wav = wav.unsqueeze(0).to(device)
    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]
    # move codes to cpu
    codes = codes.cpu().numpy()
    np.savez(bark_model_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)
    
    # train rvc model ################################################################################################################
    rvc_model_path = "./data/models/rvc/" + voice_id + "/" + voice_id + ".pth"
    if os.path.exists(rvc_model_path):
        os.remove(rvc_model_path)
    os.remove(bark_ref_output)
    asyncio.run(train_rvc_model(voice_id, dataset_path))
    return {"status" : "success", "voice_id" : voice_id, "msg" : "voice has been added successfully."}

async def train_rvc_model(voice_id: str, dataset_path : str):
    await lock.acquire()
    try:
        rvc_workspace: RvcWorkspace = None
        try:
            rvc_workspace = rvc_ws.RvcWorkspace(voice_id).load()
        except Exception:
            rvc_workspace = rvc_ws.RvcWorkspace(voice_id).create({
                    'vsr': 'v2 40k'
            })   
        finally:  
            rvc_workspace.data['dataset'] = dataset_path
            rvc_workspace.save()    
        # resample and split dataset
        response = rvc_ws.process_dataset(rvc_workspace)
        # extract pitches
        response = rvc_ws.pitch_extract(rvc_workspace)
        # create index file
        response = rvc_ws.create_index(rvc_workspace)
        # train rvc model 
        response = rvc_ws.train_model(rvc_workspace, 'f0', 101)
        # copy to RVC models
        response = rvc_ws.copy_model(rvc_workspace, "e_100")
        print(response)
    finally:
        lock.release()

def process_clone_voice(voice_id: str, 
                        transcript: str, 
                        total_try_count: int):
    voice_output_path = 'bark/assets/prompts/' + voice_id + '.npz'
    current_directory = os.path.abspath(os.getcwd())
    dataset_path = current_directory + "/dataset/" + voice_id + "/"
    if os.path.exists(dataset_path) == False:
        os.mkdir(dataset_path)

    bark_voices = []
    max_threads_count = 1
    thread_index = 0
    while (thread_index < total_try_count):
        threads = []
        for index in range(thread_index, thread_index + max_threads_count):
            if (index >= total_try_count):
                break
            voice_name = uuid.uuid4()
            output_path = "output/" + str(voice_name) + ".wav"
            bark_voices.append(output_path)
            thread = threading.Thread(target=generate, args=(transcript, voice_output_path, output_path))
            thread.start()
            threads.append(thread)
    
        for thread in threads:
            thread.join()
        thread_index = thread_index + max_threads_count

    # s3_urls = [] 
    candidate_array = process_rvc_model(voice_id, bark_voices)

    # for bark_voice in candidate_array:
    #     bucket_name = 'voice-dev1'
    #     audio_name = uuid.uuid4()
    #     object_name = 'tmp_audio/' + str(audio_name) + ".wav"
    #     extra_args = {'ACL': 'public-read'}
    #     session = boto3.Session(
    #         aws_access_key_id=AWS_ACCESS_KEY_ID,
    #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    #     )
    #     s3 = session.client('s3')
    #     s3.upload_file(bark_voice, bucket_name, object_name, ExtraArgs=extra_args)
    #     s3_url = "https://{}.s3.us-east-2.amazonaws.com/{}".format(bucket_name, object_name)
    #     os.remove(bark_voice)
    #     s3_urls.append(s3_url)

    for temp in bark_voices:
        os.remove(temp)

    return candidate_array

def process_rvc_model(voice_id: str, candidate_array):
    result_array = []
    model_path = voice_id + "/" + voice_id + ".pth"
    for bvoice_path in candidate_array:
        # Read audio file and get data and sample rate
        data, sample_rate = sf.read(bvoice_path)
        # Convert audio data to int16 array
        audio_array = (data * 32767).astype(np.int16)
        
        audio_rvc = gen(model_path, 0, ['harvest'], (sample_rate, audio_array), 0.0, 0.88, 3, 0.33, 128, [])
        rvc_uuid = uuid.uuid4()
        output_path = "output/" + str(rvc_uuid) + ".wav"
        write_wav(output_path, audio_rvc[0], audio_rvc[1])
        result_array.append(output_path)
    return result_array

def get_transcript_similarity(file_path, expected_text):
    try:
        with sr.AudioFile(file_path) as source:
            r = sr.Recognizer()
            audio = r.record(source)
        transcript = r.recognize_google(audio)
        similarity_score = 1 - (distance(transcript.lower(), expected_text.lower()) / max(len(transcript), len(expected_text)))
        return similarity_score
    except Exception:
        return 0
        
def get_mean_pitch(file_path):
    y, sr = librosa.load(file_path)
    f0 = librosa.yin(y=y, fmin=50, fmax=2000, sr=sr)
    fund_freq = f0.mean()

    snd = parselmouth.Sound(file_path)
    duration = snd.get_total_duration()
    return fund_freq / duration

def get_prosody_similarity(file_path, reference_path):
    audio1, sr1 = librosa.load(file_path)
    audio2, sr2 = librosa.load(reference_path)
    harmonic1, percussive1 = librosa.effects.hpss(audio1)
    pitch1 = librosa.yin(y=harmonic1, fmin=50, fmax=2000, sr=sr1)

    harmonic2, percussive2 = librosa.effects.hpss(audio2)
    pitch2 = librosa.yin(y=harmonic2, fmin=50, fmax=2000, sr=sr2)

    normalized_pitch_vector_1 = (pitch1 - np.mean(pitch1)) / np.std(pitch1)
    normalized_pitch_vector_2 = (pitch2 - np.mean(pitch2)) / np.std(pitch2)

    distance, path = librosa.sequence.dtw(normalized_pitch_vector_1, normalized_pitch_vector_2)  
    similarity_score = distance / len(path[0])

    return similarity_score.mean()

# def get_mfcc(file_path, reference_path):
#     y1, sr1 = librosa.load(reference_path)
#     y2, sr2 = librosa.load(file_path)
#     mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
#     mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)
#     similarity = cosine_similarity(mfcc1.T, mfcc2.T)
#     return similarity.mean()

def get_audio_duration(file_path):
    data, samplerate = sf.read(file_path)
    duration = len(data) / float(samplerate)
    return duration

def filter_duration_array(audio_files, duration_count):
    sorted_duration_array = sorted(audio_files, key=get_audio_duration)    
    filter_array = []
    for index in range(0, duration_count):    
        filter_array.append(sorted_duration_array[index])
    return filter_array

def filter_transcript_array(audio_files, count, transcript):
    sorted_score_array = sorted(audio_files, key=lambda x:get_transcript_similarity(x, transcript), reverse=True)
    filter_array = []
    for index in range(0, count):    
        filter_array.append(sorted_score_array[index])
    return filter_array

def filter_prosody_array(audio_files, reference_path, transcript):
    # duration_array = filter_duration_array(audio_files, int(2 * len(audio_files) / 3))
    # transcript_array = []
    # top_transcript_similarity = 0
    # for file_path in audio_files:
    #     # if get_mfcc(file_path, reference_path) < 0.9:
    #     #     continue
    #     transcript_similarity = get_transcript_similarity(file_path, transcript)
    #     if top_transcript_similarity < transcript_similarity:
    #         top_transcript_similarity = transcript_similarity
    #     transcript_array.append(TranscriptInfo(file_path, transcript_similarity))

    # sorted_transcript_array = []
    # for info in transcript_array:
    #     # if info.transcript_similarity > top_transcript_similarity - 0.05 or info.transcript_similarity >= 0.89:
    #     if info.transcript_similarity > 0.8:
    #         sorted_transcript_array.append(info.file_path)
    # sorted_score_array = sorted(audio_files, key=lambda x:get_prosody_similarity(x, reference_path))
    sorted_transcript_array = filter_transcript_array(audio_files, int(len(audio_files) / 2), transcript)
    sorted_score_array = sorted(sorted_transcript_array, key=lambda x:get_mean_pitch(x), reverse=True)
    filtered_array = []
    if len(sorted_score_array) > 0:
        filtered_array.append(sorted_score_array[0])
    if len(sorted_score_array) > 1:
        filtered_array.append(sorted_score_array[1])
    if len(sorted_score_array) > 2:
        filtered_array.append(sorted_score_array[2])
    duration_array = filter_duration_array(filtered_array, 1)
    return duration_array

def filter_audio_array(file_array, candidate_count, reference_input, transcript):
    candidate_array = []
    sorted_array = filter_prosody_array(file_array, reference_input, transcript)
    # reference_loudness = get_lufs(reference_input)
    if candidate_count > len(sorted_array):
        candidate_count = len(sorted_array)
    for index in range(0, candidate_count):    
        # candidate_loudness = get_lufs(sorted_array[index])
        # equal_lufs(sorted_array[index], float(reference_loudness) - float(candidate_loudness))
        candidate_array.append(sorted_array[index])

    return candidate_array

def generate(transcript, voice_output_path, output_path):
    # sentences = nltk.sent_tokenize(transcript.replace("\n", " ").strip())
    # print(sentences)
    # silence = np.zeros(int(0.03 * SAMPLE_RATE))  # quarter second of silence
    # pieces = []
    # for sentence in sentences:
    #     audio_array = generate_audio(sentence, history_prompt=voice_output_path)
    #     pieces += [audio_array, silence.copy()]

    # write_wav(output_path, SAMPLE_RATE, np.concatenate(pieces))
    audio_array = generate_audio_new(transcript, history_prompt=voice_output_path)
    write_wav(output_path, SAMPLE_RATE, audio_array)

def get_lufs(file_path):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-i", file_path,
        "-af", "loudnorm=I=-16:dual_mono=true:TP=-1.5:LRA=11:print_format=json",
        "-f", "null",
        "-"
    ]
    output = subprocess.run(command, capture_output=True, text=True)
    loudness_start_idx = output.stderr.find("{")
    loudness_end_idx = output.stderr.find("}") + 1
    loudness_json = output.stderr[loudness_start_idx:loudness_end_idx].strip()
    loudnorm_data = json.loads(loudness_json)
    loudness = loudnorm_data['input_i']
    return loudness

def equal_lufs(file_path, volume):
    output_path = file_path + ".wav"
    command = [
        "ffmpeg",
        "-i", file_path,
        "-af", "volume=" + str(volume) + "dB",
        "-c:v", "copy",
        output_path
    ]
    subprocess.run(command, capture_output=True, text=True)
    os.remove(file_path)
    os.rename(output_path, file_path)


def cleanvoice(s3_url: str):
    # upload audio to cleanvoice
    cleanvoice_headers = {"Content-Type": "application/json", "X-API-Key": CLEAN_VOICE_KEY}
    data  = {
                "input": {
                    "files": [s3_url],
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

def process_generate_voice(voice_id: str, transcript: str, prefix: str):
    if prefix == "music":
        transcript = "♪ " + transcript + " ♪"
    audio_array = generate_audio(transcript)
    
    output_path = "output/" + str(uuid.uuid4()) + ".wav"
    write_wav(output_path, SAMPLE_RATE, audio_array)

    voices = []
    voices.append(output_path)
    # converted_array = process_rvc_model(voice_id, voices)

    bucket_name = 'voice-dev1'
    object_name = 'tmp_audio/' + str(uuid.uuid4()) + ".wav"
    extra_args = {'ACL': 'public-read'}
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    s3 = session.client('s3')
    s3.upload_file(output_path, bucket_name, object_name, ExtraArgs=extra_args)
    s3_url = "https://{}.s3.us-east-2.amazonaws.com/{}".format(bucket_name, object_name)
    os.remove(output_path)
    # os.remove(converted_array[0])
    return {"status" : "success", "voice_id" : voice_id, "voice_url" : s3_url}


def download_s3_file(voice_region: str, 
                    voice_bucket: str, 
                    voice_key: str,
                    local_path: str):
    try:
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        s3 = session.client('s3', region_name=voice_region)
        s3.download_file(voice_bucket, voice_key, local_path)
        with open(local_path, 'rb') as f:
            f.flush()
        return local_path
    except Exception:
        return ""

def get_bucket_name(s3_url):
    parsed_url = urlparse(s3_url)
    bucket_name = parsed_url.netloc.split(".")[0]
    return bucket_name

def download_s3_url(url: str,
                    local_path: str):
    try:
        parsed_url = url.split('/')
        bucket_name = get_bucket_name(url)
        object_key = '/'.join(parsed_url[3:])

        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        s3 = session.client('s3')
        s3.download_file(bucket_name, object_key, local_path)
        with open(local_path, 'rb') as f:
            f.flush()
        return local_path
    except Exception:
        return ""

def download_file_url(url: str,
                    local_path: str):
    try:
        response = requests.get(url)
        with open(local_path, 'wb') as file:
            file.write(response.content)
            file.flush()
        return local_path
    except Exception:
        return ""