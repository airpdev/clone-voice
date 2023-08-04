import soundfile as sf
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

from transformers import BertTokenizer
from encodec.utils import convert_audio
from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer
from bark.api import generate_audio
from bark.generation import load_codec_model, SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic
from scipy.io.wavfile import write as write_wav
from fastapi import FastAPI, Query, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from hubert.hubert_manager import HuBERTManager

access_key = os.getenv("AWS_ACCESS_KEY_ID", default=None)
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", default=None)
cleanvoice_key = os.getenv("CLEAN_VOICE_KEY", default=None)

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
preload_models()
model = load_codec_model(use_gpu=torch.cuda.is_available())
cleanvoice_url = "https://api.cleanvoice.ai/v1/edits"
lock = asyncio.Lock()

@app.get("/")
def root():
    return time.time()

@app.get("/generate_voice")
@limiter.limit("50/minute")
async def generate_voice(request: Request, transcript: str = Query(None), denoised: bool = Query(False)):
    loop = asyncio.get_event_loop()
    s3_url, voice_id = await loop.run_in_executor(None, process_generate_voice, transcript, access_key, secret_key)
    if denoised == False:
        return {"voice_id" : voice_id, "voice_url" : s3_url, "voice_clean_data" : ""}
    voice_response = await loop.run_in_executor(None, cleanvoice, s3_url, cleanvoice_key)
    return {"voice_id" : voice_id, "voice_url" : s3_url, "voice_clean_data" : voice_response}
    
@app.post("/generate_voice")
@limiter.limit("50/minute")
async def generate_voice(request: Request, transcript: str = Query(None), denoised: bool = Query(False)):
    loop = asyncio.get_event_loop()
    s3_url, voice_id = await loop.run_in_executor(None, process_generate_voice, transcript, access_key, secret_key)
    if denoised == False:
        return {"voice_id" : voice_id, "voice_url" : s3_url, "voice_clean_data" : ""}
    voice_response = await loop.run_in_executor(None, cleanvoice, s3_url, cleanvoice_key)
    return {"voice_id" : voice_id, "voice_url" : s3_url, "voice_clean_data" : voice_response}

def cleanvoice(s3_url: str, cleanvoice_key : str):
    # upload audio to cleanvoice
    cleanvoice_headers = {"Content-Type": "application/json", "X-API-Key": cleanvoice_key}
    data  = {
                "input": {
                    "files": [s3_url ],
                    "config": {}
                }
            }
    response = requests.post(cleanvoice_url, headers=cleanvoice_headers, json=data)
    response_json = response.json()
    
    cleanvoice_get_url = cleanvoice_url + "/" + response_json["id"]
    response_json = get_voice(cleanvoice_get_url, cleanvoice_key)

    if response_json["status"] == "SUCCESS":
        return {"status" : response_json["status"], "url" : response_json["url"]}
    else:
        return {"status" : response_json["status"], "url" : ""}

# get denoised audio from cleanvoice
def get_voice(url: str, cleanvoice_key : str):
    cleanvoice_headers = {"Content-Type": "application/json", "X-API-Key": cleanvoice_key}
    response = requests.get(url, headers=cleanvoice_headers)
    response_json = response.json()
    if response_json["status"] == "SUCCESS":
        return {"status" : response_json["status"], "url" : response_json["result"]["download_url"]}
    if response_json["status"] == "FAILURE":
        return {"status" : response_json["status"], "url" : ""}
    
    # delay 3 seconds
    time.sleep(3)
    return get_voice(url, cleanvoice_key)

def process_generate_voice(transcript: str, access_key: str, secret_key : str):
    voice_id = uuid.uuid4()
    full_generation, audio_array = generate_audio(transcript, output_full=True)
    
    output_path = "output/" + str(voice_id) + ".wav"
    write_wav(output_path, SAMPLE_RATE, audio_array)
    
    # getting score
    # score_value = {"STOI": "",  "PESQ": "", "SI-SDR": ""}
    # if scored:
    #     wav, sr = torchaudio.load(output_path)
    #     wav = torchaudio.functional.resample(wav, sr, bundle.sample_rate)
    #     scores = squim_model(wav)
    #     score_value = {"STOI": scores[0].item(),  "PESQ": scores[1].item(), "SI-SDR": scores[2].item()}

    # saving .npz file
    npz_path = 'bark/assets/prompts/' + str(voice_id) + '.npz'
    np.savez(npz_path, fine_prompt=full_generation["fine_prompt"], coarse_prompt=full_generation["coarse_prompt"], semantic_prompt=full_generation["semantic_prompt"])

    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    s3 = session.client('s3')
    bucket_name = 'tmp-dev-283501'
    object_name = 'bvoice_audio/' + str(voice_id) + ".wav"
    extra_args = {'ACL': 'public-read'}
    s3.upload_file(output_path, bucket_name, object_name, ExtraArgs=extra_args)
    s3_url = "https://{}.s3.us-east-2.amazonaws.com/{}".format(bucket_name, object_name)
    os.remove(output_path)
    return s3_url, voice_id


@app.get("/clone_voice")
@limiter.limit("50/minute")
async def clone_voice(request: Request, 
                        voice_id: str = Query(None), 
                        reference_region: str = Query(None), 
                        reference_bucket: str = Query(None), 
                        reference_key: str = Query(None), 
                        transcript: str = Query(None),
                        dataset_region: str = Query(None), 
                        dataset_bucket: str = Query(None), 
                        dataset_key: str = Query(None), 
                        denoised: bool = Query(False)):
    loop = asyncio.get_event_loop()
    await lock.acquire()
    try:
        s3_url = await loop.run_in_executor(None, process_clone_voice, voice_id, reference_region, reference_bucket, reference_key, transcript, dataset_region, dataset_bucket, dataset_key, access_key, secret_key)
    finally:
        lock.release()
    if denoised == False:
        return {"voice_id" : voice_id, "voice_url" : s3_url, "voice_clean_data" : ""}
    response = await loop.run_in_executor(None, cleanvoice, s3_url, cleanvoice_key)
    return {"voice_id" : voice_id, "voice_url" : s3_url, "voice_clean_data" : response}


@app.post("/clone_voice")
@limiter.limit("50/minute")
async def clone_voice(request: Request, 
                        voice_id: str = Query(None), 
                        reference_region: str = Query(None), 
                        reference_bucket: str = Query(None), 
                        reference_key: str = Query(None), 
                        transcript: str = Query(None),
                        dataset_region: str = Query(None), 
                        dataset_bucket: str = Query(None), 
                        dataset_key: str = Query(None), 
                        denoised: bool = Query(False)):
    loop = asyncio.get_event_loop()
    await lock.acquire()
    try:
        s3_url = await loop.run_in_executor(None, process_clone_voice, voice_id, reference_region, reference_bucket, reference_key, transcript, dataset_region, dataset_bucket, dataset_key, access_key, secret_key)
    finally:
        lock.release()

    if denoised == False:
        return {"voice_id" : voice_id, "voice_url" : s3_url, "voice_clean_data" : ""}
    response = await loop.run_in_executor(None, cleanvoice, s3_url, cleanvoice_key)
    return {"voice_id" : voice_id, "voice_url" : s3_url, "voice_clean_data" : response}

def process_clone_voice(voice_id: str, reference_region: str, reference_bucket: str, reference_key: str, transcript: str, dataset_region: str, dataset_bucket: str, dataset_key: str, access_key: str, secret_key : str):
    voice_output_path = 'bark/assets/prompts/' + voice_id + '.npz'
    current_directory = os.path.abspath(os.getcwd())
    dataset_path = current_directory + "/dataset/" + voice_id + "/"
    if os.path.exists(dataset_path) == False:
        os.mkdir(dataset_path)
    if os.path.exists(voice_output_path) == False:
        reference_name = uuid.uuid4()
        reference_intput = "output/" + str(reference_name) + ".wav"
        reference_output = reference_intput + ".wav"
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        s3 = session.client('s3', region_name=reference_region)
        s3.download_file(reference_bucket, reference_key, reference_intput)
        with open(reference_intput, 'rb') as f:
            f.flush()
        
        # download dataset related with voice-id
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        s3 = session.client('s3', region_name=dataset_region)
        objects = s3.list_objects_v2(Bucket=dataset_bucket, Prefix=dataset_key)
        for obj in objects['Contents']:
            key = obj['Key']
            filename = key.split("/")[-1]
            if ".wav" in filename:
                s3.download_file(dataset_bucket, key, dataset_path + filename)

        ######## getting semantic_tokens #######################################
        stream = ffmpeg.input(reference_intput)
        stream = ffmpeg.output(stream, reference_output, acodec='pcm_s16le', ar=model.sample_rate, ac=model.channels)
        ffmpeg.run(stream)
        
        wav, sr = torchaudio.load(reference_output)
        # if wav.shape[0] == 2:  # Stereo to mono if needed
        #     wav = wav.mean(0, keepdim=True)
        semantic_vectors = hubert_model.forward(wav, input_sample_hz=sr)
        semantic_tokens = tokenizer.get_token(semantic_vectors)

        ######### getting npz ##################################################
        # wav, sr = torchaudio.load(reference_output)
        # wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        wav = wav.unsqueeze(0).to(device)
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = model.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]
        # move codes to cpu
        codes = codes.cpu().numpy()
        np.savez(voice_output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)
        os.remove(reference_intput)
        os.remove(reference_output)

    audio_array = generate_audio(transcript, history_prompt=voice_output_path)
    # generation with more control
    # x_semantic = generate_text_semantic(
    #     transcript,
    #     history_prompt=voice_output_path,
    #     temp=0.7,
    #     top_k=50,
    #     top_p=0.95,
    # )

    # x_coarse_gen = generate_coarse(
    #     x_semantic,
    #     history_prompt=voice_output_path,
    #     temp=0.7,
    #     top_k=50,
    #     top_p=0.95,
    # )
    # x_fine_gen = generate_fine(
    #     x_coarse_gen,
    #     history_prompt=voice_output_path,
    #     temp=0.5,
    # )
    # audio_array = codec_decode(x_fine_gen)

    voice_name = uuid.uuid4()
    output_path = "output/" + str(voice_name) + ".wav"
    write_wav(output_path, SAMPLE_RATE, audio_array)

    rvc_output_path = process_rvc_model(voice_id, output_path, dataset_path)
    os.remove(output_path)

    bucket_name = 'tmp-dev-283501'
    object_name = 'bvoice_audio/' + str(voice_name) + ".wav"
    extra_args = {'ACL': 'public-read'}
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    s3 = session.client('s3')
    s3.upload_file(rvc_output_path, bucket_name, object_name, ExtraArgs=extra_args)
    s3_url = "https://{}.s3.us-east-2.amazonaws.com/{}".format(bucket_name, object_name)
    os.remove(rvc_output_path)
       
    return s3_url

def process_rvc_model(voice_id: str, bvoice_path: str, dataset_path: str):
    import webui.ui.tabs.training.training.rvc_workspace as rvc_ws
    from webui.ui.tabs.rvc import gen

    model_path = voice_id + "/" + voice_id + ".pth"
    if os.path.exists("./data/models/rvc/" + model_path) == False:
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

    # Read audio file and get data and sample rate
    data, sample_rate = sf.read(bvoice_path)
    # Convert audio data to int16 array
    audio_array = (data * 32767).astype(np.int16)
    
    audio_rvc = gen(model_path, 0, ['harvest'], (sample_rate, audio_array), 0.0, 0.88, 3, 0.33, 128, [])
    rvc_uuid = uuid.uuid4()
    output_path = "output/" + str(rvc_uuid) + ".wav"
    write_wav(output_path, audio_rvc[0], audio_rvc[1])

    return output_path







