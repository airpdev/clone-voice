import numpy as np
import time
import torchaudio
import torch
import ffmpeg
import os
import soundfile as sf
import asyncio
import requests
import json
import uuid
import rvc.engine.rvc_workspace as rvc_ws
import subprocess
from rvc.engine.rvc import gen

from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer
from bark.api import generate_audio
from bark.generation import load_codec_model, SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic
from bark.api import semantic_to_waveform
from scipy.io.wavfile import write as write_wav
from hubert.hubert_manager import HuBERTManager
import nltk
nltk.download('punkt')

script = "Hey, have you heard about this new text-to-audio model called 'Bark'? Apparently, it's the most realistic and natural-sounding text-to-audio model out there right now.".replace("\n", " ").strip()
# script = "I was doing some research on players in marketing.".replace("\n", " ").strip()

preload_models(text_use_gpu=True, text_use_small=False, coarse_use_gpu=True, coarse_use_small=False, fine_use_gpu=True, fine_use_small=False, codec_use_gpu=True, force_reload=False)
encode_model = load_codec_model(use_gpu=True)

device = 'cuda'
hubert_model = HuBERTManager()
hubert_model.make_sure_hubert_installed()
hubert_model.make_sure_tokenizer_installed()
hubert_model = CustomHubert(checkpoint_path='data/models/hubert/hubert.pt')
tokenizer = CustomTokenizer.load_from_checkpoint('data/models/hubert/tokenizer_large.pth', map_location=torch.device(device))

def train_rvc_model(voice_id: str, dataset_path : str):
    current_directory = os.path.abspath(os.getcwd())
    dataset_path = current_directory + dataset_path
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
        return 1
    except Exception as e:
        print(e)
        return 0

def process_generate_voice(voice_id:str, transcript: str, reference_path: str):
    voice_output_path = 'bark/assets/prompts/v2/' + voice_id + '.npz'
    if os.path.exists(voice_output_path) == False:
        reference_output = reference_path + ".wav"
        ######## getting semantic_tokens #######################################
        stream = ffmpeg.input(reference_path)
        stream = ffmpeg.output(stream, reference_output, acodec='pcm_s16le', ar=encode_model.sample_rate, ac=encode_model.channels)
        ffmpeg.run(stream)
        
        wav, sr = torchaudio.load(reference_output)
        semantic_vectors = hubert_model.forward(wav, input_sample_hz=sr)
        semantic_tokens = tokenizer.get_token(semantic_vectors)

        ######### getting npz ##################################################
        wav = wav.unsqueeze(0).to(device)
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = encode_model.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]
        # move codes to cpu
        codes = codes.cpu().numpy()
        np.savez(voice_output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)
        os.remove(reference_output)

    GEN_TEMP = 0.6
    sentences = nltk.sent_tokenize(transcript)
    print(sentences)
    silence = np.zeros(int(0.2 * SAMPLE_RATE))  # quarter second of silence

    pieces = []
    for sentence in sentences:
        semantic_tokens = generate_text_semantic(
            sentence,
            history_prompt=voice_output_path,
            temp=GEN_TEMP,
            min_eos_p=0.05,  # this controls how likely the generation is to end
        )
        audio_array = semantic_to_waveform(semantic_tokens, history_prompt=voice_output_path)
        # audio_array = generate_audio(sentence)
        pieces += [audio_array, silence.copy()]
    
    output_path = "output/" + str(uuid.uuid4()) + ".wav"
    write_wav(output_path, SAMPLE_RATE, np.concatenate(pieces))

    return output_path

def process_rvc_model(voice_id: str, bvoice_path):
    model_path = voice_id + "/" + voice_id + ".pth"
    # Read audio file and get data and sample rate
    data, sample_rate = sf.read(bvoice_path)
    # Convert audio data to int16 array
    audio_array = (data * 32767).astype(np.int16)
    
    audio_rvc = gen(model_path, 0, ['harvest'], (sample_rate, audio_array), 0.0, 0.88, 3, 0.33, 128, [])

    rvc_uuid = uuid.uuid4()
    output_path = "output/" + str(rvc_uuid) + ".wav"
    write_wav(output_path, audio_rvc[0], audio_rvc[1])

    return output_path

def get_reference(file_path, voice_id):
    reference_path = "output/" + voice_id + "_ref.wav"    
    command = [
        "ffmpeg",
        "-i", file_path,
        "-ss", str(0.0),
        "-to", str(13.0),
        "-vn",
        "-c:a", "pcm_s16le",
        "-ar", str(encode_model.sample_rate),
        "-ac", str(encode_model.channels),
        reference_path
    ]
    subprocess.run(command)
    return reference_path
# bark_generated_path = process_generate_voice("voice4", script, "references/reference.wav")
# print("bark:", bark_generated_path)

print("started training")
status = train_rvc_model("voice", "/dataset/")
print("training:", status)

output_path = process_rvc_model("voice", "resources/input1.wav")
print("rvc:", output_path)

# print("ref:", get_reference("references/dataset.wav", "voice1"))