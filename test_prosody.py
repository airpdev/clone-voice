import parselmouth
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import boto3
import matplotlib.pyplot as plt
import subprocess
import json
import os
import gc
import soundfile as sf
import speech_recognition as sr
from Levenshtein import distance
import librosa
import numpy as np
import uuid
import time
import requests
import ffmpeg


def get_audio_duration(file_path):
    data, samplerate = sf.read(file_path)
    duration = len(data) / float(samplerate)
    return duration

# def get_audio_pesq(file_path, reference_path):
#     ref_audio, fs = sf.read(reference_path)
#     degraded_audio, _ = sf.read(file_path)
#     pesq_value = pesq(16000, ref_audio, degraded_audio, 'wb')
#     return pesq_value

def equal_lufs(file_path, volume):
    output_path = file_path + ".wav"
    command = [
        "ffmpeg",
        "-i", file_path,
        "-af", "volume=" + str(volume) + "dB",
        "-c:v", "copy",
        output_path
    ]
    print(command)

    subprocess.run(command, capture_output=True, text=True)
    # os.remove(file_path)
    # os.rename(output_path, file_path)

def get_candidate_array(file_array, candidate_count, duration_count, reference_output):
    sorted_duration_array = sorted(file_array, key=get_audio_duration)    
    filter_array = []
    for index in range(0, duration_count):    
        filter_array.append(sorted_duration_array[index])
    
    if candidate_count > duration_count:
        candidate_count = duration_count

    candidate_array = []
    # sorted_array = sorted(filter_array, key=lambda x:get_audio_pesq(x, reference_output), reverse=True)
    # reference_loudness = get_lufs(reference_output)
    for index in range(0, candidate_count):    
        # candidate_loudness = get_lufs(sorted_array[index])
        # equal_lufs(sorted_array[index], float(reference_loudness) - float(candidate_loudness))
        candidate_array.append(filter_array[index])

    return candidate_array
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

from sklearn.metrics.pairwise import cosine_similarity
def get_mfcc(file_path, reference_path):
    y1, sr1 = librosa.load(reference_path)
    y2, sr2 = librosa.load(file_path)
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)
    similarity = cosine_similarity(mfcc1.T, mfcc2.T)
    return similarity.mean()

from pydub import AudioSegment
def extract_prosodic_features(audio_path):
    y, sr = librosa.load(audio_path)
    # Extract prosodic features (e.g., pitch, energy, etc.) using librosa functions
    pitch = librosa.yin(y=y, fmin=50, fmax=2000, sr=sr)
    energy = librosa.feature.rms(y=y)[0]
    # Calculate statistics of the extracted features
    pitch_mean = np.mean(pitch)
    energy_mean = np.mean(energy)
    
    return pitch_mean, energy_mean

def calculate_similarity(audio1_pitch, audio1_energy, audio2_pitch, audio2_energy):
    # Calculate the similarity using desired metric (e.g., Euclidean distance)
    similarity = np.sqrt((audio1_pitch - audio2_pitch)**2 + (audio1_energy - audio2_energy)**2)
    return similarity

# def get_prosody_similarity(file_path, reference_path):
#     audio1_pitch, audio1_energy = extract_prosodic_features(file_path)
#     audio2_pitch, audio2_energy = extract_prosodic_features(reference_path)
#     return calculate_similarity(audio1_pitch, audio1_energy, audio2_pitch, audio2_energy)
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


# def get_prosody_similarity2(file_path, reference_path):
#     audio1, sr1 = librosa.load(file_path)
#     audio2, sr2 = librosa.load(reference_path)
#     pitch1 = librosa.yin(y=audio1, fmin=50, fmax=2000, sr=sr1)
#     pitch2 = librosa.yin(y=audio2, fmin=50, fmax=2000, sr=sr2)

#     normalized_pitch_vector_1 = (pitch1 - np.mean(pitch1)) / np.std(pitch1)
#     normalized_pitch_vector_2 = (pitch2 - np.mean(pitch2)) / np.std(pitch2)

#     distance, path = librosa.sequence.dtw(normalized_pitch_vector_1, normalized_pitch_vector_2)  
#     similarity_score = distance / len(path[0])

#     # Calculate the similarity between the two pitch vectors
#     # similarity = librosa.sequence.dtw(X=pitch_mean1.T, Y=pitch_mean2.T)[0]
#     return similarity_score.mean()

class TranscriptInfo:
    def __init__(self, file_path, transcript_similarity):
        self.file_path = file_path
        self.transcript_similarity = transcript_similarity

from urllib.parse import urlparse
def get_bucket_name(s3_url):
    parsed_url = urlparse(s3_url)
    bucket_name = parsed_url.netloc.split(".")[0]
    return bucket_name

# urls = ["https://fra1.digitaloceanspaces.com/cleanvoice/uploads/converted_F87ZvqF_46dff255-cfd1-4183-9217-13f81feacc3f_enhanced.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=TNTLOTODENTEDRISPGU5%2F20230831%2Ffra1%2Fs3%2Faws4_request&X-Amz-Date=20230831T133534Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=a1ffd910e3bce14b502e1f7ec410b08963af723439e2a3a94ebbc74d9a9038f5",
#         "https://fra1.digitaloceanspaces.com/cleanvoice/uploads/converted_nmq2C4F_8aa0db41-e05e-48c2-92aa-e2f9802a4632_enhanced.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=TNTLOTODENTEDRISPGU5%2F20230831%2Ffra1%2Fs3%2Faws4_request&X-Amz-Date=20230831T133543Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=ae6c5cd8b8fc53667f8303ac2722ba89a5ff46492096eb217e20aea39899eb59"]
# audio_files = []
# for url in urls:
#     tmp_path = "output/" + str(uuid.uuid4()) + ".wav"
#     tmp_path = download_file_url(url, tmp_path)
#     if len(tmp_path) > 0:
#         audio_files.append(tmp_path)
# # audio_files.append("11labs/reference.wav")
# current_time = int(time.time())
# print(audio_files)

# transcript_array = []
# top_transcript_similarity = 0
# for file_path in audio_files:
#     # if get_mfcc(file_path, "test/reference.wav") < 0.9:
#     #     continue
#     # print("\n" + file_path)
#     # print(get_mfcc(file_path, "test2/reference.wav"))
#     transcript_similarity = get_transcript_similarity(file_path, "hey James")
#     if top_transcript_similarity < transcript_similarity:
#         top_transcript_similarity = transcript_similarity
#     transcript_array.append(TranscriptInfo(file_path, transcript_similarity))


# sorted_transcript_array = []
# for info in transcript_array:
#     if info.transcript_similarity > top_transcript_similarity - 0.05 or info.transcript_similarity >= 0:
#         sorted_transcript_array.append(info.file_path)
# print(sorted_transcript_array)


def get_prosody_similarity1(file_path, reference_path):
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

def select_prosody(audio_files, reference_path):
    audio_ref, sr_ref = librosa.load(reference_path)
    harmonic_ref, percussive_ref = librosa.effects.hpss(audio_ref)
    pitch_ref = librosa.yin(y=harmonic_ref, fmin=50, fmax=2000, sr=sr_ref)
    normalized_pitch_vector_ref = (pitch_ref - np.mean(pitch_ref)) / np.std(pitch_ref)

    best_similarity = 0.0
    selected_path = ""
    for file_path in audio_files:
        audio, sr = librosa.load(file_path)
        harmonic, percussive = librosa.effects.hpss(audio)
        pitch = librosa.yin(y=harmonic, fmin=50, fmax=2000, sr=sr)
        normalized_pitch_vector = (pitch - np.mean(pitch)) / np.std(pitch)
        distance, path = librosa.sequence.dtw(normalized_pitch_vector, normalized_pitch_vector_ref)  
        similarity_score = distance / len(path[0])
        similarity = similarity_score.mean()
        if best_similarity == 0.0 or best_similarity > similarity:
            best_similarity = similarity
            selected_path = file_path
       
    return selected_path, best_similarity

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
    # os.remove(file_path)
    # os.rename(output_path, file_path)

def get_reference(file_path, voice_id):
    audio_duration = get_audio_duration(file_path)
    if audio_duration < 13.0:
        return None
    reference_path = "output/" + voice_id + "_ref.wav"    
    command = [
        "ffmpeg",
        "-i", file_path,
        "-ss", str(0.0),
        "-to", str(13.0),
        "-vn",
        "-c:a", "pcm_s16le",
        "-ar", str(24000),
        "-ac", str(1),
        reference_path
    ]
    subprocess.run(command)
    return reference_path

from pydub import AudioSegment
from pydub.silence import split_on_silence
def remove_silence(audio, silence_threshold=-50, min_silence_duration=300):
    non_silent_audio = AudioSegment.empty()
    chunks = split_on_silence(audio, silence_thresh=silence_threshold, min_silence_len=min_silence_duration)
    for chunk in chunks:
        non_silent_audio += chunk
    return non_silent_audio

# audio = AudioSegment.from_file("references/dataset.wav")
# modified_audio = remove_silence(audio)
# modified_audio.export("references/dataset_desilenced.wav", format="wav")
# reference_path = get_reference("references/dataset_desilenced.wav", "123")


audio_files = []
for index in range(1, 16):            
    audio_files.append("output/" + str(index) + ".wav")

# current_time = int(time.time())
# # selected_path, best_similarity = select_prosody(audio_files, "test/voice4/reference.wav")
# # print(selected_path, best_similarity)
# sorted_score_array = sorted(audio_files, key=lambda x:get_prosody_similarity1(x, "test/voice4/reference.wav"))
# print(sorted_score_array)
# print(time.time() - current_time)
for file_path in audio_files:
    print("\n" + file_path)

#     bucket_name = 'voice-dev1'
#     audio_name = uuid.uuid4()
#     object_name = 'bark_audio/' + str(audio_name) + ".wav"
#     extra_args = {'ACL': 'public-read'}
#     session = boto3.Session(
#         aws_access_key_id="AKIATBIHACXJDWFQQCNH",
#         aws_secret_access_key="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"
#     )
#     s3 = session.client('s3')
#     s3.upload_file(file_path, bucket_name, object_name, ExtraArgs=extra_args)
#     s3_url = "https://{}.s3.us-east-2.amazonaws.com/{}".format(bucket_name, object_name)
#     print(s3_url)
    quality = get_mean_pitch(file_path)
    print("Prosody Quality:", quality)

    # print("mfcc:", get_mfcc(file_path, "11labs/reference.wav"))
    # print("prosody_similarity:", get_prosody_similarity(file_path, "11labs/reference.wav"))
    # print("prosody_similarity1:", get_prosody_similarity1(file_path, "test/voice5/reference.wav"))
    # print("prosody_similarity2:", get_prosody_similarity2(file_path, "11labs/reference.wav"))
    print("transcript_similarity:", get_transcript_similarity(file_path, "You can record a short video on our platform using your webcam or phone in any language.You can also import a spreadsheet, integrate with over 5,000 apps, or connect via API to connect your data.Finally, with just one click, you can generate thousands of personalized videos that look real and send them over email, SMS, Linkedin, or Zapier. Ready to get started?"))
    # target_file = 'test/reference.wav'
    # source_signal, source_sr = librosa.load(file_path)
    # target_signal, target_sr = librosa.load(target_file)

    # source_pitch = librosa.yin(source_signal, fmin=50, fmax=2000, sr=source_sr)
    # target_pitch = librosa.yin(target_signal, fmin=50, fmax=2000, sr=target_sr)

    # pitch_matched_signal = librosa.effects.pitch_shift(source_signal, sr=source_sr, n_steps=target_pitch.mean() - source_pitch.mean())

    # output_file = file_path + ".wav"
    # sf.write(output_file, pitch_matched_signal, source_sr)

    # y, sr = librosa.load(file_path)
    # f0 = librosa.yin(y, fmin=50, fmax=2000, sr=sr)
    # fund_freq = f0.mean()
    # print("Pitch frequency: {:.2f} Hz".format(fund_freq))

    # y, sr  = librosa.load(file_path)
    # f0 = librosa.yin(y, fmin=50, fmax=2000, sr=sr)
    # plt.plot(f0)
    # plt.xlabel('Time')
    # plt.ylabel('F0')
    # plt.title('F0 Curve')
    # plt.savefig('2-25.png')

    # tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
    # print(file_path, "Tempo", tempo)

    # sound = parselmouth.Sound(file_path)
    # pitch = sound.to_pitch()
    # print(file_path, "pitch", pitch)

    # intensity = sound.to_intensity()
    # #print(file_path, "intensity", intensity)

    # y_emphasized = librosa.effects.preemphasis(audio_data)
    # print(file_path, "y_emphasized", len(y_emphasized))

    # command = [
    #     "ffmpeg",
    #     "-hide_banner",
    #     "-nostdin",
    #     "-i", file_path,
    #     "-af", "loudnorm=I=-16:dual_mono=true:TP=-1.5:LRA=11:print_format=json",
    #     "-f", "null",
    #     "-"
    # ]
    # output = subprocess.run(command, capture_output=True, text=True)
    # loudness_start_idx = output.stderr.find("{")
    # loudness_end_idx = output.stderr.find("}") + 1
    # loudness_json = output.stderr[loudness_start_idx:loudness_end_idx].strip()
    # loudnorm_data = json.loads(loudness_json)
    # loudness = loudnorm_data['input_i']
    # print(file_path, "Loudness", loudness)

    # equal_lufs(file_path, 10)
    # try:
    #     with sr.AudioFile(file_path) as source:
    #         r = sr.Recognizer()
    #         audio = r.record(source)
    #     transcript = r.recognize_google(audio)
    #     expected_text = "to master the world of anthropology"
    #     # expected_text = "hey James."
    #     similarity_score = 1 - (distance(transcript.lower(), expected_text.lower()) / max(len(transcript), len(expected_text)))
    #     print("trascript score:", similarity_score, transcript)
    # except Exception:
    #     print("parsing error")

    # ref_audio, fs = sf.read('test2/reference.wav')
    # degraded_audio, _ = sf.read(file_path)
    # pesq_value = pesq(16000, ref_audio, degraded_audio, 'wb')
    # print(file_path, "pesq_value", pesq_value)

    # mean = np.mean(audio_data)
    # rms = np.sqrt(np.mean(np.square(audio_data)))
    # print(file_path, "Mean:", mean)
    # print(file_path, "RMS:", rms)

    # mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate)
    # mfccs = mfccs.T
    # print(file_path, "mfccs", mfccs)

    # pitches = librosa.yin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    # pitch_range = max(pitches) - min(pitches)
    # print(file_path, "Pitch Range:", pitch_range)

