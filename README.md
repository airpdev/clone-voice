# 🐶Bhuman

<br>
<p align="center">
<img src="https://user-images.githubusercontent.com/5068315/235310676-a4b3b511-90ec-4edf-8153-7ccf14905d73.png" width="500"></img>
</p>
<br>


## 🚀  APIs

- Add voice with referencen and dataset
  * POST: https://bvoice.dev.bhuman.ai/add_voice
  * Params: voice_id, bark_ref_region, bark_ref_bucket, bark_ref_key, dataset_region, dataset_bucket, dataset_key
  * Response: API returns status result with voice id.
  * Description: Generate .npz file and train rvc model

- Clone voice from reference audio and transcript
  * GET/POST: https://bvoice.dev.bhuman.ai/clone_voice
  * Params: voice_id, reference_region, reference_bucket, reference_key, transcript, total_try_count
  * Response: API returns s3 bucket urls for cloned voices.
  * Description: Clone voice from text and reference audio(s3 url) using model and passing it to rvc model.

- Select the best prosody
  * POST: https://bvoice.dev.bhuman.ai/prosody_select
  * Params: voice_id, reference_region, reference_bucket, reference_key, transcript, candidate_count, urls
  * Response: API returns the selected urls.
  * Description: Select the closest audios (candidate count) with reference audio.


## 🚀 Generating/Cloning voice

* You need to create the account in elevenlabs.ai.

- Using docker
  * sudo docker build -t clone-voice .
  * sudo docker run --gpus all -d --rm -p 8080:8080 clone-voice  

- Using uvicorn
  * uvicorn test_clean:app --host 0.0.0.0 --port 8080

- APIs
  * Generating speech with narrator's voice and the given text:
    (Post) http://34.45.132.232:8080/clone_voice
    Input: {
             "transcript": "No one would have believed. In the last years of the nineteenth century"
           }
  * Voice conversion of singer's song
    Command: python test_clone.py