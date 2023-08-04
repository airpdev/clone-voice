# 🐶Bhuman

<br>
<p align="center">
<img src="https://user-images.githubusercontent.com/5068315/235310676-a4b3b511-90ec-4edf-8153-7ccf14905d73.png" width="500"></img>
</p>
<br>


## 🚀  APIs

- Generate voice from transcript
  * GET/POST: https://bvoice.dev.bhuman.ai/generate_voice
  * Params: transcript, denoised
  * Response: API returns s3 bucket url for generated voice.
  * Description: Generate voice from text (transcript field) using model and upload it to s3 bucket.

- Clone voice from reference audio and transcript
  * GET/POST: https://bvoice.dev.bhuman.ai/clone_voice
  * Params: voice_id, reference_region, reference_bucket, reference_key, transcript, dataset_region, dataset_bucket, dataset_key, denoised
  * Response: API returns s3 bucket url for cloned voice.
  * Description: Clone voice from text and reference audio(s3 url) using model and passing it to rvc model.
