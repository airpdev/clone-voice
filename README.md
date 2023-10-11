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


