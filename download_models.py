import numpy
import torch
import torchaudio

from hubert.hubert_manager import HuBERTManager
from torchaudio.prototype.pipelines import SQUIM_OBJECTIVE as bundle
from bark.generation import load_codec_model, preload_models
from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer

hubert_model = HuBERTManager()
hubert_model.make_sure_hubert_installed()
hubert_model.make_sure_tokenizer_installed()
preload_models()
model = load_codec_model(use_gpu=torch.cuda.is_available())
squim_model = bundle.get_model()
