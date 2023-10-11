import os
import uuid
import torch.cuda
import torchaudio

flag_strings = ['denoise', 'denoise output', 'separate background']

def flatten_audio(audio_tensor: torch.Tensor | tuple[torch.Tensor, int] | tuple[int, torch.Tensor], add_batch=True):
    if isinstance(audio_tensor, tuple):
        if isinstance(audio_tensor[0], int):
            return audio_tensor[0], flatten_audio(audio_tensor[1])
        elif torch.is_tensor(audio_tensor[0]):
            return flatten_audio(audio_tensor[0]), audio_tensor[1]
    if audio_tensor.dtype == torch.int16:
        audio_tensor = audio_tensor.float() / 32767.0
    if audio_tensor.dtype == torch.int32:
        audio_tensor = audio_tensor.float() / 2147483647.0
    if len(audio_tensor.shape) == 2:
        if audio_tensor.shape[0] == 2:
            # audio_tensor = audio_tensor[0, :].div(2).add(audio_tensor[1, :].div(2))
            audio_tensor = audio_tensor.mean(0)
        elif audio_tensor.shape[1] == 2:
            # audio_tensor = audio_tensor[:, 0].div(2).add(audio_tensor[:, 1].div(2))
            audio_tensor = audio_tensor.mean(1)
        audio_tensor = audio_tensor.flatten()
    if add_batch:
        audio_tensor = audio_tensor.unsqueeze(0)
    return audio_tensor


def merge_and_match(x, y, sr):
    # import scipy.signal
    x = x / 2
    y = y / 2
    import torchaudio.functional as F
    y = F.resample(y, sr, int(sr * (x.shape[-1] / y.shape[-1])))
    if x.shape[0] > y.shape[0]:
        x = x[-y.shape[0]:]
    else:
        y = y[-x.shape[0]:]
    return x.add(y)

def unload_rvc():
    import rvc.modules.rvc as rvc
    rvc.unload_rvc()

def load_rvc(model):
    if not model:
        return unload_rvc()
    import rvc.modules.rvc as rvc
    maximum = rvc.load_rvc(model)


def denoise(sr, audio):
    if not torch.is_tensor(audio):
        audio = torch.tensor(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    audio = audio.detach().cpu().numpy()
    import noisereduce.noisereduce as noisereduce
    audio = torch.tensor(noisereduce.reduce_noise(y=audio, sr=sr))
    return sr, audio


def gen(rvc_model_selected, speaker_id, pitch_extract, audio_in, up_key, index_rate, filter_radius, protect, crepe_hop_length, flag):
    background = None
    audio = None
    sr, audio_in = audio_in
    audio_tuple = (sr, torch.tensor(audio_in))

    audio_tuple = flatten_audio(audio_tuple)

    if 'separate background' in flag:
        if not torch.is_tensor(audio_tuple[1]):
            audio_tuple = (audio_tuple[0], torch.tensor(audio_tuple[1]).to(torch.float32))
        if len(audio_tuple[1].shape) != 1:
            audio_tuple = (audio_tuple[0], audio_tuple[1].flatten())
        import rvc.modules.split_audio as split_audio
        foreground, background, sr = split_audio.split(*audio_tuple)
        audio_tuple = flatten_audio((sr, foreground))
        background = flatten_audio(background)
    if 'denoise' in flag:
        audio_tuple = denoise(*audio_tuple)

    if rvc_model_selected:
        print('Selected model', rvc_model_selected)
        if len(audio_tuple[1].shape) == 1:
            audio_tuple = (audio_tuple[0], audio_tuple[1].unsqueeze(0))
        tmp_voice = uuid.uuid4()
        tmp_path = "output/" + str(tmp_voice) + ".wav"
        torchaudio.save(tmp_path, audio_tuple[1], audio_tuple[0])

        import rvc.modules.rvc as rvc
        rvc.load_rvc(rvc_model_selected)

        index_file = ''
        try:
            model_basedir = os.path.join('data', 'models', 'rvc', os.path.dirname(rvc_model_selected))
            index_files = [f for f in os.listdir(model_basedir) if f.endswith('.index')]
            if len(index_files) > 0:
                for f in index_files:
                    full_path = os.path.join(model_basedir, f)
                    if 'added' in f:
                        index_file = full_path
                if not index_file:
                    index_file = os.path.join(model_basedir, index_files[0])
        except:
            pass

        out1, out2 = rvc.vc_single(speaker_id, tmp_path, up_key, None, pitch_extract, index_file, '', index_rate, filter_radius, 0, 1, protect, crepe_hop_length)
        os.remove(tmp_path)
        audio_tuple = out2

    if background is not None and 'separate background' in flag:
        audio = audio_tuple[1] if torch.is_tensor(audio_tuple[1]) else torch.tensor(audio_tuple[1])
        audio_tuple = (audio_tuple[0], flatten_audio(audio, False))
        background = flatten_audio(background if torch.is_tensor(background) else torch.tensor(background), False)
        if audio_tuple[1].dtype == torch.int16:
            audio = audio_tuple[1]
            audio = audio.float() / 32767.0
            audio_tuple = (audio_tuple[0], audio)
        audio = audio_tuple[1]
        audio_tuple = (audio_tuple[0], merge_and_match(audio_tuple[1], background, audio_tuple[0]))

    if 'denoise output' in flag:
        audio_tuple = denoise(*audio_tuple)

    if torch.is_tensor(audio_tuple[1]):
        audio_tuple = (audio_tuple[0], audio_tuple[1].flatten().detach().cpu().numpy())

    return audio_tuple

