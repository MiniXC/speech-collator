import os
import pickle
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn import ConstantPad1d, ConstantPad2d
import torchaudio
from nnAudio.features.mel import MelSpectrogram
import torchaudio.transforms as AT
from librosa.filters import mel as librosa_mel
from time import time
import pandas as pd
from tqdm.auto import tqdm
pd.options.mode.chained_assignment=None

class JitWrapper():
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = torch.jit.load(path)

    def __getstate__(self):
        self.path 
        return self.path

    def __setstate__(self, d):
        self.path = d
        self.model = torch.jit.load(d)

_DATA_DIR = os.getenv("SPEECHCOLLATOR_PATH", os.path.dirname(__file__))

def resample(x, vpw=5):
    return np.interp(np.linspace(0, 1, vpw), np.linspace(0, 1, len(x)), x)

class SpeechCollator():
    def __init__(
        self,
        phone2idx,
        speaker2idx,
        measures=None,
        pad_to_max_length=True,
        pad_to_multiple_of=None,
        include_audio=False,
        overwrite_max_length=False,
        max_lengths={
            "frame": 512,
            "phone": 384,
        },
        audio_args={
            "sample_rate": 22050,
            "hop_length": 256,
            "win_length": 1024,
            "n_fft": 1024,
            "n_mels": 80,
        },
        return_keys=None,
        wave_augmentation_func=None,
        mel_augmentation_func=None,
        use_speaker_prompt=False,
        speaker_prompt_frames=256,
        speaker_prompt_wave_augmentation_func=None,
        vocex_model=None,
        expand_seq=False,
        overwrite_cache=False,
    ):
        # download dvector and wav2mel
        local_data_dir = f"{_DATA_DIR}/data"
        if not os.path.exists(local_data_dir):
            os.makedirs(local_data_dir)
        if not os.path.exists(os.path.join(local_data_dir, "dvector-step250000.pt")):
            print("Downloading dvector from https://github.com/yistLin/dvector")
            os.system(f"wget -P {local_data_dir} https://github.com/yistLin/dvector/releases/download/v1.1.1/dvector-step250000.pt 2> /dev/null")
        if not os.path.exists(os.path.join(local_data_dir, "wav2mel.pt")):
            print("Downloading wav2mel from https://github.com/yistLin/dvector")
            os.system(f"wget -P {local_data_dir} https://github.com/yistLin/dvector/releases/download/v1.1.1/wav2mel.pt 2> /dev/null")
        self.sampling_rate = audio_args["sample_rate"]
        self.measures = measures
        self.phone2idx = phone2idx
        self.speaker2idx = speaker2idx
        # find max audio length & max duration
        self.max_frame_length = max_lengths["frame"]
        self.max_phone_length = max_lengths["phone"]
        self.pad_to_max_length = pad_to_max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.wav2mel = JitWrapper(f"{local_data_dir}/wav2mel.pt")
        self.dvector = JitWrapper(f"{local_data_dir}/dvector-step250000.pt")
        self.num_masked = 0
        self.num_total = 0
        self.percentage_mask_tokens = 0
        self.mel_spectrogram = AT.Spectrogram(
            n_fft=audio_args["n_fft"],
            win_length=audio_args["win_length"],
            hop_length=audio_args["hop_length"],
            pad=0,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.mel_basis = librosa_mel(
            sr=self.sampling_rate,
            n_fft=audio_args["n_fft"],
            n_mels=audio_args["n_mels"],
            fmin=0,
            fmax=8000,
        )
        self.mel_basis = torch.from_numpy(self.mel_basis).float()
        self.include_audio = include_audio
        self.overwrite_max_length = overwrite_max_length
        self.audio_args = audio_args
        self.pad_value = self.phone2idx["<pad>"]
        if return_keys is None:
            if self.measures is not None:
                self.return_keys = [
                    "audio",
                    "mel",
                    "phone_durations",
                    "phones",
                    "speaker",
                    "measures",
                ]
            else:
                self.return_keys = [
                    "audio",
                    "mel",
                    "phone_durations",
                    "phones",
                    "speaker",
                    "vocex",
                ]
        else:
            self.return_keys = return_keys

        self.wave_augmentation_func = wave_augmentation_func
        self.mel_augmentation_func = mel_augmentation_func

        self.use_speaker_prompt = use_speaker_prompt
        self.speaker_prompt_frames = speaker_prompt_frames-1
        self.speaker_prompt_wave_augmentation_func = speaker_prompt_wave_augmentation_func

        self.vocex_model = vocex_model

        self.expand = expand_seq

        self.overwrite_cache = overwrite_cache

    @staticmethod
    def drc(x, C=1, clip_val=1e-7):
        return torch.log(torch.clamp(x, min=clip_val) * C)
        
    def _expand(self, values, durations):
        out = []
        for value, d in zip(values, durations):
            out += [value] * max(0, int(d))
        if isinstance(values, list):
            return np.array(out)
        elif isinstance(values, torch.Tensor):
            return torch.stack(out)
        elif isinstance(values, np.ndarray):
            return np.array(out)
    
    def collate_fn(self, batch):
        result = {}

        # speaker prompt
        if self.use_speaker_prompt:
            prompts = [[x for x in row["audio_speaker_prompt"] if x != row["audio"]] for row in batch]
            # random speaker prompt
            prompts = [p[np.random.randint(0, len(p))] for p in prompts]
            prompt_audios = [
                torchaudio.load(p)[0][0].numpy() for p in prompts
            ]
            prompt_len = self.speaker_prompt_frames*self.audio_args["hop_length"]
            # concat random segments of prompt audios
            prompt_audio_arr = np.zeros(len(prompt_audios)*prompt_len)
            for i, prompt_audio in enumerate(prompt_audios):
                if len(prompt_audio)-prompt_len > 0:
                    start = np.random.randint(0, len(prompt_audio)-prompt_len)
                else:
                    start = 0
                temp_audio = prompt_audio[start:start+prompt_len]
                temp_audio = temp_audio / np.abs(temp_audio).max()
                audio_len = len(temp_audio)
                prompt_audio_arr[i*prompt_len:i*prompt_len+audio_len] = temp_audio
            
            if self.speaker_prompt_wave_augmentation_func is not None:
                prompt_audio_arr = self.speaker_prompt_wave_augmentation_func(prompt_audio_arr)

            prompt_audio_arr = torch.tensor(prompt_audio_arr).reshape(len(prompt_audios), prompt_len)
            prompt_mels = self.mel_spectrogram(prompt_audio_arr)
            prompt_mels = torch.sqrt(prompt_mels)
            prompt_mels = prompt_mels.to(torch.float32)
            prompt_mels = torch.matmul(self.mel_basis, prompt_mels)
            prompt_mels = SpeechCollator.drc(prompt_mels).transpose(1, 2)

        for i, row in enumerate(batch):
            phones = row["phones"]
            batch[i]["phones"] = np.array([self.phone2idx[phone.replace("ˌ", "")] for phone in row["phones"]])
            sr = self.sampling_rate
            start = int(sr * row["start"])
            end = int(sr * row["end"])
            audio_path = row["audio"]
            # load audio with torch audio and then resample
            audio, sr = torchaudio.load(audio_path)
            # resample
            if sr != self.sampling_rate:
                audio = torchaudio.transforms.Resample(sr, self.sampling_rate)(audio)
            audio = audio[0].numpy()
            audio = audio[start:end]
            audio = audio / np.abs(audio).max()
            if self.wave_augmentation_func is not None:
                augmented_audio = audio.copy()
                augmented_audio = self.wave_augmentation_func(augmented_audio)
            else:
                augmented_audio = None

            durations = np.array(row["phone_durations"])

            max_audio_len = int(durations.sum() * self.audio_args["hop_length"])
            if len(audio) < max_audio_len:
                audio = np.pad(audio, (0, max_audio_len - len(audio)))
                if augmented_audio is not None:
                    augmented_audio = np.pad(augmented_audio, (0, max_audio_len - len(augmented_audio)))
            elif len(audio) > max_audio_len:
                audio = audio[:max_audio_len]
                if augmented_audio is not None:
                    augmented_audio = augmented_audio[:max_audio_len]
            
            duration_permutation = np.argsort(durations+np.random.normal(0, durations.std(), len(durations)))
            duration_mask_rm = durations[duration_permutation].cumsum() >= self.max_frame_length
            duration_mask_rm = duration_mask_rm[np.argsort(duration_permutation)]
            batch[i]["phones"][duration_mask_rm] = self.phone2idx["<mask>"]
            duration_mask_rm_exp = np.repeat(duration_mask_rm, durations * self.audio_args["hop_length"])
            dur_sum = sum(durations)
            self.num_total += 1
            self.num_masked += 1 if sum(duration_mask_rm) > 0 else 0
            self.percentage_mask_tokens += sum(duration_mask_rm_exp) / len(duration_mask_rm_exp)
            durations[duration_mask_rm] = 0
            batch[i]["audio"] = audio[~duration_mask_rm_exp]
            if augmented_audio is not None:
                batch[i]["augmented_audio"] = augmented_audio[~duration_mask_rm_exp]
            new_mel_len = int(np.ceil(len(batch[i]["audio"]) / self.audio_args["hop_length"]))
            # compute mel spectrogram
            mel = self.mel_spectrogram(torch.tensor(batch[i]["audio"]).unsqueeze(0))
            mel = torch.sqrt(mel[0])
            mel = torch.matmul(self.mel_basis, mel)
            mel = SpeechCollator.drc(mel)

            if augmented_audio is not None:
                augmented_mel = self.mel_spectrogram(torch.tensor(batch[i]["augmented_audio"]).unsqueeze(0))
                augmented_mel = torch.sqrt(augmented_mel[0])
                augmented_mel = torch.matmul(self.mel_basis, augmented_mel)
                augmented_mel = SpeechCollator.drc(augmented_mel)
            else:
                augmented_mel = None

            batch[i]["mel"] = mel.T        

            if augmented_mel is not None:
                batch[i]["augmented_mel"] = augmented_mel.T
            if batch[i]["mel"].shape[0] > new_mel_len:
                batch[i]["mel"] = batch[i]["mel"][:new_mel_len]
                if augmented_mel is not None:
                    batch[i]["augmented_mel"] = batch[i]["augmented_mel"][:new_mel_len]
            if batch[i]["mel"].shape[0] < new_mel_len:
                batch[i]["mel"] = torch.cat([batch[i]["mel"], torch.zeros(1, batch[i]["mel"].shape[1])])
                if augmented_mel is not None:
                    batch[i]["augmented_mel"] = torch.cat([batch[i]["augmented_mel"], torch.zeros(1, batch[i]["augmented_mel"].shape[1])])
            
            unexpanded_silence_mask = ["[" in p for p in phones]
            silence_mask = self._expand(unexpanded_silence_mask, durations)
            batch[i]["phone_durations"] = durations.copy()
            durations = durations + (np.random.rand(*durations.shape))
            batch[i]["durations"] = durations

            if self.vocex_model is not None and "vocex" in self.return_keys:
                vocex_cache_path = audio_path.replace(".wav", "_vocex.npy")
                if os.path.exists(vocex_cache_path) and not self.overwrite_cache:
                    batch[i]["vocex"] = torch.tensor(np.load(vocex_cache_path)).float()
                else:
                    with torch.no_grad():
                        v_results = self.vocex_model(batch[i]["mel"].unsqueeze(0), inference=True)
                        pitch = v_results["measures"]["pitch"][0]
                        energy = v_results["measures"]["energy"][0]
                        va = v_results["measures"]["voice_activity_binary"][0]
                        pitch = (pitch - pitch.mean()) / pitch.std()
                        energy = (energy - energy.mean()) / energy.std()
                        va = (va - 0.5) * 2
                        p_dur_gt0 = batch[i]["phone_durations"]
                        current_idx = 0
                        vals_per_window = 5
                        new_repr = np.zeros((len(p_dur_gt0), vals_per_window*3+1))
                        for j, d in enumerate(p_dur_gt0):
                            if d == 0:
                                new_repr[j, :] = 0
                                continue
                            pitch_window = pitch[current_idx:current_idx+d]
                            energy_window = energy[current_idx:current_idx+d]
                            va_window = va[current_idx:current_idx+d]
                            new_repr[j, 1:vals_per_window+1] = resample(pitch_window, vals_per_window)
                            new_repr[j, vals_per_window+1:vals_per_window*2+1] = resample(energy_window, vals_per_window)
                            new_repr[j, vals_per_window*2+1:vals_per_window*3+1] = resample(va_window, vals_per_window)
                            current_idx += d
                        new_repr[:, 0] = p_dur_gt0
                        new_repr[:, 0] = np.log10(new_repr[:, 0] + 1)
                        np.save(vocex_cache_path, new_repr)
                        batch[i]["vocex"] = torch.tensor(new_repr).float()
                if self.expand:
                    batch[i]["vocex"] = self._expand(batch[i]["vocex"], batch[i]["phone_durations"])

            if self.measures is not None:
                measure_paths = {
                    m: audio_path.replace(".wav", "_{m}.pkl")
                    for m in [measure.name for measure in self.measures]
                }
                if all([os.path.exists(path) for path in measure_paths]):
                    measures = {}
                    for measure in self.measures:
                        with open(measure_paths[measure.name], "rb") as f:
                            measures[measure.name] = pickle.load(f)
                else:
                    measure_dict = {
                        measure.name: measure(batch[i]["audio"], row["phone_durations"], silence_mask, True)
                        for measure in self.measures
                    }
                    measures = {
                        key: (value["measure"])
                        for key, value in measure_dict.items()
                    }
                batch[i]["measures"] = measures
                # cast to 32 bit
                for key, value in batch[i]["measures"].items():
                    batch[i]["measures"][key] = value.astype(np.float32)
            batch[i]["audio_path"] = audio_path
        max_frame_length = max([sum(x["phone_durations"]) for x in batch])
        max_phone_length = max([len(x["phones"]) for x in batch])
        min_frame_length = min([sum(x["phone_durations"]) for x in batch])
        
        random_min_frame_length = np.random.randint(0, min_frame_length)
        if self.pad_to_multiple_of is not None:
            max_frame_length = (max_frame_length // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of
            max_phone_length = (max_phone_length // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of
        if self.pad_to_max_length and self.overwrite_max_length:
            max_frame_length = max(self.max_frame_length, max_frame_length)
            max_phone_length = max(self.max_phone_length, max_phone_length)
        max_audio_length = (max_frame_length * self.audio_args["hop_length"])
        if "vocex" in self.return_keys and self.vocex_model is not None:
            if self.expand:
                batch[0]["vocex"] = ConstantPad2d(
                    (0, 0, 0, max_frame_length - batch[0]["vocex"].shape[0]), 0
                )(batch[0]["vocex"])
            else:
                batch[0]["vocex"] = ConstantPad2d(
                    (0, 0, 0, max_phone_length - batch[0]["vocex"].shape[0]), 0
                )(batch[0]["vocex"])
        if "audio" in self.return_keys or "dvector" in self.return_keys:
            batch[0]["audio"] = ConstantPad1d(
                (0, max_audio_length - len(batch[0]["audio"])), 0
            )(torch.tensor(batch[0]["audio"]))
        if "augmented_audio" in self.return_keys:
            batch[0]["augmented_audio"] = ConstantPad1d(
                (0, max_audio_length - len(batch[0]["augmented_audio"])), 0
            )(torch.tensor(batch[0]["augmented_audio"]))
        if "mel" in self.return_keys:
            batch[0]["mel"] = ConstantPad2d(
                (0, 0, 0, max_frame_length - batch[0]["mel"].shape[0]), 0
            )(batch[0]["mel"])
        if "augmented_mel" in self.return_keys:
            batch[0]["augmented_mel"] = ConstantPad2d(
                (0, 0, 0, max_frame_length - batch[0]["augmented_mel"].shape[0]), 0
            )(batch[0]["augmented_mel"])
        if "phone_durations" in self.return_keys:
            batch[0]["phone_durations"] = ConstantPad1d(
                (0, max_phone_length - len(batch[0]["phone_durations"])), 0
            )(torch.tensor(batch[0]["phone_durations"]))
        if "durations" in self.return_keys:
            batch[0]["durations"] = ConstantPad1d(
                (0, max_phone_length - len(batch[0]["durations"])), 0
            )(torch.tensor(batch[0]["durations"]))
        if not self.expand:
            if "phones" in self.return_keys:
                batch[0]["phones"] = ConstantPad1d((0, max_phone_length - len(batch[0]["phones"])), self.pad_value
                )(torch.tensor(batch[0]["phones"]))
        else:
            for i in range(len(batch)):
                batch[i]["phones"] = self._expand(batch[i]["phones"], batch[i]["phone_durations"])
            if "phones" in self.return_keys:
                batch[0]["phones"] = ConstantPad1d((0, max_frame_length - len(batch[0]["phones"])), self.pad_value
                )(torch.tensor(batch[0]["phones"]))
        if self.measures is not None and "measures" in self.return_keys:
            for measure in self.measures:
                batch[0]["measures"][measure.name] = ConstantPad1d(
                    (0, max_frame_length - len(batch[0]["measures"][measure.name])), 0
                )(torch.tensor(batch[0]["measures"][measure.name]))
        for i in range(1, len(batch)):
            if "vocex" in self.return_keys and self.vocex_model is not None:
                if not isinstance(batch[i]["vocex"], torch.Tensor):
                    batch[i]["vocex"] = torch.tensor(batch[i]["vocex"])
            if "audio" in self.return_keys or "dvector" in self.return_keys:
                batch[i]["audio"] = torch.tensor(batch[i]["audio"])
            if "augmented_audio" in self.return_keys:
                batch[i]["augmented_audio"] = torch.tensor(batch[i]["augmented_audio"])
            if "phone_durations" in self.return_keys:
                batch[i]["phone_durations"] = torch.tensor(batch[i]["phone_durations"])
            if "durations" in self.return_keys:
                batch[i]["durations"] = torch.tensor(batch[i]["durations"])
            if "phones" in self.return_keys:
                batch[i]["phones"] = torch.tensor(batch[i]["phones"])
            if self.measures is not None and "measures" in self.return_keys:
                for measure in self.measures:
                    batch[i]["measures"][measure.name] = torch.tensor(batch[i]["measures"][measure.name])
        with torch.no_grad():
            if "dvector" in self.return_keys:
                result["dvector"] = []
                for x in batch:
                    try:
                        embed = self.dvector.model.embed_utterance(self.wav2mel.model(x["audio"].unsqueeze(0), 22050)).squeeze(0)
                    except RuntimeError:
                        embed = torch.zeros(256)
                    result["dvector"].append(embed)
                result["dvector"] = torch.stack(result["dvector"])
                torch.cuda.empty_cache()
        if "vocex" in self.return_keys and self.vocex_model is not None:
            result["vocex"] = pad_sequence([x["vocex"] for x in batch], batch_first=True, padding_value=0)
        if "audio" in self.return_keys:
            result["audio"] = pad_sequence([x["audio"] for x in batch], batch_first=True, padding_value=0)
        if "augmented_audio" in self.return_keys:
            result["augmented_audio"] = pad_sequence([x["augmented_audio"] for x in batch], batch_first=True, padding_value=0)
        if "mel" in self.return_keys:
            result["mel"] = pad_sequence([x["mel"] for x in batch], batch_first=True, padding_value=0)
        if "augmented_mel" in self.return_keys:
            result["augmented_mel"] = pad_sequence([x["augmented_mel"] for x in batch], batch_first=True, padding_value=0)
        if "phone_durations" in self.return_keys:
            result["phone_durations"] = pad_sequence([x["phone_durations"] for x in batch], batch_first=True, padding_value=0)
        if "durations" in self.return_keys:
            result["durations"] = pad_sequence([x["durations"] for x in batch], batch_first=True, padding_value=0)
        if "phones" in self.return_keys:
            result["phones"] = pad_sequence([x["phones"] for x in batch], batch_first=True, padding_value=self.pad_value)
        if "speaker" in self.return_keys:
            speakers = [str(x["speaker"]).split("/")[-1] if ("/" in str(x["speaker"])) else x["speaker"] for x in batch]
            # speaker2idx
            result["speaker"] = torch.tensor([self.speaker2idx[x] for x in speakers])

        if self.overwrite_max_length and "phone_durations" in self.return_keys and "val_ind" in self.return_keys:
            MAX_FRAMES = self.max_frame_length
            MAX_PHONES = self.max_phone_length
            BATCH_SIZE = len(batch)
            result["phone_durations"][:, -1] = MAX_FRAMES - result["phone_durations"].sum(-1)
            result["val_ind"] = torch.arange(0, MAX_PHONES).repeat(BATCH_SIZE).reshape(BATCH_SIZE, MAX_PHONES)
            result["val_ind"] = result["val_ind"].flatten().repeat_interleave(result["phone_durations"].flatten(), dim=0).reshape(BATCH_SIZE, MAX_FRAMES)

        if self.measures is not None and "measures" in self.return_keys:
            result["measures"] = {}
            for measure in self.measures:
                result["measures"][measure.name] = pad_sequence([x["measures"][measure.name] for x in batch], batch_first=True, padding_value=self.pad_value)
        elif "measures" in self.return_keys:
            result["measures"] = None

        result = {
            k: v for k, v in result.items() if k in self.return_keys
        }

        if self.use_speaker_prompt:
            result["speaker_prompt_mel"] = prompt_mels

        result["frame_mask"] = result["mel"].sum(dim=-1) != 0
        result["phone_mask"] = result["phones"] != self.pad_value

        return result

def create_speaker2idx(dataset, additonal_tokens=["<unk>"]):
    speaker2idx = {
        k: v for k, v in zip(additonal_tokens, range(len(additonal_tokens)))
    }
    for row in tqdm(dataset):
        if row["speaker"] not in speaker2idx:
            speaker2idx[row["speaker"]] = len(speaker2idx)
    return speaker2idx

def create_phone2idx(dataset, additonal_tokens=["<pad>", "<mask>", "<unk>"]):
    phone2idx = {
        k: v for k, v in zip(additonal_tokens, range(len(additonal_tokens)))
    }
    for row in tqdm(dataset):
        for phone in row["phones"]:
            if phone not in phone2idx:
                phone2idx[phone] = len(phone2idx)
    return phone2idx
    