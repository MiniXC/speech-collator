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
                ]
        else:
            self.return_keys = return_keys

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

            durations = np.array(row["phone_durations"])

            max_audio_len = int(durations.sum() * self.audio_args["hop_length"])
            if len(audio) < max_audio_len:
                audio = np.pad(audio, (0, max_audio_len - len(audio)))
            elif len(audio) > max_audio_len:
                audio = audio[:max_audio_len]
            
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
            new_mel_len = int(np.ceil(len(batch[i]["audio"]) / self.audio_args["hop_length"]))
            # compute mel spectrogram
            mel = self.mel_spectrogram(torch.tensor(batch[i]["audio"]).unsqueeze(0))
            mel = torch.sqrt(mel[0])
            mel = torch.matmul(self.mel_basis, mel)
            mel = SpeechCollator.drc(mel)

            batch[i]["mel"] = mel.T
            if batch[i]["mel"].shape[0] > new_mel_len:
                batch[i]["mel"] = batch[i]["mel"][:new_mel_len]
            if batch[i]["mel"].shape[0] < new_mel_len:
                batch[i]["mel"] = torch.cat([batch[i]["mel"], torch.zeros(1, batch[i]["mel"].shape[1])])
            
            unexpanded_silence_mask = ["[" in p for p in phones]
            silence_mask = self._expand(unexpanded_silence_mask, durations)
            batch[i]["phone_durations"] = durations.copy()
            durations = durations + (np.random.rand(*durations.shape))
            batch[i]["durations"] = durations
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
        if "audio" in self.return_keys or "dvector" in self.return_keys:
            batch[0]["audio"] = ConstantPad1d(
                (0, max_audio_length - len(batch[0]["audio"])), 0
            )(torch.tensor(batch[0]["audio"]))
        if "mel" in self.return_keys:
            batch[0]["mel"] = ConstantPad2d(
                (0, 0, 0, max_frame_length - batch[0]["mel"].shape[0]), 0
            )(batch[0]["mel"])
        if "phone_durations" in self.return_keys:
            batch[0]["phone_durations"] = ConstantPad1d(
                (0, max_phone_length - len(batch[0]["phone_durations"])), 0
            )(torch.tensor(batch[0]["phone_durations"]))
        if "durations" in self.return_keys:
            batch[0]["durations"] = ConstantPad1d(
                (0, max_phone_length - len(batch[0]["durations"])), 0
            )(torch.tensor(batch[0]["durations"]))
        if "phones" in self.return_keys:
            batch[0]["phones"] = ConstantPad1d((0, max_phone_length - len(batch[0]["phones"])), 0
            )(torch.tensor(batch[0]["phones"]))
        if self.measures is not None and "measures" in self.return_keys:
            for measure in self.measures:
                batch[0]["measures"][measure.name] = ConstantPad1d(
                    (0, max_frame_length - len(batch[0]["measures"][measure.name])), 0
                )(torch.tensor(batch[0]["measures"][measure.name]))
        for i in range(1, len(batch)):
            if "audio" in self.return_keys or "dvector" in self.return_keys:
                batch[i]["audio"] = torch.tensor(batch[i]["audio"])
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
        if "audio" in self.return_keys:
            result["audio"] = pad_sequence([x["audio"] for x in batch], batch_first=True, padding_value=self.pad_value)
        if "mel" in self.return_keys:
            result["mel"] = pad_sequence([x["mel"] for x in batch], batch_first=True, padding_value=self.pad_value)
        if "phone_durations" in self.return_keys:
            result["phone_durations"] = pad_sequence([x["phone_durations"] for x in batch], batch_first=True, padding_value=self.pad_value)
        if "durations" in self.return_keys:
            result["durations"] = pad_sequence([x["durations"] for x in batch], batch_first=True, padding_value=self.pad_value)
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
    