#!/usr/bin/env python3
"""
Paired STFT dataset that optionally trims STFT frames so `frames % time_divisible == 0`.
"""
from typing import List, Tuple
import numpy as np
import soundfile as sf
import librosa
import torch
from . import utils  # relative import if used as package; else use absolute import
import os

# We assume utils.stft_torch is available; if not, import as below:
from utils import stft_torch  # if running as script adjust PYTHONPATH accordingly

class PairedSTFTDataset(torch.utils.data.Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], sr=16000, segment_len=4.0,
                 n_fft=512, hop=128, win_length=None, random_crop=True, spec_augment=False, random_gain=False,
                 time_divisible: int = 0, pad_audio: bool = False):
        """
        time_divisible: if >0, trim STFT frames so frames % time_divisible == 0.
                        If pad_audio=True, then pad waveform before STFT instead of trimming.
        """
        self.pairs = pairs
        self.sr = int(sr)
        self.segment_len = float(segment_len)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.win_length = n_fft if win_length is None else int(win_length)
        self.random_crop = bool(random_crop)
        self.spec_augment = bool(spec_augment)
        self.random_gain = bool(random_gain)
        self.time_divisible = int(time_divisible)
        self.pad_audio = bool(pad_audio)
        if len(self.pairs) > 0:
            first_mix = self.pairs[0][0]
            data, sr0 = sf.read(first_mix, always_2d=True)
            self.mics = data.shape[1] if data.ndim == 2 else 1
        else:
            self.mics = None

    def __len__(self):
        return len(self.pairs)

    def _load_audio(self, path: str):
        data, sr = sf.read(path, always_2d=True)
        data = data.astype(np.float32)
        if sr != self.sr:
            data = np.stack([librosa.resample(data[:, c], orig_sr=sr, target_sr=self.sr) for c in range(data.shape[1])], axis=1)
        return data

    def _pad_wave_to_frames(self, wave: np.ndarray, required_frames: int):
        # compute required samples to make frames >= required_frames
        # frames = 1 + floor((N - n_fft) / hop)  -> N = (frames - 1) * hop + n_fft
        target_samples = (required_frames - 1) * self.hop + self.n_fft
        if wave.shape[0] >= target_samples:
            return wave
        pad = target_samples - wave.shape[0]
        return np.pad(wave, ((0, pad), (0, 0)), mode='constant')

    def __getitem__(self, idx):
        mix_path, ref_path = self.pairs[idx]
        mix = self._load_audio(mix_path)
        ref = self._load_audio(ref_path)
        if ref.ndim == 2:
            ref = ref[:, 0]
        if mix.ndim == 1:
            mix = mix[:, None]
        M_here = mix.shape[1]
        if self.mics is None:
            self.mics = M_here
        if M_here != self.mics:
            raise RuntimeError(f"Mismatch mic count: expected {self.mics}, found {M_here} in file {mix_path}")

        if self.random_crop:
            seg_len_samps = int(self.segment_len * self.sr)
            max_start = max(0, len(ref) - seg_len_samps)
            s = np.random.randint(0, max_start+1) if max_start > 0 else 0
            e = s + seg_len_samps
            mix = mix[s:e, :]
            ref = ref[s:e]
            length = seg_len_samps
        else:
            length = len(ref)

        if self.random_gain:
            g = 10 ** (np.random.uniform(-6, 6) / 20.0)
            mix = mix * g

        # If pad_audio selected and time_divisible used, we may pad signals to ensure frame divisibility
        # We'll compute frames after STFT; easier to pad waveform to nearest frames multiple
        spec_list = []
        for m in range(M_here):
            mic_wave = torch.from_numpy(mix[:, m].astype(np.float32)).unsqueeze(0)
            S = stft_torch(mic_wave, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win_length)[0]
            spec_list.append(S.unsqueeze(0))
        mix_spec_multi = torch.cat(spec_list, dim=0).unsqueeze(0)  # (1, M, 2, frames, freq)

        ref_wave = torch.from_numpy(ref.astype(np.float32)).unsqueeze(0)
        ref_spec = stft_torch(ref_wave, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win_length)[0].unsqueeze(0)

        # Trim frames to divisible by time_divisible if requested
        if self.time_divisible and mix_spec_multi.shape[3] > 0:
            frames = int(mix_spec_multi.shape[3])
            k = self.time_divisible
            frames_trim = (frames // k) * k
            if frames_trim == 0:
                # keep at least one block (corner case)
                frames_trim = min(frames, k)
            if frames_trim != frames:
                # Optionally pad audio before STFT would be done here. For now we trim.
                mix_spec_multi = mix_spec_multi[:, :, :, :frames_trim, :]
                # align ref
                if ref_spec.shape[2] >= frames_trim:
                    ref_spec = ref_spec[:, :, :frames_trim, :]
                else:
                    # if ref shorter, crop mix to ref frames
                    mix_spec_multi = mix_spec_multi[:, :, :, :ref_spec.shape[2], :]

        return mix_spec_multi, ref_spec, length, mix_path, ref_path

    def get_num_mics(self):
        return int(self.mics) if self.mics is not None else None
