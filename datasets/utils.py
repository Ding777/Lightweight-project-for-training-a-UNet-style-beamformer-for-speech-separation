
---

## `utils.py`
```python
#!/usr/bin/env python3
"""Utility functions shared by train/infer/dataset."""
import os
import glob
import math
import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
import numpy as np
from typing import Tuple, List, Dict

# -------------------------
# find pairs helper
# -------------------------
def _longest_common_prefix_len(a: str, b: str) -> int:
    L = min(len(a), len(b))
    for i in range(L):
        if a[i] != b[i]:
            return i
    return L

def find_pairs_in_dir(folder,
                      audio_exts=(".wav", ".flac"),
                      mix_mark="_mix",
                      ref_suffixes=("_clean_ref", "_ref", "_clean", "_target", "_target_ref")):
    """
    Find (mix, ref) pairs by matching prefixes before known markers.
    Works for names such as:
       scene_000024_mix_multich.wav  <->  scene_000024_target_tile5_ch0.wav

    Returns list of (mix_path, ref_path). Only pairs where a ref is found are returned.
    """

    if folder is None or not os.path.isdir(folder):
        return []

    # collect audio files
    files = []
    for ext in audio_exts:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    files = sorted(set(files))

    mix_map = {}   # prefix -> mix_path
    ref_map = {}   # prefix -> list of ref_paths

    for f in files:
        bname = os.path.basename(f)
        name_no_ext, _ = os.path.splitext(bname)
        lname = name_no_ext.lower()

        # --- detect mix by presence of the mix_mark ("_mix") anywhere in the name ---
        if mix_mark in lname:
            idx = lname.find(mix_mark)
            prefix = lname[:idx]
            # keep first mix deterministically
            if prefix not in mix_map:
                mix_map[prefix] = f
            continue

        # --- detect explicit target tile pattern, prefer that ---
        if "_target_tile" in lname:
            idx = lname.find("_target_tile")
            prefix = lname[:idx]
            ref_map.setdefault(prefix, []).append(f)
            continue

        # --- generic ref suffixes (endswith) ---
        matched = False
        for s in ref_suffixes:
            if lname.endswith(s.lower()):
                prefix = lname[:-len(s)]
                ref_map.setdefault(prefix, []).append(f)
                matched = True
                break
        if matched:
            continue
        # otherwise ignore unknown files

    # Build pairs by matching prefixes. Prefer ref candidate with "_ch0" if present.
    pairs = []
    for prefix, mix_path in mix_map.items():
        if prefix not in ref_map:
            # fuzzy match
            matched_ref = None
            for rpref, rlist in ref_map.items():
                if prefix in rpref or rpref in prefix:
                    chosen = None
                    for rp in rlist:
                        if "_ch0" in os.path.basename(rp).lower():
                            chosen = rp
                            break
                    if chosen is None and rlist:
                        chosen = rlist[0]
                    matched_ref = chosen
                    break
            if matched_ref is not None:
                pairs.append((mix_path, matched_ref))
        else:
            # exact prefix match
            candidates = ref_map[prefix]
            chosen = None
            for c in candidates:
                if "_ch0" in os.path.basename(c).lower():
                    chosen = c
                    break
            if chosen is None and candidates:
                chosen = candidates[0]
            if chosen:
                pairs.append((mix_path, chosen))

    # deterministic order
    pairs = sorted(pairs, key=lambda x: os.path.basename(x[0]))
    return pairs


# -------------------------
# STFT / ISTFT helpers
# -------------------------
def stft_torch(waveform: torch.Tensor, n_fft=512, hop_length=128, win_length=None, window=None, center=True):
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() == 3 and waveform.size(-1) == 1:
        waveform = waveform.squeeze(-1)
    if waveform.dim() != 2:
        raise ValueError("stft_torch expects waveform shaped (B, T) or (T,)")
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(win_length).to(waveform.device)
    S = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                   window=window, center=center, return_complex=False)
    S = S.permute(0, 3, 2, 1).contiguous()  # (B, 2, frames, freq)
    return S

def istft_torch(spec: torch.Tensor, n_fft=512, hop_length=128, win_length=None, window=None, center=True, length=None):
    if spec.dim() == 3:
        spec = spec.unsqueeze(0)
    if spec.dim() != 4:
        raise ValueError("istft_torch expects spec shaped (B,2,frames,freq)")
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(win_length).to(spec.device)
    S = spec.permute(0, 3, 2, 1).contiguous()  # (B, freq, frames, 2)
    S_complex = torch.view_as_complex(S)
    wav = torch.istft(S_complex, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                      window=window, center=center, length=length)
    return wav

# -------------------------
# conditioning features
# -------------------------
def make_cond_features_from_multispec(multi_spec: torch.Tensor) -> torch.Tensor:
    B, M, two, T, Freq = multi_spec.shape
    assert two == 2, "expected real/imag dim=2"
    r = multi_spec[:, :, 0, :, :]
    im = multi_spec[:, :, 1, :, :]
    raw_chs = []
    for m in range(M):
        raw_chs.append(r[:, m:m+1, :, :])
        raw_chs.append(im[:, m:m+1, :, :])
    raw_ch = torch.cat(raw_chs, dim=1)
    cond_list = [raw_ch]
    if M > 1:
        r0 = r[:, 0:1, :, :]
        im0 = im[:, 0:1, :, :]
        mag0 = torch.sqrt(r0*r0 + im0*im0 + 1e-12)
        Re_list, Im_list, ILD_list, sin_list, cos_list = [], [], [], [], []
        for m in range(1, M):
            rm = r[:, m:m+1, :, :]
            im_m = im[:, m:m+1, :, :]
            mag_m = torch.sqrt(rm*rm + im_m*im_m + 1e-12)
            ReC = r0 * rm + im0 * im_m
            ImC = im0 * rm - r0 * im_m
            ipd = torch.atan2(ImC, ReC)
            sin_ipd = torch.sin(ipd)
            cos_ipd = torch.cos(ipd)
            ild = torch.log(mag_m + 1e-12) - torch.log(mag0 + 1e-12)
            Re_list.append(ReC)
            Im_list.append(ImC)
            ILD_list.append(ild)
            sin_list.append(sin_ipd)
            cos_list.append(cos_ipd)
        cond_list.append(torch.cat(Re_list, dim=1))
        cond_list.append(torch.cat(Im_list, dim=1))
        cond_list.append(torch.cat(ILD_list, dim=1))
        cond_list.append(torch.cat(sin_list, dim=1))
        cond_list.append(torch.cat(cos_list, dim=1))
    cond = torch.cat(cond_list, dim=1)
    return cond

def cond_channel_count(M: int) -> int:
    if M <= 1:
        return 2
    return 2*M + 5*(M-1)

# -------------------------
# mix scale
# -------------------------
def compute_mix_scale_from_multispec(multi_spec: torch.Tensor, eps=1e-8):
    B, M, two, T, Freq = multi_spec.shape
    r0 = multi_spec[:, 0:1, 0, :, :]
    im0 = multi_spec[:, 0:1, 1, :, :]
    mag0_sq = r0*r0 + im0*im0
    mean_mag0_sq = mag0_sq.mean(dim=[2,3], keepdim=True)
    scale = torch.sqrt(mean_mag0_sq + eps)
    return scale

# -------------------------
# si-sdr
# -------------------------
def si_sdr(est_wav: torch.Tensor, ref_wav: torch.Tensor, eps=1e-8):
    B = ref_wav.shape[0]
    ref_zm = ref_wav - ref_wav.mean(dim=1, keepdim=True)
    est_zm = est_wav - est_wav.mean(dim=1, keepdim=True)
    s_target = (torch.sum(ref_zm * est_zm, dim=1, keepdim=True) / (torch.sum(ref_zm * ref_zm, dim=1, keepdim=True) + eps)) * ref_zm
    e_noise = est_zm - s_target
    ratio = torch.sum(s_target * s_target, dim=1) / (torch.sum(e_noise * e_noise, dim=1) + eps)
    sdr = 10.0 * torch.log10(ratio + eps)
    return sdr.mean()

# -------------------------
# init helper
# -------------------------
def init_weights_xavier(module: torch.nn.Module):
    for m in module.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
            try:
                torch.nn.init.xavier_uniform_(m.weight)
            except Exception:
                pass
            if getattr(m, 'bias', None) is not None:
                try:
                    torch.nn.init.zeros_(m.bias)
                except Exception:
                    pass
        elif isinstance(m, (torch.nn.GroupNorm, torch.nn.InstanceNorm2d, torch.nn.LayerNorm)):
            try:
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            except Exception:
                pass

# -------------------------
# collate (used by dataloader)
# -------------------------
def pad_collate_fn(batch, spec_augment=False, freq_mask_param=8, time_mask_param=20, num_freq_masks=1, num_time_masks=1):
    mix_items = []
    ref_items = []
    lengths = []
    mix_paths = []
    ref_paths = []
    first_mix, first_ref, _, _, _ = batch[0]
    _, M, _, frames0, freq0 = first_mix.shape
    frames_list = []
    for item in batch:
        mix_spec_multi, ref_spec, length, mix_path, ref_path = item
        if mix_spec_multi.dim() == 5 and mix_spec_multi.size(0) == 1:
            mix_spec_multi = mix_spec_multi.squeeze(0)
        if ref_spec.dim() == 4 and ref_spec.size(0) == 1:
            ref_spec = ref_spec.squeeze(0)
        if mix_spec_multi.shape[0] != M:
            raise RuntimeError(f"In-batch mic mismatch: expected M={M} got {mix_spec_multi.shape[0]}")
        frames_list.append(mix_spec_multi.shape[2])
        mix_items.append(mix_spec_multi)
        ref_items.append(ref_spec)
        lengths.append(length)
        mix_paths.append(mix_path)
        ref_paths.append(ref_path)
    max_frames = max(frames_list)
    padded_mix = []
    padded_ref = []
    for ms, rs in zip(mix_items, ref_items):
        frames_i = ms.shape[2]
        pad_frames = max_frames - frames_i
        if pad_frames == 0:
            padded_mix.append(ms)
            padded_ref.append(rs)
        else:
            padded_mix.append(F.pad(ms, (0,0,0,pad_frames), mode='constant', value=0.0))
            padded_ref.append(F.pad(rs, (0,0,0,pad_frames), mode='constant', value=0.0))
    mix_batch = torch.stack(padded_mix, dim=0)
    ref_batch = torch.stack(padded_ref, dim=0)
    # simple SpecAugment applied lightly
    if spec_augment:
        B, M, two, T, Freq = mix_batch.shape
        for b in range(B):
            for _ in range(num_freq_masks):
                f = np.random.randint(0, freq_mask_param+1)
                f0 = np.random.randint(0, max(1, Freq - f))
                mix_batch[b, :, :, :, f0:f0+f] = 0.0
                ref_batch[b, :, :, :, f0:f0+f] = 0.0
            for _ in range(num_time_masks):
                t = np.random.randint(0, time_mask_param+1)
                t0 = np.random.randint(0, max(1, T - t))
                mix_batch[b, :, :, t0:t0+t, :] = 0.0
                ref_batch[b, :, :, t0:t0+t, :] = 0.0
    lengths_tensor = torch.tensor([int(min(l, max_frames)) for l in lengths], dtype=torch.int64)
    return mix_batch, ref_batch, lengths_tensor, mix_paths, ref_paths
