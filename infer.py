#!/usr/bin/env python3
import os
import argparse
import torch
import soundfile as sf
import numpy as np
import librosa
from datasets.utils import find_pairs_in_dir, compute_mix_scale_from_multispec, stft_torch, istft_torch, make_cond_features_from_multispec, cond_channel_count
from datasets.paired_stft import PairedSTFTDataset
from models.unet_beamformer import UnetBeamformer

def inference_and_save(encoder, pairs, out_dir, args, device):
    encoder.eval()
    os.makedirs(out_dir, exist_ok=True)

    for mix_path, ref_path in pairs:
        mix_wav, sr_in = sf.read(mix_path, always_2d=True)
        if mix_wav.ndim == 2 and mix_wav.shape[1] > 1:
            mix_wav_m = mix_wav
        else:
            mix_wav_m = mix_wav[:, 0] if mix_wav.ndim == 2 else mix_wav
            mix_wav_m = mix_wav_m[:, None]

        if sr_in != args.sr:
            # resample per-channel
            mix_wav_m = np.stack([librosa.resample(mix_wav_m[:, c], orig_sr=sr_in, target_sr=args.sr) for c in range(mix_wav_m.shape[1])], axis=1)

        M = mix_wav_m.shape[1]
        spec_list = []

        # build multi-mic STFTs (on CPU), then move to device later
        for m in range(M):
            mic_wave = torch.from_numpy(mix_wav_m[:, m].astype(np.float32)).unsqueeze(0)  # (1, T)
            S = stft_torch(mic_wave, n_fft=args.n_fft, hop_length=args.hop, win_length=args.win_length)[0]
            spec_list.append(S.unsqueeze(0))  # (1, F, T, 2) or similar

        mix_spec_multi = torch.cat(spec_list, dim=0).unsqueeze(0).to(device)  # (1, M, F, T, 2) shape depends on stft_torch
        scale = compute_mix_scale_from_multispec(mix_spec_multi)
        mix_spec_multi_norm = mix_spec_multi / (scale.unsqueeze(1) + 1e-12)

        extra_feat = None
        if (args.features_destination != 'none'):
            cond_feat = make_cond_features_from_multispec(mix_spec_multi_norm)
            extra_feat = cond_feat.to(device)

        # Run model and ISTFT under no_grad to avoid constructing a grad graph.
        with torch.no_grad():
            if extra_feat is not None:
                emb, weights, beam_spec = encoder(mix_spec_multi_norm, extra_feat=extra_feat)
            else:
                emb, weights, beam_spec = encoder(mix_spec_multi_norm)

            # denormalize
            beam_spec_denorm = beam_spec * scale

            try:
                wav_hat_tensor = istft_torch(beam_spec_denorm, n_fft=args.n_fft, hop_length=args.hop, win_length=args.win_length, length=mix_wav_m.shape[0])
                # ensure we detach from any possible grad and move to CPU before converting to numpy
                wav_hat = wav_hat_tensor.squeeze(0).detach().cpu().numpy()
            except Exception as e:
                print(f"[Warning] ISTFT failed for file {mix_path}: {e}")
                wav_hat = np.zeros((mix_wav_m.shape[0],), dtype=np.float32)

        base = os.path.basename(mix_path)
        out_name = os.path.splitext(base)[0] + "_simplebf_enh.wav"
        out_path = os.path.join(out_dir, out_name)
        sf.write(out_path, wav_hat, args.sr)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--test_dir', required=True)
    p.add_argument('--out_dir', default='out/test_enh')
    p.add_argument('--n_fft', type=int, default=320)
    p.add_argument('--hop', type=int, default=128)
    p.add_argument('--win_length', type=int, default=None)
    p.add_argument('--sr', type=int, default=16000)
    p.add_argument('--depth', type=int, default=1)
    p.add_argument('--embed_dim', type=int, default=16)
    p.add_argument('--base_ch', type=int, default=16)
    p.add_argument('--features_destination', choices=['none','encoder','stft'], default='none')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.ckpt, map_location=device)
    args_dict = ck.get('args', {})

    pairs = find_pairs_in_dir(args.test_dir)
    if not pairs:
        raise RuntimeError("No pairs found in test_dir.")

    # build dummy dataset to detect M
    ds = PairedSTFTDataset([pairs[0]], sr=args.sr, n_fft=args.n_fft, hop=args.hop, win_length=args.win_length, random_crop=False, time_divisible=(2**args.depth))
    M = ds.get_num_mics()
    feat_dim = 0
    if (args.features_destination != 'none'):
        feat_dim = cond_channel_count(M)

    encoder = UnetBeamformer(in_mics=M, embed_dim=args.embed_dim, base_ch=args.base_ch, depth=args.depth, feature_dim=feat_dim).to(device)
    encoder.load_state_dict(ck['encoder_state'])

    inference_and_save(encoder, pairs, args.out_dir, args, device)

if __name__ == "__main__":
    main()
