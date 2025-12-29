#!/usr/bin/env python3
"""
Train script that wires dataset -> model -> training loop.
"""
import os
import argparse
import random
import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from datasets.utils import find_pairs_in_dir, compute_mix_scale_from_multispec, si_sdr, init_weights_xavier, pad_collate_fn, make_cond_features_from_multispec, cond_channel_count
from datasets.paired_stft import PairedSTFTDataset
from models.unet_beamformer import UnetBeamformer


def train_one_epoch(encoder, opt, loader, device, epoch, args, scheduler=None):
    encoder.train()
    total_loss = 0.0
    total_items = 0
    pbar = tqdm(loader, desc=f"Train {epoch}")
    for batch_idx, batch in enumerate(pbar):
        try:
            mix_spec_multi, ref_spec, lengths, _, _ = batch
        except Exception as e:
            print(f"[DataLoader error] at batch {batch_idx}: {e}")
            continue
        mix_spec_multi = mix_spec_multi.to(device)
        ref_spec = ref_spec.to(device)
        B = ref_spec.shape[0]
        scale = compute_mix_scale_from_multispec(mix_spec_multi)
        mix_spec_multi_norm = mix_spec_multi / (scale.unsqueeze(1) + 1e-12)
        ref_spec_norm = ref_spec / (scale + 1e-12)

        extra_feat = None
        if (args.features_destination != 'none') and (args.encoder_type in ('features','both')):
            cond_feat = make_cond_features_from_multispec(mix_spec_multi_norm)
            extra_feat = cond_feat.to(device)

        if extra_feat is not None:
            emb, weights, beam_spec = encoder(mix_spec_multi_norm, extra_feat=extra_feat)
        else:
            emb, weights, beam_spec = encoder(mix_spec_multi_norm)

        mag = torch.sqrt(ref_spec_norm[:, 0:1, :, :]**2 + ref_spec_norm[:, 1:2, :, :]**2 + 1e-12)
        valid_frame_mask = (mag.squeeze(1).abs().sum(dim=-1) > 0).float()
        mask = valid_frame_mask.unsqueeze(1).unsqueeze(-1)
        mask_full = mask.expand(-1, beam_spec.shape[1], -1, beam_spec.shape[3])

        loss_mse = nn.functional.mse_loss(beam_spec * mask_full, ref_spec_norm * mask_full, reduction='sum')
        denom = mask_full.sum().clamp_min(1.0)
        loss_mse = loss_mse / denom

        if args.loss in ('sisdr','both'):
            try:
                beam_spec_denorm = beam_spec * scale
                ref_spec_denorm = ref_spec
                est_wav = torch.istft(torch.view_as_complex(beam_spec_denorm.permute(0,3,2,1).contiguous()), n_fft=args.n_fft, hop_length=args.hop, win_length=(args.win_length or args.n_fft))
                ref_wav = torch.istft(torch.view_as_complex(ref_spec_denorm.permute(0,3,2,1).contiguous()), n_fft=args.n_fft, hop_length=args.hop, win_length=(args.win_length or args.n_fft))
                if est_wav.dim() == 1:
                    est_wav = est_wav.unsqueeze(0)
                if ref_wav.dim() == 1:
                    ref_wav = ref_wav.unsqueeze(0)
                minL = min(est_wav.shape[1], ref_wav.shape[1])
                est_wav = est_wav[:, :minL]
                ref_wav = ref_wav[:, :minL]
                sdr_val = si_sdr(est_wav, ref_wav)
                loss_sisdr = -sdr_val
            except Exception as e:
                print("[Warning] istft failure in train:", e)
                loss_sisdr = torch.tensor(0.0, device=device)
        else:
            loss_sisdr = torch.tensor(0.0, device=device)

        if args.loss == 'mse':
            loss = loss_mse
        elif args.loss == 'sisdr':
            loss = loss_sisdr
        else:
            loss = loss_mse + args.sisdr_weight * loss_sisdr

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip_grad)
        opt.step()
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        total_loss += float(loss.item()) * B
        total_items += B
        pbar.set_postfix_str(f"loss={total_loss/total_items:.6f}")
    return total_loss / max(1, total_items)

@torch.no_grad()
def validate(encoder, loader, device, args):
    encoder.eval()
    tot_loss = 0.0
    tot_items = 0
    for mix_spec_multi, ref_spec, lengths, _, _ in tqdm(loader, desc="Validate"):
        mix_spec_multi = mix_spec_multi.to(device)
        ref_spec = ref_spec.to(device)
        scale = compute_mix_scale_from_multispec(mix_spec_multi)
        mix_spec_multi_norm = mix_spec_multi / (scale.unsqueeze(1) + 1e-12)
        ref_spec_norm = ref_spec / (scale + 1e-12)
        extra_feat = None
        if (args.features_destination != 'none') and (args.encoder_type in ('features','both')):
            cond_feat = make_cond_features_from_multispec(mix_spec_multi_norm)
            extra_feat = cond_feat.to(device)
        if extra_feat is not None:
            emb, weights, beam_spec = encoder(mix_spec_multi_norm, extra_feat=extra_feat)
        else:
            emb, weights, beam_spec = encoder(mix_spec_multi_norm)
        mag = torch.sqrt(ref_spec_norm[:, 0:1, :, :]**2 + ref_spec_norm[:, 1:2, :, :]**2 + 1e-12)
        valid_frame_mask = (mag.squeeze(1).abs().sum(dim=-1) > 0).float()
        mask = valid_frame_mask.unsqueeze(1).unsqueeze(-1)
        mask_full = mask.expand(-1, beam_spec.shape[1], -1, beam_spec.shape[3])
        loss_mse = nn.functional.mse_loss(beam_spec * mask_full, ref_spec_norm * mask_full, reduction='sum') / mask_full.sum().clamp_min(1.0)
        if args.loss in ('sisdr','both'):
            try:
                est_wav = torch.istft(torch.view_as_complex((beam_spec*scale).permute(0,3,2,1).contiguous()), n_fft=args.n_fft, hop_length=args.hop, win_length=(args.win_length or args.n_fft))
                ref_wav = torch.istft(torch.view_as_complex(ref_spec.permute(0,3,2,1).contiguous()), n_fft=args.n_fft, hop_length=args.hop, win_length=(args.win_length or args.n_fft))
                minL = min(est_wav.shape[1], ref_wav.shape[1])
                est_wav = est_wav[:, :minL]
                ref_wav = ref_wav[:, :minL]
                sdr_val = si_sdr(est_wav, ref_wav)
                loss_sisdr = -sdr_val
            except Exception as e:
                print("[Warning] istft fail validate:", e)
                loss_sisdr = torch.tensor(0.0, device=next(encoder.parameters()).device)
        else:
            loss_sisdr = torch.tensor(0.0, device=next(encoder.parameters()).device)
        if args.loss == 'mse':
            loss = loss_mse
        elif args.loss == 'sisdr':
            loss = loss_sisdr
        else:
            loss = loss_mse + args.sisdr_weight * loss_sisdr
        tot_loss += float(loss.item()) * mix_spec_multi.shape[0]
        tot_items += mix_spec_multi.shape[0]
    return tot_loss / max(1, tot_items)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True)
    p.add_argument('--val_dir', required=True)
    p.add_argument('--test_dir', required=True)
    p.add_argument('--out_dir', default='out')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-3)
    p.add_argument('--device', default='cuda')
    p.add_argument('--n_fft', type=int, default=512)
    p.add_argument('--hop', type=int, default=128)
    p.add_argument('--win_length', type=int, default=None)
    p.add_argument('--sr', type=int, default=16000)
    p.add_argument('--segment', type=float, default=4.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--embed_dim', type=int, default=64)
    p.add_argument('--base_ch', type=int, default=32)
    p.add_argument('--loss', choices=['mse','sisdr','both'], default='mse')
    p.add_argument('--sisdr_weight', type=float, default=1.0)
    p.add_argument('--clip_grad', type=float, default=1.0)
    p.add_argument('--features_destination', choices=['none','encoder','stft'], default='none')
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cpu') if args.device == 'cpu' else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    train_pairs = find_pairs_in_dir(args.train_dir)
    val_pairs = find_pairs_in_dir(args.val_dir)
    test_pairs = find_pairs_in_dir(args.test_dir)
    if len(train_pairs) == 0:
        raise RuntimeError("No train pairs found.")
    if len(test_pairs) == 0:
        raise RuntimeError("No test pairs found.")

    time_divisible = 2 ** args.depth
    train_dataset = PairedSTFTDataset(train_pairs, sr=args.sr, segment_len=args.segment,
                                      n_fft=args.n_fft, hop=args.hop, win_length=args.win_length,
                                      random_crop=True, spec_augment=False, random_gain=False,
                                      time_divisible=time_divisible)
    val_dataset = PairedSTFTDataset(val_pairs, sr=args.sr, segment_len=args.segment,
                                      n_fft=args.n_fft, hop=args.hop, win_length=args.win_length,
                                      random_crop=False, time_divisible=time_divisible)

    M_train = train_dataset.get_num_mics()
    M_val = val_dataset.get_num_mics()
    if M_val is not None and M_train is not None and M_val != M_train:
        print(f"Warning: train M={M_train} val M={M_val}. Using train M.")
    M = M_train if M_train is not None else (M_val if M_val is not None else 1)
    print(f"Detected M = {M}")

    collate = lambda b: pad_collate_fn(b, spec_augment=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=pad_collate_fn)

    feat_dim = 0
    if (args.features_destination != 'none'):
        feat_dim = cond_channel_count(M)

    encoder = UnetBeamformer(in_mics=M, embed_dim=args.embed_dim, base_ch=args.base_ch, depth=args.depth, feature_dim=feat_dim).to(device)
    init_weights_xavier(encoder)

    opt = torch.optim.AdamW(list(encoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    def lr_lambda(step):
        warm = 200
        if step < warm:
            return float(step) / float(max(1.0, warm))
        progress = float(step - warm) / float(max(1, total_steps - warm))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val = 1e9
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(encoder, opt, train_loader, device, epoch, args, scheduler=scheduler)
        print(f"[epoch {epoch}] train_loss={train_loss:.6f}")
        val_loss = validate(encoder, val_loader, device, args)
        print(f"[epoch {epoch}] val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.out_dir, 'ckpt_best.pth')
            torch.save({'encoder_state': encoder.state_dict(), 'opt_state': opt.state_dict(), 'args': vars(args)}, ckpt_path)
            print(f"Saved best checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
