#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnetBeamformer(nn.Module):
    def __init__(self,
                 in_mics: int,
                 embed_dim: int = 64,
                 base_ch: int = 64,
                 k_t: int = 1,
                 k_f: int = 3,
                 depth: int = 3,
                 feature_dim: int = 0):
        super().__init__()
        self.in_mics = int(in_mics)
        self.in_ch = self.in_mics * 2
        self.embed_dim = int(embed_dim)
        self.base_ch = int(base_ch)
        self.k_t = int(k_t)
        self.k_f = int(k_f)
        self.depth = max(1, int(depth))
        self.feature_dim = int(feature_dim) if feature_dim is not None else 0

        if self.feature_dim > 0:
            self.extra_proj = nn.Conv2d(self.feature_dim, self.base_ch, kernel_size=1)
        else:
            self.extra_proj = None

        self.input_conv = nn.Sequential(
            nn.Conv2d(self.in_ch + self.base_ch, self.base_ch,
                      kernel_size=(self.k_t, self.k_f), padding=(0, self.k_f // 2)),
            nn.GroupNorm(1, self.base_ch),
            nn.PReLU()
        )

        def make_res_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(3,3), padding=(1,1)),
                nn.GroupNorm(1, out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=(3,3), padding=(1,1)),
                nn.GroupNorm(1, out_ch),
            )

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.enc_skip_proj = nn.ModuleList()
        self.enc_out_chs = []

        ch = self.base_ch
        for i in range(self.depth):
            out_ch = ch * 2
            self.enc_blocks.append(make_res_block(ch, out_ch))
            self.enc_skip_proj.append(nn.Conv2d(ch, out_ch, kernel_size=1))
            self.downs.append(nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
                nn.GroupNorm(1, out_ch),
                nn.SiLU()
            ))
            self.enc_out_chs.append(out_ch)
            ch = out_ch

        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=(3,3), padding=(1,1)),
            nn.GroupNorm(1, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, kernel_size=(3,3), padding=(1,1)),
            nn.GroupNorm(1, ch),
            nn.SiLU()
        )

        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.dec_skip_proj = nn.ModuleList()

        for i in range(self.depth):
            up_ch = ch
            out_ch = max(1, ch // 2)
            self.up_convs.append(nn.ConvTranspose2d(up_ch, out_ch, kernel_size=(2,1), stride=(2,1)))
            self.dec_blocks.append(make_res_block(out_ch, out_ch))
            enc_skip_ch = self.enc_out_chs[self.depth - 1 - i]
            self.dec_skip_proj.append(nn.Conv2d(enc_skip_ch, out_ch, kernel_size=1))
            ch = out_ch

        self.out_conv = nn.Sequential(
            nn.Conv2d(ch, self.embed_dim, kernel_size=1),
            nn.GroupNorm(1, self.embed_dim),
            nn.PReLU()
        )
        self.aux_head = nn.Conv2d(self.embed_dim, 2, kernel_size=1)

        self.bf_gru_hidden = max(16, self.embed_dim)
        self.bf_gru = nn.GRU(input_size=self.embed_dim, hidden_size=self.bf_gru_hidden,
                             num_layers=1, batch_first=True, bidirectional=False)
        self.bf_linear = nn.Linear(self.bf_gru_hidden, self.in_mics)
        nn.init.zeros_(self.bf_linear.bias)
        nn.init.xavier_uniform_(self.bf_linear.weight)

    def forward(self, multi_spec, extra_feat=None):
        # accept (B, M, 2, T, F)
        if multi_spec.dim() == 5 and multi_spec.size(2) == 2:
            B, M, two, T, F = multi_spec.shape
            x_raw = multi_spec
            x_stack = multi_spec.view(B, M * 2, T, F)
        elif multi_spec.dim() == 5 and multi_spec.size(-1) == 2:
            B, mics, freq_num, seq_len, two = multi_spec.shape
            x_raw = multi_spec.permute(0, 1, 4, 3, 2).contiguous()
            x_stack = x_raw.view(B, mics * 2, seq_len, freq_num)
            B, M, two, T, F = x_raw.shape
        else:
            raise ValueError("SimpleBeamformer got unexpected multi_spec shape: " + str(tuple(multi_spec.shape)))

        if (extra_feat is not None) and (self.extra_proj is not None):
            if extra_feat.dim() != 4:
                raise ValueError("extra_feat must be (B, C, T, F)")
            proj_feat = self.extra_proj(extra_feat)
            fused = proj_feat
        else:
            fused = torch.zeros((B, self.base_ch, T, F), device=x_stack.device, dtype=x_stack.dtype)

        xc = torch.cat([x_stack, fused], dim=1)
        x = self.input_conv(xc)

        skips = []
        for i in range(self.depth):
            block = self.enc_blocks[i]
            h = block(x)
            skip = x
            if skip.shape[1] != h.shape[1]:
                skip = self.enc_skip_proj[i](skip)
            if skip.shape[2] != h.shape[2] or skip.shape[3] != h.shape[3]:
                skip = F.interpolate(skip, size=(h.shape[2], h.shape[3]), mode='nearest')
            x = h + skip
            skips.append(x)
            x = self.downs[i](x)

        x = self.bottleneck(x)

        for i in range(self.depth):
            skip = skips.pop()
            up = self.up_convs[i](x)
            skip_mapped = self.dec_skip_proj[i](skip)
            if up.shape[2] != skip_mapped.shape[2] or up.shape[3] != skip_mapped.shape[3]:
                skip_mapped = F.interpolate(skip_mapped, size=(up.shape[2], up.shape[3]), mode='nearest')
            x = up + skip_mapped
            x = self.dec_blocks[i](x)
            if x.shape[1] == up.shape[1]:
                x = x + up
            else:
                up_proj = nn.Conv2d(up.shape[1], x.shape[1], kernel_size=1).to(x.device)
                x = x + up_proj(up)

        emb = self.out_conv(x)  # (B, E, T_out, F_out)

        # Strict: dataset must ensure T_out == T and F_out == F
        T_target = x_raw.shape[3]
        F_target = x_raw.shape[4]
        if emb.shape[2] != T_target or emb.shape[3] != F_target:
            # Fail loudly — dataset should ensure divisibility; this helps debugging.
            raise RuntimeError(f"[SimpleBeamformer] embedding size mismatch: emb (T={emb.shape[2]},F={emb.shape[3]}) vs raw (T={T_target},F={F_target}). "
                               "Make sure dataset time_divisible (=2**depth) or adjust depth/n_fft/hop.")

        pooled_time = emb.mean(dim=-1).permute(0, 2, 1).contiguous()  # (B, T, E)
        gru_out, _ = self.bf_gru(pooled_time)
        logits = self.bf_linear(gru_out)
        weights = torch.softmax(logits, dim=-1)  # (B, T, M)

        real = x_raw[:, :, 0, :, :]  # (B, M, T, F)
        imag = x_raw[:, :, 1, :, :]

        if weights.shape[1] != real.shape[2]:
            raise RuntimeError(f"Weights time dim {weights.shape[1]} != input T {real.shape[2]}")

        w_perm = weights.permute(0, 2, 1).unsqueeze(-1)  # (B, M, T, 1)
        real_w = (real * w_perm).sum(dim=1)
        imag_w = (imag * w_perm).sum(dim=1)
        beam_spec = torch.stack([real_w, imag_w], dim=1)  # (B, 2, T, F)

        # final sanity check
        assert beam_spec.shape[2] == T_target and beam_spec.shape[3] == F_target

        return emb, weights, beam_spec

    def per_mic_embed(self, multi_spec):
        if multi_spec.dim() != 5:
            raise ValueError("per_mic_embed expects multi_spec shape (B,M,2,T,F)")
        B, M, two, T, F = multi_spec.shape
        x = multi_spec.view(B * M, two, T, F)
        proj = nn.Conv2d(2, self.embed_dim, kernel_size=1).to(multi_spec.device)
        feat = proj(x)
        feat = feat.view(B, M, self.embed_dim, T, F)
        return feat
