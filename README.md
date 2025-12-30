# Project-for-training-a-UNet-style-beamformer-for-speech-separation
Audio enhancement 

An easy-to-train system for speech enhancement from multi-microphone recordings. The encoder-decoder is used by a U-Net–style Beamformer that (1) consumes complex STFT inputs and preserves frequency resolution by down/up-sampling only across time frames, and (2) can optionally condition on simple hand-crafted features (per-channel real/imag, IPD, ILD, LPS). The decoder is projected directly to a 2-channel complex STFT with a final 1×1 convolution (self.outc), i.e. spectral = self.outc(x), producing the enhanced complex spectrum. Models are trained using STFT-domain MSE and/or waveform SI-SDR losses.

# Unet-Beamformer Project

Project for training a UNet-style beamformer.
Key features:
- Manual conditioning features (ILD/IPD/etc) and learned features.
- Dataset trims STFT frames so frames % (2**depth) == 0 (prevents decoder mismatches).
- Model downsamples/upsamples only in time (frequency preserved).
- No internal shape padding/cropping in the model — dataset is authoritative.

## Install

Create a virtualenv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Example usage

Train (single pair folders):
python train.py \
  --train_dir /path/to/train_folder \
  --val_dir   /path/to/val_folder \
  --test_dir  /path/to/test_folder \
  --epochs 5 \
  --batch 2 \
  --depth 3 \
  --n_fft 512 --hop 128 --sr 16000
Run inference (uses checkpoint out/ckpt_best.pth if available):
python infer.py --ckpt out/ckpt_best.pth --test_dir /path/to/test_folder --out_dir out/test_enh
## Notes

The dataset will trim STFT frames to be divisible by 2**depth. If you prefer audio padding instead, modify paired_stft.py (option provided).

If shapes still mismatch, check --depth value; choose depth so frames % (2**depth) == 0 for your STFT frame settings (or let dataset trim automatically).
