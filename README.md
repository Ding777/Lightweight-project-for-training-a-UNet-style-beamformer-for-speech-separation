# Lightweight-project-for-training-a-UNet-style-beamformer-for-speech-separation
Audio enhancement 


# Unet-Beamformer Project

Lightweight project for training a simple time-only UNet-style beamformer.
Key features:
- Manual conditioning features (ILD/IPD/etc).
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
