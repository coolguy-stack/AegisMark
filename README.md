# AegisMark — Robust Invisible Watermarking

A practical system that embeds invisible watermarks into images and detects tampering. Works with common platform transforms like cropping, blur, and JPEG compression.

## What it does

**Embed** — Hides a 32-bit message invisibly in an image using:
- A small learned residual (neural network prediction)
- A deterministic pseudo-noise pattern per bit

**Detect** — Recovers the hidden bits and estimates watermark presence. Uses shift + scale search to handle crops, rescaling, blur, and JPEG.

**Tamper detection** — Flags images as tampered if watermark presence falls below threshold.

## Quick start

### Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt
```

### Data

Drop JPG/PNG images into:
```
data/train/
data/val/
```

Even 100-500 images works for testing.

### Train

```bash
python training/train.py \
  --train data/train --val data/val \
  --epochs 15 --batch_size 8 --image_size 256 \
  --bit_len 32 --lr 1e-3 \
  --lambda_rec 0.3 --lambda_bit 3.0 --lambda_adv 0.0 --lambda_tv 0.002 \
  --model_dir model_out
```

This creates `model_out/encoder.pt` and `model_out/decoder.pt`.

### Test watermark presence

```bash
python scripts/presence_eval.py --encoder model_out/encoder.pt --bit_len 32 --key aegis
```

Good results:
- Watermarked images: presence ≈ 0.15
- Wrong key: presence ≈ 0.003
- Clean images: presence ≈ 0.00007

Detection threshold around 0.02-0.03.

### Test robustness

```bash
python scripts/robust_eval.py --encoder model_out/encoder.pt --decoder model_out/decoder.pt --bit_len 32
```

Basic decoder gets ~82% bit accuracy on clean/JPEG, ~52% on crops/blur.

### Robust detection

Embed a watermark:
```bash
python scripts/embed_savebits.py \
  --encoder model_out/encoder.pt \
  --in_img data/val/test.jpg \
  --out_img test_wm.jpg \
  --bits_out test_bits.npy
```

Detect with shift/scale search:
```bash
python scripts/detect_pn_shift_scale.py \
  --encoder model_out/encoder.pt \
  --img test_wm.jpg \
  --bits test_bits.npy
```

Results:
- Clean: 100% bit accuracy
- 90% crop: 100% bit accuracy  
- 80% crop: 70-80% bit accuracy
- Blur: 100% bit accuracy

## How it works

The encoder adds two types of noise:
1. **Learned residual**: Neural network predicts optimal perturbation  
2. **PN pattern**: Fixed pseudo-noise pattern for each bit

The watermarking equation:
```
x_w = clip(x + r_learned + α · Σ_b s_b · P_b, 0, 1)
```

Where `s_b ∈ {-1,+1}` are the message bits, `P_b` are the pseudo-noise patterns, and `α` controls embedding strength.

For detection, correlate the image against the known PN patterns. The shift/scale search makes it robust to geometric transforms.

**Training loss:**
```
L = λ_rec · MSE(x_w, x) + λ_bit · BCE(logits, bits) + λ_tv · TV(r)
```

Where MSE preserves image quality, BCE ensures bit recovery, and TV regularizes the residual.

## Repository structure

```
training/        # Training code and models
scripts/         # Evaluation and detection scripts  
inference/       # SageMaker deployment
pipelines/       # AWS training pipelines
app/            # Lambda API
```

## Results

**Presence detection:**
- Watermarked: 0.151 ± 0.008
- Wrong key: 0.0031  
- Clean: 0.000066

**Bit recovery (robust detector):**
- Clean: 100%
- JPEG: 100%
- 90% crop: 100%
- 80% crop: 70-80%
- Gaussian blur: 100%

## Troubleshooting

- **venv issues**: `sudo apt install python3.12-venv`
- **TorchScript errors**: Check type hints in model.py
- **Training crashes**: Lower batch size or learning rate

## License

MIT

## References

Built on research from the [Awesome GenAI Watermarking](https://github.com/and-mill/Awesome-GenAI-Watermarking) collection.