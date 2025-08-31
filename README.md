# AegisMark — Robust Invisible Watermarking

**A practical system for embedding and detecting invisible watermarks in images**

Born from running a meme page and watching content get stolen daily, AegisMark started as a weekend project to create invisible watermarks that survive the real world of social media compression and cropping. What began as protecting memes evolved into a robust watermarking system for any digital content.

**Fun fact**: The first successful detection was on a heavily JPEG-compressed meme that had been screenshot, cropped, and reposted three times.

AegisMark embeds invisible watermarks that survive common image transformations like JPEG compression, blur, small rotations, rescaling, and moderate crops. The system includes a local web application and comprehensive benchmarking tools.

## ✨ Features

- **Invisible watermarking** with excellent perceptual quality (PSNR ≥ 44 dB, SSIM ≥ 0.99)
- **Robust detection** after JPEG, blur, resize, rotation, and cropping
- **Owner tokens** for cryptographic ownership verification
- **Web interface** for easy encoding/decoding
- **Comprehensive benchmarking** with HTML reports
- **Fast performance** with optimized correlation-based detection

## 🔧 How It Works

AegisMark uses pseudo-noise (PN) pattern embedding with a sophisticated two-pass detection system:

### Encoding
- Generates bit-specific PN patterns stored in `encoder.pt`
- Embeds watermark as luminance-balanced RGB residual
- Supports 32-bit payloads (configurable)
- Owner tokens derive deterministic signatures via SHA-256

### Detection
- Two-pass convolution search (coarse → refined)
- LoG/HP filtering with Hann windowing
- Content-adaptive null scoring for low false positives
- Outputs presence score, margin, and recovered bits

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- TorchScript encoder model (`encoder.pt`)

### Installation

```bash
# Clone and setup environment
git clone <repository-url>
cd aegismark
python3 -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt

# Place your encoder model
mkdir -p model_out
cp /path/to/encoder.pt model_out/encoder.pt

# Launch web app
uvicorn app.local_api:app --port 8000 --reload
```

Open http://127.0.0.1:8000 in your browser.

## 🖥️ Web Interface

### Encode Images
1. **Upload image** - Choose your source image
2. **Set owner token** (optional) - e.g., "alice-2025" for ownership verification
3. **Configure parameters**:
   - Alpha: 0.0085-0.010 (embedding strength)
   - JPEG quality: 80-95
4. **Download** watermarked image and note the signature

### Detect Watermarks
1. **Upload image** to analyze
2. **Verify ownership** (optional) - enter the same token used during encoding
3. **View results**:
   - Presence score and confidence (strong/weak/absent)
   - Recovered signature and bit accuracy
   - Owner verification status

### Benchmark Suite
- Upload multiple test images
- Automated testing across various attacks
- Generates comprehensive HTML reports with:
  - Invisibility metrics (PSNR, SSIM)
  - Detection rates per attack type
  - False positive rates
  - ROC curves and AUC scores

## 🛠️ CLI Usage

### Basic Embedding

```bash
python scripts/embed_pn_only.py \
  --encoder model_out/encoder.pt \
  --in_img tests/source.jpg \
  --out_img tests/watermarked.jpg \
  --bits_out tests/bits.npy \
  --alpha 0.009 --save_jpeg_q 95
```

### Batch Detection

```bash
# Start server first
uvicorn app.local_api:app --port 8000 --reload &

# Check multiple images
python scripts/quick_batch_check.py tests/watermarked*.jpg
```

### Full Benchmark

```bash
# Optional: optimize for CPU performance
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 AEGIS_PSEUDO_SEEDS=1337

python -u scripts/benchmark.py \
  --src_dir data/validation \
  --n 50 --size 256 \
  --embed_mode pn --alpha 0.009

# View results at: runs/bench_YYYYMMDD_HHMMSS/report.html
```

## 📁 Repository Structure

```
aegismark/
├── app/
│   ├── local_api.py        # FastAPI server
│   └── static/index.html   # Web UI
├── inference/
│   └── inference.py        # Core detection engine
├── scripts/
│   ├── embed_pn_only.py    # CLI embedding
│   ├── quick_batch_check.py # Batch detection
│   └── benchmark.py        # Comprehensive benchmarking
├── training/
│   ├── requirements.txt    # Dependencies
│   └── train_detector.py   # Experimental neural training
├── model_out/
│   └── encoder.pt         # PN pattern bank (required)
└── runs/                  # Benchmark outputs
```

## ⚡ API Reference

### Health Check
```http
GET /health
```
Returns system status, thresholds, and encoder hash.

### Encode Image
```http
POST /encode
Content-Type: multipart/form-data

file: image file
alpha: embedding strength (0.0085-0.010)
jpeg_q: JPEG quality (80-95)
token: owner token (optional)
```

### Detect Watermark
```http
POST /detect
Content-Type: multipart/form-data

file: image file
token: verification token (optional)
```

### Run Benchmark
```http
POST /benchmark
Content-Type: multipart/form-data

files[]: multiple image files
n: number of samples
size: resize dimension
alpha: embedding strength
embed_mode: "pn"
```

## ⚙️ Configuration

Customize detection via environment variables:

```bash
# Detection thresholds
export AEGIS_TLO=0.025     # Weak presence threshold
export AEGIS_THI=0.055     # Strong presence threshold  
export AEGIS_MARGIN=0.010  # Minimum margin

# Performance tuning
export AEGIS_PSEUDO_SEEDS=1337,2027  # Null bank seeds
export OMP_NUM_THREADS=4              # CPU threading
export MKL_NUM_THREADS=4
```

## 📊 Performance

**Invisibility**: PSNR ≥ 44 dB, SSIM ≥ 0.99 at α=0.009

**Robustness**: High detection rates for:
- JPEG compression (Q=70-95)
- Gaussian blur (σ ≤ 1.5)
- Scaling (0.8-1.2x)
- Rotation (±2°)
- Center crops (80%+)

**Speed**: ~50ms detection on CPU (256×256 images)

## 🔬 Technical Details

### Mathematical Foundation

**Embedding Process**

Given image `x ∈ [0,1]^(H×W×3)` and L-bit payload `b ∈ {0,1}^L` (default L=32):

1. **PN Pattern Generation**: Store bit-specific patterns `P_i ∈ R^(H×W)`, standardized and Hann-windowed

2. **Bit-to-Sign Mapping**: `s_i ∈ {-1,+1}` where `s_i = 2b_i - 1`

3. **Luminance-Balanced Embedding**: Using RGB weights `w = (0.2989, 0.5870, 0.1140)`:
   ```
   a = w / (w^T w)
   R = Σ(i=1 to L) s_i · P_i / std(P_i)  
   R_rgb = [a_0 R, a_1 R, a_2 R]
   x_w = clip(x + α R_rgb, 0, 1)
   ```

**Detection Process**

1. **Preprocessing**: Convert to grayscale high-pass using LoG (Laplacian of Gaussian) or HP (Laplacian)

2. **Two-Pass Convolution Search**:
   - **Coarse**: scales {0.95, 1.00, 1.05}, angle {0°}, stride 8
   - **Refine**: around best coarse result with Δs≈0.06, angles {-1°, 0°, 1°}, stride 2

3. **Normalized Correlation** (with Hann window W):
   ```
   y = (hp ⊙ m) * (P ⊙ W) / √[(hp² ⊙ m) * W · m * (P² ⊙ W)]
   ```

4. **Scoring**:
   - Per-bit logits from max correlation over shifts
   - **Presence** = mean absolute per-bit logit
   - **Presence_null** = max over pseudo PN banks (content-dependent null)
   - **Margin** = presence - presence_null

5. **Decision Rule**:
   - **Strong** if presence ≥ T_HI and margin ≥ MARG
   - **Weak** if presence ≥ T_LO and margin ≥ 0.5·MARG
   - **Absent** otherwise

## 🚧 Troubleshooting

**ModuleNotFoundError**: Ensure repository root is on PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/aegismark"
```

**Server won't detect**: Check `/health` endpoint and tune thresholds

**Token mismatch**: Tokens are case/whitespace sensitive

**Performance issues**: Set `AEGIS_PSEUDO_SEEDS=1337` for faster demos

## 🔮 Future Work

- **Neural encoder/decoder**: End-to-end learning with ECC
- **Enhanced robustness**: Advanced geometric transforms
- **Larger payloads**: 96-128 channel bits with error correction
- **Mobile optimization**: Quantized models for edge deployment

## 📄 License

MIT License - see LICENSE file for details.



## 🙏 Acknowledgments

Built on decades of spread-spectrum watermarking research. Special thanks to the computer vision and security communities for foundational work on robust image watermarking.

**Resources used:**
- [Awesome GenAI Watermarking](https://github.com/raywzy/Awesome-GenAI-Watermarking) - Comprehensive community collection of watermarking research and techniques

---

**Getting Started?** Try the web interface first, then explore the CLI tools and benchmarking system. For production use, tune thresholds on your specific image dataset and attack models.
