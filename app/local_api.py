from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
import base64, io, os, json
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

import inference.inference as inf

app = FastAPI()

INFO = None
ENC  = None
P    = None
BIT_LEN = None
H = W = None

T_LO  = float(os.environ.get("AEGIS_TLO",  "0.025"))
T_HI  = float(os.environ.get("AEGIS_THI",  "0.055"))
MARG  = float(os.environ.get("AEGIS_MARGIN","0.010"))

class DetectJSON(BaseModel):
    image_base64: str

def _load_encoder(model_dir="model_out"):
    enc = torch.jit.load(f"{model_dir}/encoder.pt", map_location="cpu").eval()
    sd = enc.state_dict()
    P = None
    for k in ("pn.P", "pn.P_patch", "pn.P_full", "P", "P_patch"):
        if k in sd:
            P = sd[k].float()
            break
    if P is None:
        raise RuntimeError("Could not find PN pattern tensor in encoder state_dict")
    if P.dim() == 4 and P.size(1) == 1:
        P = P.squeeze(1)
    bit_len, H, W = P.shape
    P = P / (P.view(bit_len, -1).std(dim=1, keepdim=True).clamp_min(1e-8).view(bit_len,1,1))
    return enc, P, bit_len, H, W

@app.on_event("startup")
def _startup():
    global INFO, ENC, P, BIT_LEN, H, W
    INFO = inf.model_fn("model_out")
    ENC, P, BIT_LEN, H, W = _load_encoder("model_out")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "bit_len": INFO.get("bit_len"),
        "H": INFO.get("H"),
        "W": INFO.get("W"),
        "t_lo": T_LO,
        "t_hi": T_HI,
        "margin": MARG
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return Path("app/static/index.html").read_text(encoding="utf-8")

def _call_inference(img: Image.Image):
    b = io.BytesIO()
    img.save(b, format="PNG")
    payload = {"image_base64": base64.b64encode(b.getvalue()).decode()}
    inputs = inf.input_fn(json.dumps(payload), "application/json")
    out = inf.predict_fn(inputs, INFO)

    presence      = float(out.get("presence", 0.0))
    presence_null = float(out.get("presence_null", 0.0))
    margin        = float(out.get("margin", 0.0))
    bits          = out.get("bits", [])

    strong  = presence >= T_HI and margin >= MARG
    weak    = (not strong) and (presence >= T_LO) and (margin >= MARG * 0.5)
    present = strong or weak
    conf    = "strong" if strong else ("weak" if weak else "absent")

    try:
        sig_hex = np.packbits(bits).tobytes().hex()
    except Exception:
        sig_hex = ""

    return {
        "present": present,
        "confidence": conf,
        "presence": presence,
        "presence_null": presence_null,
        "margin": margin,
        "bit_len": out.get("bit_len"),
        "bits": bits,
        "signature_hex": sig_hex
    }

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    return _call_inference(img)

@app.post("/detect_json")
def detect_json(body: DetectJSON):
    raw = base64.b64decode(body.image_base64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return _call_inference(img)

def _embed_pn_pil(img: Image.Image, alpha: float, bits: np.ndarray | None):
    img_r = img.resize((W, H), Image.BICUBIC).convert("RGB")
    x = TF.to_tensor(img_r).unsqueeze(0)

    if bits is None:
        bits = np.random.randint(0, 2, size=(BIT_LEN,), dtype=np.uint8)
    sgn = torch.from_numpy((bits * 2 - 1).astype(np.int8)).to(dtype=torch.float32).view(BIT_LEN, 1, 1)
    R = (P * sgn).sum(dim=0, keepdim=False)
    R = R / R.std().clamp_min(1e-8)

    w = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)
    a = w / (w @ w)

    R3 = torch.stack([a[0]*R, a[1]*R, a[2]*R], dim=0).unsqueeze(0)
    y = torch.clamp(x + alpha * R3, 0.0, 1.0)

    y_pil = TF.to_pil_image(y.squeeze(0))
    return y_pil, bits

@app.post("/encode")
async def encode(
    file: UploadFile = File(...),
    alpha: float = Form(0.0085),
    jpeg_q: int = Form(95)
):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    y_pil, bits = _embed_pn_pil(img, float(alpha), None)

    bio = io.BytesIO()
    y_pil.save(bio, format="JPEG", quality=int(jpeg_q), optimize=True)
    b64 = base64.b64encode(bio.getvalue()).decode()

    try:
        sig_hex = np.packbits(bits).tobytes().hex()
    except Exception:
        sig_hex = ""

    return {
        "image_base64": b64,
        "alpha": float(alpha),
        "jpeg_q": int(jpeg_q),
        "bit_len": int(BIT_LEN),
        "bits": bits.tolist(),
        "signature_hex": sig_hex
    }
