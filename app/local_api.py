from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
import base64, io, os, json, hashlib
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

import inference.inference as inf

app = FastAPI()

INFO = None
ENC = None
P = None
BIT_LEN = None
H = W = None
ENC_SHA16 = None

T_LO  = float(os.environ.get("AEGIS_TLO",  "0.025"))
T_HI  = float(os.environ.get("AEGIS_THI",  "0.055"))
MARG  = float(os.environ.get("AEGIS_MARGIN","0.010"))

class DetectJSON(BaseModel):
    image_base64: str
    token: str | None = None

def _load_encoder(model_dir="model_out"):
    enc = torch.jit.load(f"{model_dir}/encoder.pt", map_location="cpu").eval()
    sd = enc.state_dict()
    P = None
    for k in ("pn.P","pn.P_patch","pn.P_full","P","P_patch"):
        if k in sd:
            P = sd[k].float()
            break
    if P is None:
        raise RuntimeError("PN pattern tensor not found in encoder")
    if P.dim() == 4 and P.size(1) == 1:
        P = P.squeeze(1)
    bit_len, H, W = P.shape
    P = P / (P.view(bit_len, -1).std(dim=1, keepdim=True).clamp_min(1e-8).view(bit_len,1,1))
    return enc, P, bit_len, H, W

def token_to_bits(token: str, bit_len: int, salt: str) -> np.ndarray:
    t = token.strip().encode("utf-8")
    s = salt.encode("ascii")
    need_bytes = (bit_len + 7) // 8
    buf = bytearray()
    ctr = 0
    seed = hashlib.sha256(t + b"|" + s).digest()
    while len(buf) < need_bytes:
        buf.extend(hashlib.sha256(seed + ctr.to_bytes(2, "big")).digest())
        ctr += 1
    arr = np.frombuffer(bytes(buf[:need_bytes]), dtype=np.uint8)
    bits = np.unpackbits(arr)[:bit_len]
    return bits.astype(np.uint8)

def bits_hex(bits_like) -> str:
    b = np.array(bits_like, dtype=np.uint8)
    return np.packbits(b).tobytes().hex()

@app.on_event("startup")
def _startup():
    global INFO, ENC, P, BIT_LEN, H, W, ENC_SHA16
    INFO = inf.model_fn("model_out")
    with open("model_out/encoder.pt","rb") as f:
        ENC_SHA16 = hashlib.sha256(f.read()).hexdigest()[:16]
    ENC, P, BIT_LEN, H, W = _load_encoder("model_out")

@app.get("/health")
def health():
    cfg = {k:v for k,v in os.environ.items() if k.startswith("AEGIS_")}
    return {
        "status":"ok",
        "bit_len": INFO.get("bit_len"),
        "H": INFO.get("H"),
        "W": INFO.get("W"),
        "t_lo": T_LO, "t_hi": T_HI, "margin": MARG,
        "encoder_sha16": ENC_SHA16,
        "cfg": cfg
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return Path("app/static/index.html").read_text(encoding="utf-8")

def _call_inference(img: Image.Image):
    b = io.BytesIO(); img.save(b, format="PNG")
    payload = {"image_base64": base64.b64encode(b.getvalue()).decode()}
    inputs = inf.input_fn(json.dumps(payload), "application/json")
    out = inf.predict_fn(inputs, INFO)
    presence      = float(out.get("presence", 0.0))
    presence_null = float(out.get("presence_null", 0.0))
    margin        = float(out.get("margin", presence - presence_null))
    bits          = out.get("bits", [])
    strong  = presence >= T_HI and margin >= MARG
    weak    = (not strong) and (presence >= T_LO) and (margin >= MARG*0.5)
    present = strong or weak
    conf    = "strong" if strong else ("weak" if weak else "absent")
    sig_hex = bits_hex(bits) if bits else ""
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
async def detect(file: UploadFile = File(...), token: str | None = Form(None)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    resp = _call_inference(img)
    if token and token.strip():
        exp_bits = token_to_bits(token, BIT_LEN, ENC_SHA16)
        got_bits = np.array(resp["bits"], dtype=np.uint8)
        if got_bits.size == exp_bits.size:
            acc = float((got_bits == exp_bits).mean())
            ham = int(np.count_nonzero(got_bits ^ exp_bits))
            resp["token"] = token.strip()
            resp["token_match_acc"] = acc
            resp["token_hamming"] = ham
            resp["token_sig_hex"] = bits_hex(exp_bits)
            resp["same_signature"] = (resp.get("signature_hex","") == resp["token_sig_hex"])
    return resp

@app.post("/detect_json")
def detect_json(body: DetectJSON):
    raw = base64.b64decode(body.image_base64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    resp = _call_inference(img)
    if body.token and body.token.strip():
        exp_bits = token_to_bits(body.token, BIT_LEN, ENC_SHA16)
        got_bits = np.array(resp["bits"], dtype=np.uint8)
        if got_bits.size == exp_bits.size:
            acc = float((got_bits == exp_bits).mean())
            ham = int(np.count_nonzero(got_bits ^ exp_bits))
            resp["token"] = body.token.strip()
            resp["token_match_acc"] = acc
            resp["token_hamming"] = ham
            resp["token_sig_hex"] = bits_hex(exp_bits)
            resp["same_signature"] = (resp.get("signature_hex","") == resp["token_sig_hex"])
    return resp

def _embed_pn_pil(img: Image.Image, alpha: float, bits_np: np.ndarray | None):
    img_r = img.resize((W, H), Image.BICUBIC).convert("RGB")
    x = TF.to_tensor(img_r).unsqueeze(0)
    if bits_np is None:
        bits_np = np.random.randint(0, 2, size=(BIT_LEN,), dtype=np.uint8)
    sgn = torch.from_numpy((bits_np * 2 - 1).astype(np.int8)).to(dtype=torch.float32).view(BIT_LEN,1,1)
    R = (P * sgn).sum(dim=0, keepdim=False)
    R = R / R.std().clamp_min(1e-8)
    w = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)
    a = w / (w @ w)
    R3 = torch.stack([a[0]*R, a[1]*R, a[2]*R], dim=0).unsqueeze(0)
    y = torch.clamp(x + float(alpha) * R3, 0.0, 1.0)
    y_pil = TF.to_pil_image(y.squeeze(0))
    return y_pil, bits_np

@app.post("/encode")
async def encode(
    file: UploadFile = File(...),
    alpha: float = Form(0.0085),
    jpeg_q: int = Form(95),
    token: str | None = Form(None)
):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    bits_np = None
    token_sig_hex = ""
    tok = (token or "").strip()
    if tok:
        bits_np = token_to_bits(tok, BIT_LEN, ENC_SHA16)
        token_sig_hex = bits_hex(bits_np)
    y_pil, bits_used = _embed_pn_pil(img, float(alpha), bits_np)
    bio = io.BytesIO()
    y_pil.save(bio, format="JPEG", quality=int(jpeg_q), optimize=True)
    b64 = base64.b64encode(bio.getvalue()).decode()
    return {
        "image_base64": b64,
        "alpha": float(alpha),
        "jpeg_q": int(jpeg_q),
        "bit_len": int(BIT_LEN),
        "bits": bits_used.tolist(),
        "signature_hex": bits_hex(bits_used),
        "token": tok if tok else None,
        "token_sig_hex": token_sig_hex if tok else None
    }
