from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import base64, io, os, json
from PIL import Image
import numpy as np
import inference.inference as inf
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI()
INFO = None
T_LO = float(os.environ.get("AEGIS_TLO", "0.025"))
T_HI = float(os.environ.get("AEGIS_THI", "0.055"))
MARG = float(os.environ.get("AEGIS_MARGIN", "0.010"))

class DetectJSON(BaseModel):
    image_base64: str

@app.on_event("startup")
def _startup():
    global INFO
    INFO = inf.model_fn("model_out")

@app.get("/health")
def health():
    return {"status":"ok","bit_len": INFO.get("bit_len"), "H": INFO.get("H"), "W": INFO.get("W"), "t_lo": T_LO, "t_hi": T_HI, "margin": MARG}

@app.get("/", response_class=HTMLResponse)
def home():
    return Path("app/static/index.html").read_text(encoding="utf-8")

def _flatten_bits(b):
    if isinstance(b, (list, tuple)):
        if len(b) > 0 and isinstance(b[0], (list, tuple, np.ndarray)):
            return [int(x) for x in b[0]]
        return [int(x) for x in b]
    return []

def _call_inference(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    payload = {"image_base64": base64.b64encode(buf.getvalue()).decode()}
    inputs = inf.input_fn(json.dumps(payload), "application/json")
    out = inf.predict_fn(inputs, INFO)
    presence = float(out.get("presence", 0.0))
    presence_null = float(out.get("presence_null", 0.0))
    margin = float(out.get("margin", presence - presence_null))
    bits = _flatten_bits(out.get("bits", []))
    strong = (presence >= T_HI) and (margin >= MARG)
    weak = (not strong) and (presence >= T_LO) and (margin >= MARG * 0.5)
    present = bool(strong or weak)
    try:
        sig_hex = np.packbits(np.array(bits, dtype=np.uint8)).tobytes().hex() if bits else ""
    except Exception:
        sig_hex = ""
    conf = "strong" if strong else ("weak" if weak else "absent")
    return {"present": present, "confidence": conf, "presence": presence, "presence_null": presence_null, "margin": margin, "bit_len": len(bits), "bits": bits, "signature_hex": sig_hex}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    return _call_inference(img)

@app.post("/detect_json")
def detect_json(body: DetectJSON):
    raw = base64.b64decode(body.image_base64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return _call_inference(img)
