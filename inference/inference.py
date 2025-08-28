# inference/inference.py  — two-pass conv search (coarse → refine), robust & fast-enough

import io, json, base64, os, torch
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F

# Globals
enc = None
P = None
H = W = None
WIN = None
WINW = None
KP = None
KPW = None
KP_PSEUDO = []
KPW_PSEUDO = []

# --------- helpers ---------

def _hp_gray_raw(x):
    g = 0.2989*x[:,0:1] + 0.5870*x[:,1:2] + 0.1140*x[:,2:3]
    k = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=g.dtype, device=g.device).view(1,1,3,3)
    g = F.pad(g, (1,1,1,1), mode='reflect')
    return F.conv2d(g, k)  # [B,1,H,W]

def _hp_gray_log(x, k=5, sigma=1.0):
    g = 0.2989*x[:,0:1] + 0.5870*x[:,1:2] + 0.1140*x[:,2:3]
    ax = torch.arange(k, device=g.device, dtype=g.dtype) - (k - 1)/2
    w  = torch.exp(-(ax**2) / (2*sigma*sigma)); w = w / w.sum()
    kx = w.view(1,1,1,k); ky = w.view(1,1,k,1)
    g  = F.pad(g, (k//2, k//2, 0, 0), mode='reflect'); g = F.conv2d(g, kx)
    g  = F.pad(g, (0, 0, k//2, k//2), mode='reflect'); g = F.conv2d(g, ky)
    lap= torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=g.dtype, device=g.device).view(1,1,3,3)
    g  = F.pad(g, (1,1,1,1), mode='reflect')
    return F.conv2d(g, lap)  # [B,1,H,W]

def _pad_reflect(x, p): return F.pad(x, (p,p,p,p), mode='reflect')

def _rescale_to(x, s, size):
    nh, nw = int(size*s), int(size*s)
    y = F.interpolate(x, size=(nh,nw), mode='bilinear', align_corners=False)
    if nh < size or nw < size:
        pt = (size-nh)//2; pb = size-nh-pt
        pl = (size-nw)//2; pr = size-nw-pl
        y = F.pad(y, (pl,pr,pt,pb), mode='reflect')
    elif nh > size or nw > size:
        y = F.interpolate(y, size=(size,size), mode='bilinear', align_corners=False)
    return y

def _build_bank(P, WIN):
    KP  = (P * WIN).unsqueeze(1)        # [B,1,H,W]
    KPW = (P.pow(2) * WIN).unsqueeze(1) # [B,1,H,W]   (WIN not squared)
    return KP, KPW

def _rand_pn_like(shape, seed, dtype, device):
    g = torch.Generator().manual_seed(int(seed))
    x = torch.randn(shape, generator=g, dtype=dtype, device=device)
    x = x / (x.view(shape[0], -1).std(dim=1, keepdim=True).clamp_min(1e-8).view(shape[0],1,1))
    return x

# --------- model/io ---------

def model_fn(model_dir):
    global enc, P, H, W, WIN, WINW, KP, KPW, KP_PSEUDO, KPW_PSEUDO
    enc = torch.jit.load(f"{model_dir}/encoder.pt", map_location="cpu").eval()
    sd  = enc.state_dict()
    for k in ("pn.P", "pn.P_patch", "pn.P_full", "P", "P_patch"):
        if k in sd:
            P = sd[k].float()       # [B,H,W] or [B,1,H,W]
            break
    if P.dim() == 4 and P.size(1) == 1:
        P = P.squeeze(1)
    bit_len, H, W = P.shape
    P = P / (P.view(bit_len, -1).std(dim=1, keepdim=True).clamp_min(1e-8).view(bit_len,1,1))

    wy = torch.hann_window(H, periodic=False)
    wx = torch.hann_window(W, periodic=False)
    WIN = (wy.view(H,1) * wx.view(1,W)).to(P.dtype)
    WIN = WIN / WIN.mean()
    WINW = WIN.view(1,1,H,W)

    KP, KPW = _build_bank(P, WIN)

    KP_PSEUDO, KPW_PSEUDO = [], []
    seeds = [s for s in os.environ.get("AEGIS_PSEUDO_SEEDS", "1337,2027").split(",") if s.strip()]
    for s in seeds:
        Q = _rand_pn_like((bit_len, H, W), s, P.dtype, P.device)
        kpp, kpw = _build_bank(Q, WIN)
        KP_PSEUDO.append(kpp); KPW_PSEUDO.append(kpw)

    return {"bit_len": bit_len, "H": H, "W": W}

def input_fn(request_body, content_type):
    payload = json.loads(request_body)
    b64 = payload["image_base64"]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB").resize((W, H))
    x = TF.to_tensor(img).unsqueeze(0)  # [1,3,H,W]
    return {"x": x}

# --------- detection (two-pass) ---------

def predict_fn(inputs, _model):
    import torchvision.transforms.functional as TFv
    x0 = inputs["x"]             # [1,3,H,W]
    pad = int(os.environ.get("AEGIS_PAD", "64"))

    # Coarse grid (fast)
    coarse_scales = [float(s) for s in os.environ.get("AEGIS_COARSE_SCALES", "0.95,1.00,1.05").split(",")]
    coarse_angles = [float(a) for a in os.environ.get("AEGIS_COARSE_ANGLES", "0.0").split(",")]
    stride_coarse = int(os.environ.get("AEGIS_STRIDE_COARSE", "8"))

    # Refinement around best coarse (still fast)
    refine_delta  = float(os.environ.get("AEGIS_REFINE_DELTA", "0.06"))
    refine_angles = [float(a) for a in os.environ.get("AEGIS_REFINE_ANGLES", "-1.0,0.0,1.0").split(",")]
    stride_refine = int(os.environ.get("AEGIS_STRIDE_REFINE", "2"))

    def corr_map(hp_full, m_full, KP_, KPW_, stride):
        # normalized correlation via conv; returns per-bit best magnitude over shifts
        num   = F.conv2d(hp_full * m_full, KP_,  stride=stride)   # [1,B,h,w]
        den_h = F.conv2d(hp_full.pow(2) * m_full, WINW, stride=stride).sqrt().clamp_min(1e-8)
        den_p = F.conv2d(m_full, KPW_, stride=stride).sqrt().clamp_min(1e-8)
        y = num / (den_h * den_p)
        pos = y.amax(dim=(2,3))          # [1,B]
        neg = (-y).amax(dim=(2,3))       # [1,B]
        return torch.where(pos >= neg, pos, -neg).squeeze(0)  # [B]

    def pass_once(scales, angles, stride, KP_, KPW_):
        best = None; best_params = (None, None)
        ones = torch.ones_like(x0[:, :1])
        for s in scales:
            xs = _rescale_to(x0, s, H); ms = _rescale_to(ones, s, H)
            xp = _pad_reflect(xs, pad); mp = _pad_reflect(ms, pad)
            for a in angles:
                xa = TFv.rotate(xp, a, interpolation=TFv.InterpolationMode.BILINEAR, expand=False)
                ma = TFv.rotate(mp, a, interpolation=TFv.InterpolationMode.BILINEAR, expand=False)
                hp1 = _hp_gray_raw(xa)
                hp2 = _hp_gray_log (xa, k=5, sigma=1.0)
                c1  = corr_map(hp1, ma, KP_, KPW_, stride)
                c2  = corr_map(hp2, ma, KP_, KPW_, stride)
                c   = torch.where(c2.abs() > c1.abs(), c2, c1)   # per-bit
                if best is None:
                    best = c; best_params = (s, a)
                else:
                    better = c.abs() > best.abs()
                    if better.any():
                        best = torch.where(better, c, best)
                        best_params = (s, a)
        return best, best_params

    # --- main PN: coarse then refine
    best_c, (s0, a0) = pass_once(coarse_scales, coarse_angles, stride_coarse, KP, KPW)
    if s0 is None: s0 = 1.0
    if a0 is None: a0 = 0.0
    refine_scales = [max(0.80, s0 - refine_delta), s0, min(1.20, s0 + refine_delta)]
    best_r, _     = pass_once(refine_scales, refine_angles, stride_refine, KP, KPW)
    best_main     = torch.where(best_r.abs() > best_c.abs(), best_r, best_c)  # [B]

    presence = float(best_main.abs().mean().item())
    bits     = (best_main > 0).int().tolist()

    # --- null presence: max over pseudo banks
    presence_null = 0.0
    for kpp, kpw in zip(KP_PSEUDO, KPW_PSEUDO):
        c_n, (sn, an) = pass_once(coarse_scales, coarse_angles, stride_coarse, kpp, kpw)
        if sn is None: sn = s0
        if an is None: an = a0
        refine_scales_n = [max(0.80, sn - refine_delta), sn, min(1.20, sn + refine_delta)]
        r_n, _ = pass_once(refine_scales_n, refine_angles, stride_refine, kpp, kpw)
        best_n = torch.where(r_n.abs() > c_n.abs(), r_n, c_n)
        presence_null = max(presence_null, float(best_n.abs().mean().item()))

    margin = presence - presence_null
    return {
        "presence": presence,
        "presence_null": presence_null,
        "margin": margin,
        "bits": bits,
        "bit_len": len(bits)
    }

def output_fn(prediction, accept):
    return json.dumps(prediction)
