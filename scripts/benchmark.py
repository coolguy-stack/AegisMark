#!/usr/bin/env python3
import argparse, os, time, json, random, io, csv, math, base64, pathlib
from datetime import datetime
from collections import defaultdict
import numpy as np
from PIL import Image, ImageFilter
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import inference.inference as inf
import torchvision.transforms.functional as TF
import torch

def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def load_img(path, size):
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize((size, size), Image.BILINEAR)
    return im

def to_tensor_for_embed(im):
    return TF.to_tensor(im).unsqueeze(0)

def psnr(ref_rgb: Image.Image, test_rgb: Image.Image) -> float:
    ref = np.asarray(ref_rgb, dtype=np.float32) / 255.0
    tst = np.asarray(test_rgb, dtype=np.float32) / 255.0
    mse = float(np.mean((ref - tst) ** 2))
    if mse <= 1e-12: return 99.0
    return 10.0 * math.log10(1.0 / mse)

def ssim_gray_simple(ref_rgb: Image.Image, test_rgb: Image.Image) -> float:
    x = np.asarray(ref_rgb.convert("L"), dtype=np.float32) / 255.0
    y = np.asarray(test_rgb.convert("L"), dtype=np.float32) / 255.0
    mu_x = x.mean(); mu_y = y.mean()
    vx = x.var(); vy = y.var()
    vxy = ((x - mu_x) * (y - mu_y)).mean()
    C1 = (0.01 ** 2); C2 = (0.03 ** 2)
    num = (2*mu_x*mu_y + C1) * (2*vxy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (vx + vy + C2)
    if den <= 0: return 1.0
    return float(num / den)

def run_encoder(enc, x, bits, alpha):
    try:
        out = enc(x, bits, alpha)
    except (TypeError, RuntimeError):
        out = enc(x, bits)
    return out[0] if isinstance(out, tuple) else out

def load_pn_from_encoder(encoder_path, device="cpu"):
    m = torch.jit.load(encoder_path, map_location=device).eval()
    sd = m.state_dict()
    P = None
    for k in ("pn.P","pn.P_patch","pn.P_full","P","P_patch"):
        if k in sd:
            P = sd[k].float()
            break
    if P is None:
        raise RuntimeError("PN bank not found in encoder.pt")
    if P.dim() == 4 and P.size(1) == 1:
        P = P.squeeze(1)
    B, H, W = P.shape
    P = P / (P.view(B, -1).std(dim=1, keepdim=True).clamp_min(1e-8).view(B,1,1))
    return P.to(device), H, W

def embed_tensor_pn(x, P, bits, alpha=0.009):
    s = (bits*2.0 - 1.0).view(1, -1, 1, 1)
    r = (s * P.unsqueeze(0)).sum(dim=1, keepdim=True)
    r3 = r.repeat(1, 3, 1, 1)
    y = (x + alpha * r3).clamp(0,1)
    return y

def detect_pil(pil_img, info, T_LO, T_HI, MARG):
    b = io.BytesIO(); pil_img.save(b, format="PNG")
    payload = {"image_base64": base64.b64encode(b.getvalue()).decode()}
    inputs = inf.input_fn(json.dumps(payload), "application/json")
    t0 = time.time()
    out = inf.predict_fn(inputs, info)
    dt_ms = (time.time() - t0) * 1000.0
    presence = float(out.get("presence", 0.0))
    presence_null = float(out.get("presence_null", 0.0))
    margin = float(out.get("margin", presence - presence_null))
    bits = out.get("bits", [])
    strong = (presence >= T_HI) and (margin >= MARG)
    weak = (not strong) and (presence >= T_LO) and (margin >= MARG * 0.5)
    conf = "strong" if strong else ("weak" if weak else "absent")
    present = strong or weak
    return present, conf, presence, presence_null, margin, bits, dt_ms

def attacked_variants(img: Image.Image):
    img = img.convert("RGB")
    W, H = img.size
    out = []
    out.append(("clean", img))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, subsampling=0, optimize=True)
    buf.seek(0)
    jpeg = Image.open(buf).convert("RGB").copy()
    out.append(("jpeg85", jpeg))
    out.append(("blur1p5", img.filter(ImageFilter.GaussianBlur(radius=1.5))))
    for s, tag in [(0.90, "crop90"), (0.80, "crop80")]:
        w, h = int(W*s), int(H*s)
        crop = img.crop(((W-w)//2, (H-h)//2, (W+w)//2, (H+h)//2)).resize((W, H), Image.BILINEAR)
        out.append((tag, crop))
    dnup = img.resize((W//2, H//2), Image.BICUBIC).resize((W, H), Image.BICUBIC)
    out.append(("resize_dn_up", dnup))
    out.append(("rot2", img.rotate(2, resample=Image.BILINEAR, expand=False)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out_root", default="runs")
    ap.add_argument("--encoder", default="model_out/encoder.pt")
    ap.add_argument("--bit_len", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.009)
    ap.add_argument("--embed_mode", choices=["pn","neural"], default="pn")
    ap.add_argument("--t_lo", type=float, default=float(os.environ.get("AEGIS_TLO", "0.027")))
    ap.add_argument("--t_hi", type=float, default=float(os.environ.get("AEGIS_THI", "0.055")))
    ap.add_argument("--t_margin", type=float, default=float(os.environ.get("AEGIS_MARGIN", "0.006")))
    args = ap.parse_args()

    random.seed(1337)
    info = inf.model_fn(os.path.dirname(args.encoder) or ".")
    src_paths = []
    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    for root, _, files in os.walk(args.src_dir):
        for f in files:
            if f.lower().endswith(exts):
                src_paths.append(os.path.join(root, f))
    if not src_paths:
        raise FileNotFoundError(f"No images under {args.src_dir}")
    random.shuffle(src_paths)
    src_paths = src_paths[:args.n]

    tag = f"bench_{now_tag()}"
    out_dir = os.path.join(args.out_root, tag)
    ensure_dir(out_dir)

    psnr_vals, ssim_vals = [], []
    rows = []
    case_total_ms = []
    detect_ms_accum = 0.0
    n_detect_calls = 0
    pos_margins = []
    neg_margins = []

    device = "cpu"
    enc = torch.jit.load(args.encoder, map_location=device).eval()
    if args.embed_mode == "pn":
        P, Hpn, Wpn = load_pn_from_encoder(args.encoder, device=device)

    for i, sp in enumerate(src_paths):
        case_dir = os.path.join(out_dir, f"case_{i:04d}")
        ensure_dir(case_dir)
        src_img = load_img(sp, args.size)
        x = to_tensor_for_embed(src_img)
        B = 1
        bits = torch.randint(0, 2, (B, args.bit_len), dtype=torch.float32)
        with torch.no_grad():
            if args.embed_mode == "pn":
                y = embed_tensor_pn(x, P, bits, alpha=args.alpha)
            else:
                y = run_encoder(enc, x, bits, alpha=args.alpha)
        wm_img = TF.to_pil_image(y[0].clamp(0,1))
        psnr_vals.append(psnr(src_img, wm_img))
        ssim_vals.append(ssim_gray_simple(src_img, wm_img))

        pos_set = dict(attacked_variants(wm_img))
        neg_set = dict(attacked_variants(src_img))

        t_case0 = time.time()
        for attk, img_p in pos_set.items():
            present, conf, pres, pnull, marg, bits_out, dt_ms = detect_pil(img_p, info, args.t_lo, args.t_hi, args.t_margin)
            rows.append({"file": os.path.relpath(sp, args.src_dir), "case": i, "attack": attk, "label": 1, "present": int(present), "confidence": conf, "presence": pres, "presence_null": pnull, "margin": marg, "latency_ms": dt_ms})
            pos_margins.append(marg)
            detect_ms_accum += dt_ms
            n_detect_calls += 1

        for attk, img_n in neg_set.items():
            present, conf, pres, pnull, marg, bits_out, dt_ms = detect_pil(img_n, info, args.t_lo, args.t_hi, args.t_margin)
            rows.append({"file": os.path.relpath(sp, args.src_dir), "case": i, "attack": attk, "label": 0, "present": int(present), "confidence": conf, "presence": pres, "presence_null": pnull, "margin": marg, "latency_ms": dt_ms})
            neg_margins.append(marg)
            detect_ms_accum += dt_ms
            n_detect_calls += 1

        case_total_ms.append((time.time() - t_case0) * 1000.0)

    by_attack = defaultdict(list)
    for r in rows:
        by_attack[(r["attack"], r["label"])].append(r)

    det_per_attack = {}
    fpr_per_attack = {}
    attacks = sorted({r["attack"] for r in rows})
    for a in attacks:
        pos = by_attack.get((a,1), [])
        neg = by_attack.get((a,0), [])
        if pos:
            det = sum(rr["present"] for rr in pos) / len(pos)
            strong = sum(1 for rr in pos if rr["confidence"] == "strong") / len(pos)
            det_per_attack[a] = {"det_rate": det, "strong_rate": strong, "n": len(pos)}
        if neg:
            fpr = sum(rr["present"] for rr in neg) / len(neg)
            fpr_per_attack[a] = {"fpr": fpr, "n": len(neg)}

    pos_arr = np.array(pos_margins, dtype=np.float32)
    neg_arr = np.array(neg_margins, dtype=np.float32)
    all_m = np.array(pos_margins + neg_margins, dtype=np.float32)
    thr = np.quantile(all_m, np.linspace(0, 1, 101))
    tpr = []; fpr = []
    for t in thr:
        tp = (pos_arr >= t).mean() if len(pos_arr) else 0.0
        fp = (neg_arr >= t).mean() if len(neg_arr) else 0.0
        tpr.append(tp); fpr.append(fp)
    fpr = np.array(fpr); tpr = np.array(tpr)
    order = np.argsort(fpr)
    auc = float(np.trapezoid(y=tpr[order], x=fpr[order]))

    csv_path = os.path.join(out_dir, "rows.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)

    summary = {
        "bit_len": args.bit_len,
        "size": args.size,
        "n_cases": len(src_paths),
        "psnr_mean": float(np.mean(psnr_vals)) if psnr_vals else None,
        "ssim_gray_mean": float(np.mean(ssim_vals)) if ssim_vals else None,
        "latency_ms_mean": float(np.mean(case_total_ms)) if case_total_ms else None,
        "detect_ms_mean": float(detect_ms_accum / max(1, n_detect_calls)),
        "t_lo": args.t_lo,
        "t_hi": args.t_hi,
        "t_margin": args.t_margin,
        "det_per_attack": det_per_attack,
        "fpr_per_attack": fpr_per_attack,
        "roc_auc_margin": auc,
    }
    sum_path = os.path.join(out_dir, "summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote: {sum_path}")
    print(f"wrote: {csv_path}")

    html = []
    html.append("<!doctype html><meta charset='utf-8'><title>AegisMark Benchmark</title>")
    html.append("""
    <style>
      body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#222}
      h1{margin:0 0 .5rem}
      .muted{color:#666}
      .grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
      table{border-collapse:collapse;width:100%}
      th,td{border:1px solid #e6e6e6;padding:.5rem .6rem;text-align:left}
      th{background:#fafafa}
      code{background:#f3f3f3;padding:.12rem .3rem;border-radius:4px}
      .card{background:#f9f9f9;border:1px solid #e7e7e7;border-radius:10px;padding:1rem}
      @media (max-width:900px){.grid{grid-template-columns:1fr}}
    </style>
    """)
    html.append("<h1>AegisMark Benchmark</h1>")
    html.append(f"<div class='muted'>Run: <code>{tag}</code> • cases={len(src_paths)} • size={args.size} • bit_len={args.bit_len} • embed_mode={args.embed_mode}</div>")
    html.append("<div class='grid' style='margin:1rem 0'>")
    html.append("<div class='card'><h3>Invisibility (clean only)</h3>")
    html.append("<table><tr><th>PSNR (dB)</th><th>SSIM (gray)</th></tr>")
    html.append(f"<tr><td>{summary['psnr_mean']:.3f}</td><td>{summary['ssim_gray_mean']:.4f}</td></tr></table></div>")
    html.append("<div class='card'><h3>Latency</h3>")
    html.append("<table><tr><th>Mean per case (ms)</th><th>Mean per detect (ms)</th></tr>")
    html.append(f"<tr><td>{summary['latency_ms_mean']:.1f}</td><td>{summary['detect_ms_mean']:.1f}</td></tr></table>")
    html.append(f"<div class='muted' style='margin-top:.5rem'>Thresholds: T_LO={summary['t_lo']:.3f}, T_HI={summary['t_hi']:.3f}, MARG={summary['t_margin']:.3f}</div>")
    html.append(f"<div class='muted'>ROC AUC (margin): {summary['roc_auc_margin']:.3f}</div></div>")
    html.append("</div>")
    def det_table(title, data):
        html.append(f"<div class='card' style='margin:1rem 0'><h3>{title}</h3><table><tr>")
        if title.startswith("Detection"): html.append("<th>Attack</th><th>Det. rate</th><th>Strong rate</th><th>N</th>")
        else: html.append("<th>Attack</th><th>FPR</th><th>N</th>")
        html.append("</tr>")
        for a in attacks:
            if a in data:
                d = data[a]
                if title.startswith("Detection"):
                    html.append(f"<tr><td>{a}</td><td>{d['det_rate']:.2f}</td><td>{d['strong_rate']:.2f}</td><td>{d['n']}</td></tr>")
                else:
                    html.append(f"<tr><td>{a}</td><td>{d['fpr']:.2f}</td><td>{d['n']}</td></tr>")
        html.append("</table></div>")
    det_table("Detection (positives)", summary["det_per_attack"])
    det_table("False Positive Rate (negatives)", summary["fpr_per_attack"])
    html.append("<div class='muted' style='margin:1rem 0'>Artifacts:</div><ul>")
    html.append(f"<li><a href='rows.csv'>rows.csv</a></li>")
    html.append(f"<li><a href='summary.json'>summary.json</a></li>")
    html.append("</ul>")
    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w") as f:
        f.write("".join(html))
    print(f"wrote: {html_path}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
