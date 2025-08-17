import argparse, json, torch, numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('--encoder', required=True)
p.add_argument('--img', required=True)
p.add_argument('--bits', default='')
args = p.parse_args()

enc = torch.jit.load(args.encoder, 'cpu').eval()
sd  = enc.state_dict()

pn = None
pn_key = None
for k, t in sd.items():
    if k.endswith('.P'):
        pn = t.float()
        pn_key = k
        break
if pn is None:
    for suffix in ('.P_full', '.P_patch', 'P_full', 'P_patch', 'P'):
        for k, t in sd.items():
            if k.endswith(suffix):
                pn = t.float(); pn_key = k; break
        if pn is not None:
            break
if pn is None:
    raise SystemExit(f"PN buffer not found in encoder state_dict (first keys: {list(sd.keys())[:15]})")

if pn.dim() == 4 and pn.size(1) == 1:
    pn = pn.squeeze(1)
bit_len, H, W = pn.shape

pn = pn / (pn.view(bit_len, -1).std(dim=1, keepdim=True).clamp_min(1e-8).view(bit_len, 1, 1))

def highpass_gray(x):
    k = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=torch.float32).view(1,1,3,3)
    g = 0.2989*x[:,0:1] + 0.5870*x[:,1:2] + 0.1140*x[:,2:3]
    g = F.pad(g, (1,1,1,1), mode='reflect')
    return F.conv2d(g, k)

def pad_reflect(x, pad):
    return F.pad(x, (pad,pad,pad,pad), mode='reflect')

def crop_shift(xp, dx, dy, size, pad):
    return xp[:, :, pad+dy:pad+dy+size, pad+dx:pad+dx+size]

def rescale_to(x, s, size):
    nh, nw = int(size*s), int(size*s)
    y = F.interpolate(x, size=(nh,nw), mode='bilinear', align_corners=False)
    if nh < size or nw < size:
        pt = (size-nh)//2; pb = size-nh-pt
        pl = (size-nw)//2; pr = size-nw-pl
        y = F.pad(y, (pl,pr,pt,pb), mode='reflect')
    elif nh > size or nw > size:
        y = F.interpolate(y, size=(size,size), mode='bilinear', align_corners=False)
    return y

im = Image.open(args.img).convert('RGB').resize((W, H), Image.BILINEAR)
x0 = TF.to_tensor(im).unsqueeze(0)

pad = 32
shifts = list(range(-24, 25, 4))       # [-24, -20, ..., 20, 24]
scales = [0.75, 0.8, 0.85, 0.9, 1.0, 1.1]

def norm_corr(hp, pn):
    # hp: [1,H,W] after high-pass; pn: [B,H,W]
    num = (hp * pn).mean(dim=(1,2))                       # [B]
    den = hp.pow(2).mean().sqrt().clamp_min(1e-8)         # scalar
    return num / den                                      # [B], cosine-like

best_corr = None
for s in scales:
    xs = rescale_to(x0, s, H)
    xp = pad_reflect(xs, pad)
    for dy in shifts:
        for dx in shifts:
            x = crop_shift(xp, dx, dy, H, pad)
            hp = highpass_gray(x).squeeze(0)              # [1,H,W]
            corr = norm_corr(hp, pn)                      # [bit_len]
            if best_corr is None:
                best_corr = corr
            else:
                better = corr.abs() > best_corr.abs()
                best_corr = torch.where(better, corr, best_corr)

presence = float(best_corr.abs().mean().item())
bits_hat = (best_corr > 0).float()
out = {"presence": presence}
if args.bits:
    gt = torch.from_numpy(np.load(args.bits)).float().squeeze(0)
    out["bit_acc"] = float((bits_hat == gt).float().mean().item())
print(json.dumps(out, indent=2))
