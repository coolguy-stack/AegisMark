import argparse, glob, json, torch
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np, hashlib, torch.nn.functional as F

def highpass(x):
    k = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype=torch.float32,device=x.device).view(1,1,3,3)
    g = 0.2989*x[:,0:1]+0.5870*x[:,1:2]+0.1140*x[:,2:3]
    g = F.pad(g,(1,1,1,1),mode='reflect')
    return F.conv2d(g,k)

def pn_bank(bit_len,H,W,key):
    h = hashlib.sha256(key.encode()).digest()
    seed = int.from_bytes(h[:8],'big')%(2**32-1)
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((bit_len,H,W)).astype(np.float32)
    P -= P.mean(axis=(1,2),keepdims=True)
    P /= (P.std(axis=(1,2),keepdims=True)+1e-8)
    return torch.from_numpy(P)

p = argparse.ArgumentParser()
p.add_argument('--encoder', required=True)
p.add_argument('--val_glob', default='data/val/*.jpg')
p.add_argument('--bit_len', type=int, default=32)
p.add_argument('--key', default='aegis')
a = p.parse_args()

enc = torch.jit.load(a.encoder, map_location='cpu').eval()
files = sorted(glob.glob(a.val_glob))[:64]
H = W = 256

P_right = pn_bank(a.bit_len, H, W, a.key)
P_wrong = pn_bank(a.bit_len, H, W, a.key+'-wrong')

def presence_with_P(img, P):
    x = TF.to_tensor(Image.open(img).convert('RGB')).unsqueeze(0)
    hp = highpass(x).squeeze(0)               # [1,1,H,W] -> [1,H,W]
    hp = hp.expand(a.bit_len, H, W)           # [B=bit_len,H,W]
    corr = (hp * P).mean(dim=(1,2))           # [bit_len]
    return float(corr.abs().mean().item())

pos, neg_raw, neg_wrong = [], [], []
for f in files:
    x = TF.to_tensor(Image.open(f).convert('RGB')).unsqueeze(0)
    bits = (torch.rand(1,a.bit_len)>0.5).float()
    with torch.no_grad():
        xw,_ = enc(x, bits)
    TF.to_pil_image(xw.squeeze(0)).save('/tmp/_tmp.jpg')  # reuse pipeline

    pos.append(presence_with_P('/tmp/_tmp.jpg', P_right))
    neg_raw.append(presence_with_P(f, P_right))
    neg_wrong.append(presence_with_P('/tmp/_tmp.jpg', P_wrong))

def stats(v): import statistics as s; return {"mean":float(s.mean(v)), "std":float(s.pstdev(v))}
print(json.dumps({
  "presence_pos": stats(pos),
  "presence_neg_raw": stats(neg_raw),
  "presence_neg_wrongkey": stats(neg_wrong)
}, indent=2))
