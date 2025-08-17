import argparse, json, torch, numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F

p=argparse.ArgumentParser()
p.add_argument('--decoder',required=True)
p.add_argument('--img',required=True)
p.add_argument('--bit_len',type=int,default=32)
p.add_argument('--bits',default='')
a=p.parse_args()

dec=torch.jit.load(a.decoder,'cpu').eval()
im=Image.open(a.img).convert('RGB')
im=im.resize((256,256), Image.BILINEAR)
x0=TF.to_tensor(im).unsqueeze(0)

def pad_reflect(x, pad):
    return F.pad(x,(pad,pad,pad,pad),mode='reflect')

def crop_shift(xp, dx, dy, size=256, pad=24):
    return xp[:,:,pad+dy:pad+dy+size,pad+dx:pad+dx+size]

def rescale(x, s):
    H,W=x.shape[-2:]
    nh,nw=int(H*s),int(W*s)
    y=F.interpolate(x, size=(nh,nw), mode='bilinear', align_corners=False)
    if nh<256 or nw<256:
        pad_t=(256-nh)//2; pad_b=256-nh-pad_t; pad_l=(256-nw)//2; pad_r=256-nw-pad_l
        y=F.pad(y,(pad_l,pad_r,pad_t,pad_b),mode='reflect')
    elif nh>256 or nw>256:
        y=F.interpolate(y, size=(256,256), mode='bilinear', align_corners=False)
    return y

shifts=[-16,-8,0,8,16]
scales=[0.85,0.9,1.0,1.1]
probs=[]
logits_all=[]
for s in scales:
    xs=rescale(x0, s)
    xp=pad_reflect(xs, 24)
    for dy in shifts:
        for dx in shifts:
            x=crop_shift(xp, dx, dy)
            with torch.no_grad():
                logits=dec(x)
                probs.append(torch.sigmoid(logits))
                logits_all.append(logits)

P=torch.stack(probs,0).mean(0).squeeze(0)
L=torch.stack(logits_all,0).mean(0).squeeze(0)
presence=float(L.abs().mean().item())
bits_hat=(P>0.5).float()

out={"presence":presence}
if a.bits:
    gt=torch.from_numpy(np.load(a.bits)).float().squeeze(0)
    out["bit_acc"]=float((bits_hat==gt).float().mean().item())
print(json.dumps(out, indent=2))
