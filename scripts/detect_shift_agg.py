import argparse, json, torch
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--decoder', required=True)
p.add_argument('--img', required=True)
p.add_argument('--bit_len', type=int, default=32)
p.add_argument('--bits', default='')
a=p.parse_args()

dec=torch.jit.load(a.decoder,'cpu').eval()
im=Image.open(a.img).convert('RGB').resize((256,256))
x=TF.to_tensor(im).unsqueeze(0)

def shift(x, dx, dy):
    xpad=F.pad(x,(8,8,8,8),mode='reflect')
    return xpad[:,:,8+dy:8+dy+256,8+dx:8+dx+256]

shifts=[-8,-4,0,4,8]
probs=[]
with torch.no_grad():
    for dy in shifts:
        for dx in shifts:
            logits=dec(shift(x,dx,dy))
            probs.append(torch.sigmoid(logits))
P=torch.stack(probs,0).mean(0).squeeze(0)
bits_hat=(P>0.5).float()

out={"presence": float(P.abs().mean().item())}
if a.bits:
    gt=torch.from_numpy(np.load(a.bits)).float()
    out["bit_acc"]= float((bits_hat==gt).float().mean().item())
print(json.dumps(out, indent=2))
