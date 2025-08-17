import argparse, glob, json, random, io, torch
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
import torch.nn.functional as F

p=argparse.ArgumentParser()
p.add_argument('--encoder', required=True)
p.add_argument('--decoder', required=True)
p.add_argument('--bit_len', type=int, default=32)
p.add_argument('--val_glob', default='data/val/*.jpg')
a=p.parse_args()

enc = torch.jit.load(a.encoder,'cpu').eval()
dec = torch.jit.load(a.decoder,'cpu').eval()
files = sorted(glob.glob(a.val_glob))[:40]

def jpeg(im, q):
    buf=io.BytesIO(); im.save(buf, format='JPEG', quality=q); buf.seek(0)
    return Image.open(buf).convert('RGB')

def crop_resize(im, s):
    W,H = im.size; w,h = int(W*s), int(H*s)
    left,top = (W-w)//2, (H-h)//2
    return im.crop((left,top,left+w,top+h)).resize((W,H), Image.BILINEAR)

def blur(im, rad):
    return im.filter(ImageFilter.GaussianBlur(radius=rad))

attacks = [
    ('jpeg95', lambda im: jpeg(im,95)),
    ('jpeg85', lambda im: jpeg(im,85)),
    ('jpeg75', lambda im: jpeg(im,75)),
    ('crop0.9', lambda im: crop_resize(im,0.9)),
    ('crop0.8', lambda im: crop_resize(im,0.8)),
    ('blur1.5', lambda im: blur(im,1.5)),
]

def acc_on(im):
    x = TF.to_tensor(im).unsqueeze(0)
    bits = (torch.rand(1,a.bit_len)>0.5).float()
    with torch.no_grad():
        xw,_ = enc(x, bits); logits = dec(xw)
        acc0 = ((torch.sigmoid(logits)>0.5)==bits).float().mean().item()
    imw = TF.to_pil_image(xw.squeeze(0))
    scores = {}
    for name,atk in attacks:
        imatk = atk(imw)
        xa = TF.to_tensor(imatk).unsqueeze(0)
        with torch.no_grad():
            acc = ((torch.sigmoid(dec(xa))>0.5)==bits).float().mean().item()
        scores[name]=acc
    return acc0, scores

res={}
for f in files:
    im = Image.open(f).convert('RGB').resize((256,256))
    clean, sc = acc_on(im)
    for k,v in sc.items():
        res.setdefault(k, []).append(v)
    res.setdefault('clean', []).append(clean)

out={k: sum(v)/len(v) for k,v in res.items()}
print(json.dumps(out, indent=2))
