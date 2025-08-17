import torch, torch.nn.functional as F, random, io
import torchvision.transforms.functional as TF
from PIL import Image

def jpeg_compress(x, quality=85):
    B,C,H,W = x.shape
    out=[]
    for i in range(B):
        buf=io.BytesIO()
        TF.to_pil_image(x[i].cpu()).save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        y=Image.open(buf).convert('RGB')
        out.append(TF.to_tensor(y))
    return torch.stack(out,0).to(x.device)

@torch.no_grad()
def random_crop_resize(x, min_s=0.85, max_s=0.95):
    B,C,H,W=x.shape
    ys=[]
    for i in range(B):
        s=random.uniform(min_s,max_s)
        h,w=int(H*s),int(W*s)
        top=random.randint(0,H-h)
        left=random.randint(0,W-w)
        crop=x[i:i+1,:,top:top+h,left:left+w]
        ys.append(F.interpolate(crop, size=(H,W), mode='bilinear', align_corners=False))
    return torch.cat(ys,0)

def gaussian_blur(x, k=3, sigma=1.2):
    import torchvision
    return torchvision.transforms.GaussianBlur(k, sigma)(x)

def add_noise(x, sigma=0.01):
    return torch.clamp(x+sigma*torch.randn_like(x),0,1)

@torch.no_grad()
def down_up(x, s=0.6):
    B,C,H,W=x.shape
    y=F.interpolate(x, size=(int(H*s),int(W*s)), mode='bilinear', align_corners=False)
    y=gaussian_blur(y, k=3, sigma=1.0)
    y=F.interpolate(y, size=(H,W), mode='bilinear', align_corners=False)
    return y

def attacks_easy(x):
    return jpeg_compress(x, quality=random.randint(88,95))

def attacks_medium(x):
    if random.random()<0.7: x=random_crop_resize(x, 0.85, 0.95)
    if random.random()<0.5: x=gaussian_blur(x, k=3, sigma=random.uniform(0.6,1.4))
    if random.random()<0.6: x=add_noise(x, sigma=random.uniform(0.005,0.015))
    if random.random()<0.8: x=jpeg_compress(x, quality=random.randint(80,92))
    return x

def attacks_hard(x):
    if random.random()<0.9: x=random_crop_resize(x, 0.75, 0.95)
    if random.random()<0.7: x=gaussian_blur(x, k=3, sigma=random.uniform(0.8,2.0))
    if random.random()<0.7: x=add_noise(x, sigma=random.uniform(0.005,0.02))
    if random.random()<0.9: x=jpeg_compress(x, quality=random.randint(60,90))
    if random.random()<0.5: x=down_up(x, s=random.uniform(0.5,0.8))
    return x

def attack_batch(x):
    return x
