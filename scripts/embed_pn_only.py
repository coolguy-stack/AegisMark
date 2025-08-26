import argparse, numpy as np, torch, torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

p=argparse.ArgumentParser()
p.add_argument("--encoder", required=True)
p.add_argument("--in_img", required=True)
p.add_argument("--out_img", required=True)
p.add_argument("--bits_out", required=True)
p.add_argument("--alpha", type=float, default=0.006)  # lower = more invisible (try 0.004–0.008)
p.add_argument("--save_jpeg_q", type=int, default=95) # use PNG if you want: set q<=0 to force PNG
a=p.parse_args()

# load PN bank from encoder
enc = torch.jit.load(a.encoder, map_location="cpu").eval()
sd = enc.state_dict()
P = None
for k in ("pn.P", "pn.P_patch", "pn.P_full", "P", "P_patch"):
    if k in sd:
        P = sd[k].float(); break
assert P is not None, "PN bank not found in encoder state_dict"
if P.dim()==4 and P.size(1)==1: P = P.squeeze(1)
bit_len, H, W = P.shape
# normalize PN per-bit
P = P / (P.view(bit_len,-1).std(dim=1, keepdim=True).clamp_min(1e-8).view(bit_len,1,1))

# image
img = Image.open(a.in_img).convert("RGB").resize((W,H))
x = TF.to_tensor(img).unsqueeze(0)  # 1x3xHxW

# texture mask (Laplacian magnitude on gray, normalized 0..1, smoothed)
k = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=torch.float32).view(1,1,3,3)
g = 0.2989*x[:,0:1]+0.5870*x[:,1:2]+0.1140*x[:,2:3]
g = F.pad(g,(1,1,1,1),mode="reflect")
hp = F.conv2d(g,k)
m = hp.abs()
m = (m - m.min()) / (m.max()-m.min()+1e-8)
m = F.avg_pool2d(m, 3, stride=1, padding=1)  # light smooth
m = m.clamp(0,1)

# bits
bits = (torch.rand(bit_len)>0.5).float()
np.save(a.bits_out, bits.numpy().astype(np.uint8))

s = (bits*2-1).view(1,bit_len,1,1)
r_pn = (s * P.unsqueeze(0)).sum(dim=1, keepdim=True)  # 1x1xHxW
r_pn = r_pn * m * a.alpha
r_rgb = r_pn.repeat(1,3,1,1)

xw = (x + r_rgb).clamp(0,1)
out = TF.to_pil_image(xw.squeeze(0))
if a.save_jpeg_q>0:
    out.save(a.out_img, quality=a.save_jpeg_q, optimize=True)
else:
    out.save(a.out_img)  # PNG
print({"bit_len": int(bit_len), "alpha": a.alpha})
