import argparse, io, base64, torch, json
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F

p=argparse.ArgumentParser()
p.add_argument("--encoder", required=True)
p.add_argument("--img", required=True)
a=p.parse_args()

enc=torch.jit.load(a.encoder, map_location="cpu").eval()
sd=enc.state_dict()
P=None
for k in ("pn.P","pn.P_patch","pn.P_full","P","P_patch"):
    if k in sd: P=sd[k].float(); break
if P.dim()==4 and P.size(1)==1: P=P.squeeze(1)
B,H,W=P.shape
P=P/(P.view(B,-1).std(dim=1,keepdim=True).clamp_min(1e-8).view(B,1,1))

x=TF.to_tensor(Image.open(a.img).convert("RGB").resize((W,H))).unsqueeze(0)
hp_raw=F.conv2d(F.pad((0.2989*x[:,0:1]+0.5870*x[:,1:2]+0.1140*x[:,2:3]),(1,1,1,1),mode="reflect"),
                torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype=x.dtype).view(1,1,3,3))

wy=torch.hann_window(H,periodic=False)
wx=torch.hann_window(W,periodic=False)
WIN=(wy.view(H,1)*wx.view(1,W)).to(x.dtype)
WINm=WIN/WIN.mean()
mask=torch.ones(1,1,H,W,dtype=x.dtype)

def ncc_ref(hp,pn):
    num=(hp*pn*mask).sum(dim=(2,3))
    denh=(hp.pow(2)*mask).sum(dim=(2,3)).sqrt().clamp_min(1e-8)
    denp=(pn.pow(2)*mask).sum(dim=(2,3)).sqrt().clamp_min(1e-8)
    return (num/(denh*denp)).squeeze(0)

def ncc_vec_variant(hp, use_num_WIN, use_den_WIN, mask_interp="nearest"):
    num_k = (pn_WIN if use_num_WIN else pn).unsqueeze(1)
    den_p = (P.pow(2)*(WIN if use_den_WIN else 1)).unsqueeze(1)
    m = mask
    n = F.conv2d(hp*m, num_k).squeeze(0).squeeze(-1).squeeze(-1)
    dh=F.conv2d((hp.pow(2))*m, WIN.view(1,1,H,W) if use_den_WIN else torch.ones(1,1,H,W,dtype=x.dtype))
    dh=dh.sqrt().clamp_min(1e-8).squeeze(0).squeeze(-1).squeeze(-1)
    dp=F.conv2d(m, den_p).sqrt().clamp_min(1e-8).squeeze(0).squeeze(-1).squeeze(-1)
    pos=torch.maximum(n/(dh*dp), -(n/(dh*dp)))
    return pos

pn=P
pn_WIN=P*WINm

r_ref=ncc_ref(hp_raw, pn)
r_vec_P_only=ncc_vec_variant(hp_raw, False, False)
r_vec_numWIN_denWIN=ncc_vec_variant(hp_raw, True, True)
r_vec_numP_denWIN=ncc_vec_variant(hp_raw, False, True)
print(json.dumps({
  "presence_ref": float(r_ref.abs().mean().item()),
  "presence_vec_P_only": float(r_vec_P_only.abs().mean().item()),
  "presence_vec_numWIN_denWIN": float(r_vec_numWIN_denWIN.abs().mean().item()),
  "presence_vec_numP_denWIN": float(r_vec_numP_denWIN.abs().mean().item())
}, indent=2))
