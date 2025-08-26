import io, json, base64, torch
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import os

enc=None; P=None; H=W=None
WIN=None; KP=None; KPW=None; WINW=None
KP_PSEUDO=[]; KPW_PSEUDO=[]

def _center_fit(t, H, W):
    if t.dim()==3: t=t.unsqueeze(0)
    _,_,h,w=t.shape
    if h==H and w==W: return t
    if h<H or w<W:
        ph=max(H-h,0); pw=max(W-w,0)
        top=ph//2; bottom=ph-top; left=pw//2; right=pw-left
        return F.pad(t,(left,right,top,bottom),mode="reflect")
    top=(h-H)//2; left=(w-W)//2
    return t[:,:,top:top+H,left:left+W]

def _hp_gray_raw(x):
    g=0.2989*x[:,0:1]+0.5870*x[:,1:2]+0.1140*x[:,2:3]
    k=torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype=g.dtype,device=g.device).view(1,1,3,3)
    g=F.pad(g,(1,1,1,1),mode='reflect')
    return F.conv2d(g,k)

def _hp_gray_log(x,k=5,sigma=1.0):
    g=0.2989*x[:,0:1]+0.5870*x[:,1:2]+0.1140*x[:,2:3]
    ax=torch.arange(k,device=g.device,dtype=g.dtype)-(k-1)/2
    w=torch.exp(-(ax**2)/(2*sigma*sigma)); w=w/w.sum()
    kx=w.view(1,1,1,k); ky=w.view(1,1,k,1)
    g=F.pad(g,(k//2,k//2,0,0),mode='reflect'); g=F.conv2d(g,kx)
    g=F.pad(g,(0,0,k//2,k//2),mode='reflect'); g=F.conv2d(g,ky)
    lap=torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype=g.dtype,device=g.device).view(1,1,3,3)
    g=F.pad(g,(1,1,1,1),mode='reflect')
    return F.conv2d(g,lap)

def _pad_reflect(x,p): return F.pad(x,(p,p,p,p),mode='reflect')
def _crop_shift(xp,dx,dy,size,p): return xp[:,:,p+dy:p+dy+size,p+dx:p+dx+size]

def _rescale_to(x,s,size):
    nh,nw=int(size*s),int(size*s)
    y=F.interpolate(x,size=(nh,nw),mode='bilinear',align_corners=False)
    if nh<size or nw<size:
        pt=(size-nh)//2; pb=size-nh-pt; pl=(size-nw)//2; pr=size-nw-pl
        y=F.pad(y,(pl,pr,pt,pb),mode='reflect')
    elif nh>size or nw>size:
        y=F.interpolate(y,size=(size,size),mode='bilinear',align_corners=False)
    return y

def _build_bank(P,WIN):
    KP=(P*WIN).unsqueeze(1)
    KPW=(P.pow(2)*WIN).unsqueeze(1)
    return KP,KPW

def _rand_pn_like(shape,seed,dtype,device):
    g=torch.Generator().manual_seed(int(seed))
    x=torch.randn(shape, generator=g, dtype=dtype, device=device)
    x=x/(x.view(shape[0],-1).std(dim=1,keepdim=True).clamp_min(1e-8).view(shape[0],1,1))
    return x

def model_fn(model_dir):
    global enc,P,H,W,WIN,KP,KPW,WINW,KP_PSEUDO,KPW_PSEUDO
    enc=torch.jit.load(f"{model_dir}/encoder.pt",map_location="cpu").eval()
    sd=enc.state_dict()
    for k in ("pn.P","pn.P_patch","pn.P_full","P","P_patch"):
        if k in sd: P=sd[k].float(); break
    if P.dim()==4 and P.size(1)==1: P=P.squeeze(1)
    bit_len,H,W=P.shape
    P=P/(P.view(bit_len,-1).std(dim=1,keepdim=True).clamp_min(1e-8).view(bit_len,1,1))
    wy=torch.hann_window(H,periodic=False); wx=torch.hann_window(W,periodic=False)
    WIN=(wy.view(H,1)*wx.view(1,W)).to(P.dtype); WIN=WIN/WIN.mean()
    KP,KPW=_build_bank(P,WIN)
    WINW=WIN.view(1,1,H,W)
    KP_PSEUDO=[]; KPW_PSEUDO=[]
    seeds=os.environ.get("AEGIS_PSEUDO_SEEDS","1337,2027").split(",")
    for s in seeds:
        Q=_rand_pn_like((bit_len,H,W),s,P.dtype,P.device)
        kpp, kpw = _build_bank(Q,WIN)
        KP_PSEUDO.append(kpp); KPW_PSEUDO.append(kpw)
    return {"bit_len":bit_len,"H":H,"W":W}

def input_fn(request_body,content_type):
    payload=json.loads(request_body)
    b64=payload["image_base64"]
    img=Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB").resize((W,H))
    x=TF.to_tensor(img).unsqueeze(0)
    return {"x":x}

def predict_fn(inputs,_model):
    import torchvision.transforms.functional as TFv
    x0=inputs["x"]
    pad=64
    m0=torch.ones_like(x0[:,:1])

    def corr_map(hp_full,ma,KP_,KPW_,stride):
        num=F.conv2d(hp_full*ma,KP_,stride=stride)
        den_h=F.conv2d(hp_full.pow(2)*ma,WINW,stride=stride).sqrt().clamp_min(1e-8)
        den_p=F.conv2d(ma,KPW_,stride=stride).sqrt().clamp_min(1e-8)
        y=num/(den_h*den_p)
        pos=y.amax(dim=(2,3))
        neg=(-y).amax(dim=(2,3))
        return torch.where(pos>=neg,pos,-neg)

    def pass_once(scales,angles,stride,KP_,KPW_):
        best=None
        for s in scales:
            xs=_rescale_to(x0,s,H); ms=_rescale_to(m0,s,H)
            xp=_pad_reflect(xs,pad); mp=_pad_reflect(ms,pad)
            for ang in angles:
                xa=TFv.rotate(xp,ang,interpolation=TFv.InterpolationMode.BILINEAR,expand=False)
                ma=TFv.rotate(mp,ang,interpolation=TFv.InterpolationMode.BILINEAR,expand=False)
                hp1=_hp_gray_raw(xa); hp2=_hp_gray_log(xa,k=5,sigma=1.0)
                c1=corr_map(hp1,ma,KP_,KPW_,stride)
                c2=corr_map(hp2,ma,KP_,KPW_,stride)
                c=torch.where(c2.abs()>c1.abs(),c2,c1)
                best=c if best is None else torch.where(c.abs()>best.abs(),c,best)
        return best

    coarse_scales=[0.95,1.00,1.05]; coarse_angles=[0.0]
    best_main=pass_once(coarse_scales,coarse_angles,8,KP,KPW)
    best_pseudos=[]
    for i in range(len(KP_PSEUDO)):
        best_pseudos.append(pass_once(coarse_scales,coarse_angles,8,KP_PSEUDO[i],KPW_PSEUDO[i]))
    s0=1.0; a0=0.0
    refine_scales=[max(0.80,s0-0.06),s0,min(1.20,s0+0.06)]
    refine_angles=[a0-1.0,a0,a0+1.0]
    best_main=pass_once(refine_scales,refine_angles,2,KP,KPW)
    for i in range(len(KP_PSEUDO)):
        best_pseudos[i]=pass_once(refine_scales,refine_angles,2,KP_PSEUDO[i],KPW_PSEUDO[i])

    presence=float(best_main.abs().mean().item())
    presence_null=max(float(z.abs().mean().item()) for z in best_pseudos) if best_pseudos else 0.0
    margin=presence-presence_null
    bits=(best_main>0).int().tolist()
    return {"presence":presence,"presence_null":presence_null,"margin":margin,"bits":bits,"bit_len":len(bits)}

def output_fn(prediction,accept):
    return json.dumps(prediction)
