import torch, numpy as np, hashlib, torch.nn.functional as F

def _hp(x):
    k = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    g = 0.2989*x[:,0:1]+0.5870*x[:,1:2]+0.1140*x[:,2:3]
    g = F.pad(g,(1,1,1,1),mode='reflect')
    return F.conv2d(g,k)

def _pn(bit_len, h, w, key):
    hsh = hashlib.sha256(key.encode()).digest()
    seed = int.from_bytes(hsh[:8],'big')%(2**32-1)
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((bit_len,h,w)).astype(np.float32)
    P -= P.mean(axis=(1,2),keepdims=True)
    P /= (P.std(axis=(1,2),keepdims=True)+1e-8)
    return torch.from_numpy(P)

class SS:
    def __init__(self, bit_len=32, alpha=0.02, key="aegis"):
        self.bit_len=bit_len; self.alpha=alpha; self.key=key; self.P=None
    def ensure(self, x):
        B,C,H,W = x.shape
        if self.P is None or self.P.shape[-2:]!=(H,W):
            self.P = _pn(self.bit_len,H,W,self.key).to(x.device)
    def embed(self, x, bits):
        self.ensure(x)
        s = (bits*2-1).view(-1,self.bit_len,1,1)
        r = (s*self.P).sum(dim=1,keepdim=True).repeat(1,3,1,1)
        y = torch.clamp(x + self.alpha*r, 0, 1)
        return y
    def detect_logits(self, x):
        self.ensure(x)
        hp = _hp(x).repeat(1,self.bit_len,1,1)
        corr = (hp*self.P.unsqueeze(0)).mean(dim=(2,3))
        return corr

