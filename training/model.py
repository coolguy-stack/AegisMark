import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, hashlib

class ConvBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.net(x)

class HighPass(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False)
        k = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=torch.float32)
        w = torch.zeros(3,1,3,3)
        w[0,0]=k; w[1,0]=k; w[2,0]=k
        with torch.no_grad():
            self.conv.weight.copy_(w)
        for p in self.conv.parameters(): p.requires_grad = False
    def forward(self, x):
        return self.conv(x)

class PNBank(nn.Module):
    def __init__(self, bit_len, H, W, key="aegis"):
        super().__init__()
        self.bit_len=bit_len
        h = hashlib.sha256(key.encode()).digest()
        seed = int.from_bytes(h[:8],'big')%(2**32-1)
        rng = np.random.default_rng(seed)
        P = rng.standard_normal((bit_len, H, W)).astype(np.float32)
        P -= P.mean(axis=(1,2), keepdims=True)
        P /= (P.std(axis=(1,2), keepdims=True)+1e-8)
        self.register_buffer('P', torch.from_numpy(P))
    def forward(self):
        return self.P

class Encoder(nn.Module):
    def __init__(self, img_ch=3, bit_len=32, H=256, W=256, pn_alpha=0.02, res_scale=0.02, key="aegis"):
        super().__init__()
        self.bit_len=bit_len; self.H=H; self.W=W; self.pn_alpha=pn_alpha; self.res_scale=res_scale
        self.pn = PNBank(bit_len, H, W, key)
        self.feat = nn.Sequential(
            nn.Conv2d(img_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), ConvBlock(64))
        self.cond = nn.Sequential(
            nn.Linear(bit_len, 256), nn.ReLU(),
            nn.Linear(256, 64*16*16), nn.ReLU())
        self.up = nn.Sequential(
            ConvBlock(64),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1))
    def forward(self, x, bits):
        B, C, H, W = x.shape
        f = self.feat(x)
        cond = self.cond(bits).view(B, 64, 16, 16)
        cond = F.interpolate(cond, size=f.shape[-2:], mode='bilinear', align_corners=False)
        h = f + cond
        r_learn = torch.tanh(self.up(h)) * self.res_scale
        P = self.pn()
        s = (bits*2-1).view(-1, self.bit_len, 1, 1)
        r_pn = (s * P).sum(dim=1, keepdim=True).repeat(1,3,1,1) * self.pn_alpha
        r = r_learn + r_pn
        x_w = torch.clamp(x + r, 0, 1)
        return x_w, r

class Decoder(nn.Module):
    def __init__(self, img_ch=3, bit_len=32, H=256, W=256, key="aegis"):
        super().__init__()
        self.bit_len=bit_len; self.H=H; self.W=W
        self.hp = HighPass()
        self.pn = PNBank(bit_len, H, W, key)
        self.mlp = nn.Sequential(nn.Linear(bit_len, bit_len*2), nn.ReLU(), nn.Linear(bit_len*2, bit_len))
    def forward(self, x):
        hp = self.hp(x)
        P = self.pn()
        B, _, H, W = hp.shape
        g = hp.mean(dim=1).reshape(B, -1)
        Pflat = P.reshape(self.bit_len, -1).t()
        base = torch.matmul(g, Pflat) / (H*W)
        return self.mlp(base)

class AegisMark(nn.Module):
    def __init__(self, bit_len=32, image_size=256, pn_alpha=0.02, res_scale=0.02, key="aegis"):
        super().__init__()
        self.enc = Encoder(bit_len=bit_len, H=image_size, W=image_size, pn_alpha=pn_alpha, res_scale=res_scale, key=key)
        self.dec = Decoder(bit_len=bit_len, H=image_size, W=image_size, key=key)
    def forward(self, x, bits):
        x_w, r = self.enc(x, bits)
        logits = self.dec(x_w)
        return x_w, r, logits
