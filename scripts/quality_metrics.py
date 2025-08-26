import argparse, numpy as np, math
from PIL import Image
from scipy.ndimage import gaussian_filter
import json

p = argparse.ArgumentParser()
p.add_argument("--ref", required=True)
p.add_argument("--test", required=True)
a = p.parse_args()

def to_np(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32)

def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12: return 99.0
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)

def ssim_gray(a, b):
    a = 0.2989*a[...,0] + 0.5870*a[...,1] + 0.1140*a[...,2]
    b = 0.2989*b[...,0] + 0.5870*b[...,1] + 0.1140*b[...,2]
    K1, K2, L = 0.01, 0.03, 255.0
    C1, C2 = (K1*L)**2, (K2*L)**2
    mu_a = gaussian_filter(a, sigma=1.5)
    mu_b = gaussian_filter(b, sigma=1.5)
    a2 = gaussian_filter(a*a, sigma=1.5)
    b2 = gaussian_filter(b*b, sigma=1.5)
    ab = gaussian_filter(a*b, sigma=1.5)
    sigma_a2 = a2 - mu_a*mu_a
    sigma_b2 = b2 - mu_b*mu_b
    sigma_ab = ab - mu_a*mu_b
    num = (2*mu_a*mu_b + C1) * (2*sigma_ab + C2)
    den = (mu_a*mu_a + mu_b*mu_b + C1) * (sigma_a2 + sigma_b2 + C2)
    s = num / (den + 1e-12)
    return float(np.mean(s))

ref = to_np(a.ref)
tst = to_np(a.test)
out = {"psnr": float(psnr(ref, tst)), "ssim_gray": float(ssim_gray(ref, tst))}
print(json.dumps(out))
