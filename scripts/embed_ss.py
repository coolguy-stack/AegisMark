import argparse, torch
from PIL import Image
import torchvision.transforms.functional as TF
from baseline.ss_baseline import SS
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--in_img',required=True)
p.add_argument('--out_img',required=True)
p.add_argument('--bits_out',required=True)
p.add_argument('--bit_len',type=int,default=32)
p.add_argument('--alpha',type=float,default=0.02)
p.add_argument('--key',default='aegis')
a=p.parse_args()

img = TF.to_tensor(Image.open(a.in_img).convert('RGB')).unsqueeze(0)
bits = (torch.rand(1,a.bit_len)>0.5).float()
ss = SS(bit_len=a.bit_len, alpha=a.alpha, key=a.key)
y = ss.embed(img, bits)
TF.to_pil_image(y.squeeze(0)).save(a.out_img)
np.save(a.bits_out, bits.numpy())
