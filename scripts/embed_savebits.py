import argparse, torch, numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

p=argparse.ArgumentParser()
p.add_argument('--encoder',required=True)
p.add_argument('--in_img',required=True)
p.add_argument('--out_img',required=True)
p.add_argument('--bits_out',required=True)
a=p.parse_args()

enc=torch.jit.load(a.encoder,'cpu').eval()
img=TF.to_tensor(Image.open(a.in_img).convert('RGB')).unsqueeze(0)
bit_len=enc.state_dict()['cond.0.weight'].shape[1]
bits=(torch.rand(1,int(bit_len))>0.5).float()
xw,_=enc(img,bits)
TF.to_pil_image(xw.squeeze(0)).save(a.out_img)
np.save(a.bits_out, bits.numpy())
