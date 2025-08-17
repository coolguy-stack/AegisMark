import argparse, torch, numpy as np, json
from PIL import Image
import torchvision.transforms.functional as TF
from baseline.ss_baseline import SS

p=argparse.ArgumentParser()
p.add_argument('--img',required=True)
p.add_argument('--bits',required=True)
p.add_argument('--bit_len',type=int,default=32)
p.add_argument('--key',default='aegis')
a=p.parse_args()

img = TF.to_tensor(Image.open(a.img).convert('RGB')).unsqueeze(0)
bits = torch.from_numpy(np.load(a.bits)).float()
ss = SS(bit_len=a.bit_len, key=a.key)
logits = ss.detect_logits(img)
acc = ((logits>0)==bits).float().mean().item()
presence = float(logits.abs().mean().item())
print(json.dumps({"bit_acc":acc,"presence":presence},indent=2))
