#!/usr/bin/env python3
import argparse, os, numpy as np, torch
from PIL import Image
import torchvision.transforms.functional as TF

def bits_from_hex(h, bit_len):
    h = h.strip().lower().replace("0x","")
    b = bin(int(h,16))[2:].zfill(bit_len)
    return np.array([int(x) for x in b[-bit_len:]], dtype=np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--in_img", required=True)
    ap.add_argument("--out_img", required=True)
    ap.add_argument("--bits_out", required=True)
    ap.add_argument("--bit_len", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.009)
    ap.add_argument("--save_jpeg_q", type=int, default=95)
    ap.add_argument("--bits_in", default="", help="path to .npy bits (0/1) to embed")
    ap.add_argument("--token_hex", default="", help="hex token to embed; overrides bits_in")
    args = ap.parse_args()

    enc = torch.jit.load(args.encoder, map_location="cpu").eval()
    im = Image.open(args.in_img).convert("RGB")
    W = H = getattr(enc, "size", 256)
    im = im.resize((W,H), Image.BILINEAR)
    x = TF.to_tensor(im).unsqueeze(0)

    if args.token_hex:
        bits = bits_from_hex(args.token_hex, args.bit_len)
    elif args.bits_in:
        bits = np.load(args.bits_in).astype(np.uint8).ravel()
        if bits.size < args.bit_len:
            pad = np.random.randint(0,2,size=args.bit_len-bits.size,dtype=np.uint8)
            bits = np.concatenate([bits, pad], 0)
        bits = bits[:args.bit_len]
    else:
        bits = np.random.randint(0,2,size=args.bit_len,dtype=np.uint8)

    b = torch.from_numpy(bits[None,:].astype("float32"))
    with torch.no_grad():
        try:
            y,_ = enc(x, b, args.alpha)
        except (TypeError, RuntimeError):
            y,_ = enc(x, b)
    y = y.clamp(0,1)
    Image.fromarray((y[0].permute(1,2,0).numpy()*255).astype("uint8")).save(args.out_img, quality=args.save_jpeg_q)
    os.makedirs(os.path.dirname(args.bits_out) or ".", exist_ok=True)
    np.save(args.bits_out, bits)

if __name__ == "__main__":
    main()
