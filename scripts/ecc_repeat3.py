import argparse, numpy as np, json

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["encode","decode"], required=True)
p.add_argument("--bits_in", default="")
p.add_argument("--bits_hat_in", default="")
p.add_argument("--encoded_out", default="")
p.add_argument("--decoded_out", default="")
p.add_argument("--gt", default="")
p.add_argument("--rep", type=int, default=3)
a = p.parse_args()

def load_bits(path):
    b = np.load(path)
    b = np.asarray(b).astype(np.float32).ravel()
    return b

if a.mode == "encode":
    b = load_bits(a.bits_in).astype(np.uint8)
    enc = np.repeat(b, a.rep).astype(np.uint8)
    if a.encoded_out: np.save(a.encoded_out, enc)
    print(json.dumps({"mode":"encode","in_len":int(b.size),"rep":a.rep,"out_len":int(enc.size)}))
else:
    hat = load_bits(a.bits_hat_in)
    if hat.dtype.kind == "f": hat = (hat >= 0.5).astype(np.uint8)
    hat = hat.astype(np.uint8)
    assert hat.size % a.rep == 0
    dec = (hat.reshape(-1, a.rep).sum(1) >= (a.rep//2 + 1)).astype(np.uint8)
    out = {"mode":"decode","rep":a.rep,"in_len":int(hat.size),"out_len":int(dec.size)}
    if a.gt:
        gt = load_bits(a.gt).astype(np.uint8)[:dec.size]
        acc = float((gt == dec).mean())
        out["acc"] = acc
    if a.decoded_out: np.save(a.decoded_out, dec)
    print(json.dumps(out))
