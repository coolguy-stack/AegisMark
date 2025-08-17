import numpy as np

def rep_encode(bits, rep=3):
    return np.repeat(bits, rep)

def rep_decode(bits_hat, rep=3):
    assert len(bits_hat) % rep == 0
    groups = bits_hat.reshape(-1, rep)
    return (groups.sum(axis=1) >= (rep//2 + 1)).astype(np.uint8)

try:
    import reedsolo
    def rs_encode(bits, nsym=16):
        by = np.packbits(bits)
        enc = reedsolo.RSCodec(nsym).encode(by)
        return np.unpackbits(np.frombuffer(bytes(enc), dtype=np.uint8))
    def rs_decode(bits, nsym=16):
        by = np.packbits(bits)
        dec = reedsolo.RSCodec(nsym).decode(bytes(by))[0]
        return np.unpackbits(np.frombuffer(bytes(dec), dtype=np.uint8))
except Exception:
    rs_encode = None
    rs_decode = None