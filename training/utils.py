import torch, numpy as np, hashlib

def sample_bits(batch, bit_len=96):
    return torch.randint(0, 2, (batch, bit_len)).float()

def prng_from_key(key: str, salt: str = "aegis"):
    h = hashlib.sha256((salt+key).encode()).digest()
    seed = int.from_bytes(h[:8], 'big') % (2**32-1)
    g = np.random.default_rng(seed)
    return g