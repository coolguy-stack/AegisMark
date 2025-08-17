import torch.nn.functional as F

def reconstruction_loss(xw, x):
    return F.mse_loss(xw, x)

def bit_loss(logits, bits):
    return F.binary_cross_entropy_with_logits(logits, bits)

def total_variation(x):
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w