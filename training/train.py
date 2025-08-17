import os, argparse, json, torch
from torch.utils.data import DataLoader
from dataset import ImageFolderDataset
from model import AegisMark
from losses import reconstruction_loss, bit_loss, total_variation
from attacks import attack_batch
from utils import sample_bits

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--bit_len', type=int, default=32)
    p.add_argument('--image_size', type=int, default=256)
    p.add_argument('--lambda_rec', type=float, default=0.3)
    p.add_argument('--lambda_bit', type=float, default=3.0)
    p.add_argument('--lambda_adv', type=float, default=0.0)
    p.add_argument('--lambda_tv', type=float, default=0.002)
    p.add_argument('--train', default=os.getenv('SM_CHANNEL_TRAIN','/opt/ml/input/data/train'))
    p.add_argument('--val', default=os.getenv('SM_CHANNEL_VAL','/opt/ml/input/data/val'))
    p.add_argument('--model_dir', default=os.getenv('SM_MODEL_DIR','/opt/ml/model'))
    return p.parse_args()

def evaluate(model, dl, device, bit_len):
    model.eval()
    tot, correct = 0, 0
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            b = sample_bits(x.size(0), bit_len).to(device)
            xw, _, logits = model(x, b)
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == b).float().mean().item()*x.size(0)
            tot += x.size(0)
    return correct/max(tot,1)

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ds = ImageFolderDataset(args.train, args.image_size)
    val_ds   = ImageFolderDataset(args.val, args.image_size)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = AegisMark(bit_len=args.bit_len, image_size=args.image_size, pn_alpha=0.02).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for x in train_dl:
            x = x.to(device)
            bits = sample_bits(x.size(0), args.bit_len).to(device)
            xw, r, logits = model(x, bits)
            xa = attack_batch(xw.detach()).to(device)
            logits_att = model.dec(xa)
            l_rec = reconstruction_loss(xw, x)
            l_bit = bit_loss(logits, bits)
            l_tv  = total_variation(r)
            loss = args.lambda_rec*l_rec + args.lambda_bit*l_bit + args.lambda_tv*l_tv
            opt.zero_grad(); loss.backward(); opt.step()
        val_acc = evaluate(model, val_dl, device, args.bit_len)
        print(f"epoch={ep} val_bit_acc={val_acc:.4f}")
        if val_acc > best:
            best = val_acc
            torch.jit.script(model.enc.cpu()).save(os.path.join(args.model_dir, 'encoder.pt'))
            torch.jit.script(model.dec.cpu()).save(os.path.join(args.model_dir, 'decoder.pt'))
            model.enc.to(device); model.dec.to(device)
    with open(os.path.join(args.model_dir,'metrics.json'),'w') as f:
        json.dump({'val_bit_acc': best}, f)

if __name__ == '__main__':
    main()
