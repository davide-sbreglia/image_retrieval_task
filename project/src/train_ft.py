import torch, argparse
from torch import nn, optim
from torch.utils.data import DataLoader
from data  import ClassifyDataset, make_transforms
from model import build_model

def train_one_epoch(model, loader, loss_fn, opt, device):
    model.train(); model.to(device)
    total_loss, total_acc = 0, 0
    for X, y, _ in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss   = loss_fn(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_acc  += (preds==y).float().mean().item()
    return total_loss/len(loader), total_acc/len(loader)

def validate(model, loader, loss_fn, device):
    model.eval(); model.to(device)
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for X, y, _ in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            val_loss += loss_fn(logits, y).item()
            val_acc  += (logits.argmax(dim=1)==y).float().mean().item()
    return val_loss/len(loader), val_acc/len(loader)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Datasets (train & val both live under train_dir; split controlled by JSONs)
    train_ds = ClassifyDataset(args.train_dir, args.train_json, make_transforms(args.img_size, True))
    val_ds   = ClassifyDataset(args.train_dir, args.val_json,   make_transforms(args.img_size, False))
    # DataLoaders
    # NOTE: num_workers=4 is a good default, but you may need to adjust this
    tr_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=4)
    va_dl = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=4)

    model   = build_model(num_classes=args.num_classes,model_name=args.model_name).to(device)
    loss_fn = nn.CrossEntropyLoss()
    # only train the classifier head
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val = 1e9
    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, tr_dl, loss_fn, opt, device)
        va_loss, va_acc = validate    (model, va_dl, loss_fn,     device)
        print(f"[{ep:02d}] tr {tr_loss:.3f}/{tr_acc:.3f}  va {va_loss:.3f}/{va_acc:.3f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), args.out_model)

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir",     required=True)
    p.add_argument("--train_json",    required=True)
    p.add_argument("--val_json",      required=True)
    p.add_argument("--num_classes",   type=int, default=10)
    p.add_argument("--img_size",      type=int, default=224)
    p.add_argument("--bs",            type=int,   default=64)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--out_model",     default="best_ft.pth")
    p.add_argument("--model_name",    default="efficientnet_b0")
    main(p.parse_args())
