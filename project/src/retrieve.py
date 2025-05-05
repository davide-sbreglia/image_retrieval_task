import torch, faiss, json, argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data  import ClassifyDataset, make_transforms
from src.model import build_model, extract_embedding_model

def build_index(model, ds, device="cuda"):
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    embedder = extract_embedding_model(model).to(device).eval()
    embs, keys = [], []
    with torch.no_grad():
        for X, _, fns in dl:
            X = X.to(device)
            e = embedder(X).cpu().numpy()
            embs.append(e); keys += fns
    embs = np.vstack(embs).astype("float32")
    idx  = faiss.IndexFlatIP(embs.shape[1])  # cosine via L2-normed vectors
    idx.add(embs)
    return idx, keys

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # reload classification model head
    cls_model = build_model(args.num_classes)
    cls_model.load_state_dict(torch.load(args.ft_model, map_location="cpu"))
    cls_model.to(device)

    # create index on train images
    tr_ds = ClassifyDataset(args.train_dir, args.train_json,
                            make_transforms(args.img_size, False))
    index, tr_keys = build_index(cls_model, tr_ds, device)

    # now run on test set
    val_ds = ClassifyDataset(args.val_dir, {}, make_transforms(args.img_size, False))
    dl     = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)
    embedder = extract_embedding_model(cls_model).to(device).eval()

    out = []
    with torch.no_grad():
        for X, _, fns in dl:
            X = X.to(device)
            q = embedder(X).cpu().numpy().astype("float32")
            _, I = index.search(q, args.k)
            for fn, nbrs in zip(fns, I):
                out.append({
                    "filename": fn,
                    "samples": [tr_keys[i] for i in nbrs]
                })

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir",  required=True)
    p.add_argument("--train_json", required=True)
    p.add_argument("--val_dir",    required=True)
    p.add_argument("--k",          type=int, default=5)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--img_size",   type=int,   default=224)
    p.add_argument("--ft_model",   default="best_ft.pth")
    p.add_argument("--out_json",   default="submission.json")
    main(p.parse_args())
