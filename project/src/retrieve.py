import torch, faiss, json, argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from data  import ClassifyDataset, make_transforms
from model import build_model, extract_embedding_model

def build_index(embedder, ds, device="cuda"):
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    embedder = embedder.to(device).eval()
    embs, keys = [], []
    with torch.no_grad():
        for batch in dl:
            if len(batch) == 3:
                X, _, fns = batch
            elif len(batch) == 2:
                X,    fns = batch
            else:
                raise ValueError(f"Expected batch of 2 or 3 elements, got {len(batch)}")

            X = X.to(device)
            e = embedder(X).cpu().numpy()
            embs.append(e); keys += fns
            
    embs = np.vstack(embs).astype("float32")
    idx  = faiss.IndexFlatIP(embs.shape[1])  # cosine via L2-normed vectors
    idx.add(embs)
    return idx, keys

def main(args):
    out = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # reload classification model head
    cls_model = build_model(args.num_classes)
    cls_model.load_state_dict(torch.load(args.ft_model, map_location="cpu"))
    cls_model.to(device)

    # build gallery index
    gallery_ds = RetrievalDataset(os.path.join(args.test_dir, "gallery"),
                                make_transforms(args.img_size, False))
    gallery_index, gallery_keys = build_index(cls_model, gallery_ds, device)

    # embed queries and retrieve
    query_ds    = RetrievalDataset(os.path.join(args.test_dir, "query"),
                                make_transforms(args.img_size, False))
    query_dl    = DataLoader(query_ds, batch_size=64, shuffle=False, num_workers=4)
    
    with torch.no_grad():
        for X, fns in query_dl:
            X = X.to(device)
            q = embedder(X).cpu().numpy().astype("float32")
            _, I = gallery_index.search(q, args.k)
            for fn, nbrs in zip(fns, I):
                out.append({
                    "query": fn,
                    "neighbors": [gallery_keys[i] for i in nbrs]
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
