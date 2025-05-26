import os, json, random, argparse
from PIL import Image
import torch, faiss
import numpy as np
from torch.utils.data import DataLoader
from train_ft import main as train_main
from data import make_transforms, RetrievalDataset
from model import build_model, extract_embedding_model
from retrieve import build_index  # assumes build_index in retrieve.py

# import benchmark functions
from wandb_benchmark import get_backbone, extract_embeddings, evaluate_topk


def choose_best_model(train_dir, val_json, img_size, batch_size, k=3, device=None):
    """
    Benchmarks available backbones on retrieval accuracy and average inference time,
    returns the model name with highest accuracy (ties broken by lower avg_time).
    """
    # Load labels for validation
    with open(val_json) as f:
        labels = json.load(f)
    # --- Split into query/gallery filenames ---
    all_fns = list(labels.keys())
    random.shuffle(all_fns)
    split = int(0.2 * len(all_fns))  # 20% for query
    query_fns = set(all_fns[:split])
    gallery_fns = set(all_fns[split:])

    # --- Filter labels ---
    query_labels   = {k: labels[k] for k in query_fns}
    gallery_labels = {k: labels[k] for k in gallery_fns}

    # --- Custom subset datasets ---
    tfm = make_transforms(img_size, train=False)
    val_dir = train_dir  # all validation images are in the same folder

    def subset_dataset(filenames):
        from pathlib import Path
        from torch.utils.data import Dataset
        class SubsetRetrievalDataset(Dataset):
            def __init__(self, root, transform, keep_set):
                self.transform = transform
                self.samples = []
                for cls in sorted(os.listdir(root)):
                    folder = os.path.join(root, cls)
                    if not os.path.isdir(folder):
                        continue
                    for fn in os.listdir(folder):
                        if fn in keep_set:
                            self.samples.append((os.path.join(folder, fn), fn))
            def __len__(self): return len(self.samples)
            def __getitem__(self, idx):
                path, fn = self.samples[idx]
                img = Image.open(path).convert("RGB")
                return self.transform(img), fn
        return SubsetRetrievalDataset(val_dir, tfm, filenames)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    candidates = ["resnet50", "efficientnet_b0", "vit_b_16"]

    for name in candidates:
        model, _ = get_backbone(name)
        query_ds = subset_dataset(query_fns)
        gallery_ds = subset_dataset(gallery_fns)

        query_dl = DataLoader(query_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        gallery_dl = DataLoader(gallery_ds, batch_size=batch_size, shuffle=False, num_workers=4)

        query_embs, query_names, q_time = extract_embeddings(model, query_dl, device)
        gallery_embs, gallery_names, _ = extract_embeddings(model, gallery_dl, device)

        acc = evaluate_topk(query_embs, gallery_embs, query_names, gallery_names, labels, k)
        print(f"Benchmark {name}: acc={acc:.4f}, avg_time_per_image={q_time:.4f}s")
        results.append((name, acc, q_time))
    # select best by acc, then time
    results.sort(key=lambda x: (-x[1], x[2]))
    best_name, best_acc, best_time = results[0]
    print(f"→ Selected model: {best_name} (acc={best_acc:.4f}, time={best_time:.4f}s)")
    return best_name


def do_split(args):
    classes = os.listdir(args.train_dir)
    train_labels, val_labels = {}, {}
    for cls in classes:
        cls_path = os.path.join(args.train_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        files = sorted(os.listdir(cls_path))
        random.shuffle(files)
        split = int(len(files) * args.split_ratio)
        for fn in files[:split]:
            train_labels[fn] = cls
        for fn in files[split:]:
            val_labels[fn] = cls
    json.dump(train_labels, open(args.train_json, 'w'), indent=2)
    json.dump(val_labels,   open(args.val_json,   'w'), indent=2)
    print(f"→ train: {len(train_labels)}  val: {len(val_labels)}")


def do_train(args):
    train_main(args)


def do_index(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_model = build_model(args.num_classes, model_name=args.model_name).to(device)
    full_model.load_state_dict(torch.load(args.ft_model, map_location='cpu'))
    embed = extract_embedding_model(full_model, model_name=args.model_name).to(device).eval()
    gallery_ds = RetrievalDataset(args.gallery_dir, make_transforms(args.img_size, False))
    idx, keys = build_index(embed, gallery_ds, device)
    faiss.write_index(idx, args.idx_out)
    json.dump(keys, open(args.keys_out, 'w'), indent=2)
    print(f"✔ gallery index: {idx.ntotal} vectors saved to {args.idx_out}")


def do_retrieve(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls = build_model(args.num_classes, model_name=args.model_name).to(device)
    cls.load_state_dict(torch.load(args.ft_model, map_location='cpu'))
    cls.eval()
    idx = faiss.read_index(args.idx)
    gallery = json.load(open(args.keys, 'r'))
    query_ds = RetrievalDataset(args.query_dir, make_transforms(args.img_size, False))
    dl = DataLoader(query_ds, batch_size=args.k, shuffle=False, num_workers=4)
    embed = extract_embedding_model(cls, model_name=args.model_name).to(device).eval()
    out = []
    #out={} -> to use in case of dictionary output instead of json
    with torch.no_grad():
        for X, fns in dl:
            X = X.to(device)
            e = embed(X)
            e = torch.nn.functional.normalize(e, p=2, dim=1).cpu().numpy().astype('float32')
            D, I = idx.search(e, args.k)
            for fn, nbrs in zip(fns, I):
                out.append({"filename": os.path.join("test_folder", "query_images", fn),
                            "gallery_images": [os.path.join("test_folder", "gallery_images", gallery[i]) for i in nbrs]})
            #for fn, nbrs in zip(fns, I):
                #out[fn] = [gallery[i] for i in nbrs]
    json.dump(out, open(args.out_json, 'w'), indent=2)
    print(f"✔ wrote {args.out_json} for {len(out)} queries")
    #submit(out, "MLOps")


def main():
    p = argparse.ArgumentParser(description="Unified pipeline CLI with automated model selection")
    sub = p.add_subparsers(dest='cmd')
    # split
    sp = sub.add_parser('split')
    sp.add_argument('--train_dir', required=True)
    sp.add_argument('--train_json', required=True)
    sp.add_argument('--val_json', required=True)
    sp.add_argument('--split_ratio', type=float, default=0.8)
    # train
    tp = sub.add_parser('train')
    for a in ['train_dir','train_json','val_json','num_classes','img_size','bs','lr','epochs','out_model']:
        tp.add_argument(f'--{a}', required=a.endswith('dir') or a.endswith('json') or a=='out_model')
    tp.add_argument('--model_name', default='efficientnet_b0')
    # index
    ip = sub.add_parser('index')
    ip.add_argument('--ft_model', required=True)
    ip.add_argument('--gallery_dir', required=True)
    ip.add_argument('--img_size', type=int, default=224)
    ip.add_argument('--idx_out', default='gallery_idx.faiss')
    ip.add_argument('--keys_out', default='gallery_keys.json')
    ip.add_argument('--num_classes', type=int, default=10)
    ip.add_argument('--model_name', default='efficientnet_b0')
    # retrieve
    rp = sub.add_parser('retrieve')
    rp.add_argument('--ft_model', required=True)
    rp.add_argument('--query_dir', required=True)
    rp.add_argument('--idx', required=True)
    rp.add_argument('--keys', required=True)
    rp.add_argument('--k', type=int, default=5)
    rp.add_argument('--img_size', type=int, default=224)
    rp.add_argument('--out_json', default='submission.json')
    rp.add_argument('--num_classes', type=int, default=10)
    rp.add_argument('--model_name', default='efficientnet_b0')
    # full pipeline
    fp = sub.add_parser('full')
    fp.add_argument('--train_dir', required=True)
    fp.add_argument('--test_dir', required=True)
    fp.add_argument('--train_json', default='train_split.json')
    fp.add_argument('--val_json', default='val_split.json')
    fp.add_argument('--split_ratio', type=float, default=0.8)
    fp.add_argument('--num_classes', type=int, default=3)
    fp.add_argument('--img_size', type=int, default=224)
    fp.add_argument('--bs', type=int, default=32)
    fp.add_argument('--lr', type=float, default=1e-4)
    fp.add_argument('--epochs', type=int, default=10)
    fp.add_argument('--k', type=int, default=3)
    fp.add_argument('--out_model', default='best_ft.pth')
    fp.add_argument('--idx_out', default='gallery_idx.faiss')
    fp.add_argument('--keys_out', default='gallery_keys.json')
    fp.add_argument('--out_json', default='submission.json')
    # note: model_name for full is selected automatically

    args = p.parse_args()
    if args.cmd == 'split':
        do_split(args)
    elif args.cmd == 'train':
        do_train(args)
    elif args.cmd == 'index':
        do_index(args)
    elif args.cmd == 'retrieve':
        do_retrieve(args)
    elif args.cmd == 'full':
        # 1) split
        do_split(args)
        # 2) automatic model selection via retrieval benchmark
        best_model = choose_best_model(
            train_dir=args.train_dir,
            val_json=args.val_json,
            img_size=args.img_size,
            batch_size=args.bs,
            k=args.k
        )
        # 3) training
        train_args = argparse.Namespace(
            train_dir=args.train_dir,
            train_json=args.train_json,
            val_json=args.val_json,
            num_classes=args.num_classes,
            img_size=args.img_size,
            bs=args.bs,
            lr=args.lr,
            epochs=args.epochs,
            out_model=args.out_model,
            model_name=best_model
        )
        do_train(train_args)
        # 4) build index on test gallery
        idx_args = argparse.Namespace(
            ft_model=args.out_model,
            gallery_dir=os.path.join(args.test_dir,'gallery'),
            img_size=args.img_size,
            idx_out=args.idx_out,
            keys_out=args.keys_out,
            num_classes=args.num_classes,
            model_name=best_model
        )
        do_index(idx_args)
        # 5) retrieve on test queries
        ret_args = argparse.Namespace(
            ft_model=args.out_model,
            query_dir=os.path.join(args.test_dir,'query'),
            idx=args.idx_out,
            keys=args.keys_out,
            k=args.k,
            img_size=args.img_size,
            out_json=args.out_json,
            num_classes=args.num_classes,
            model_name=best_model
        )
        do_retrieve(ret_args)
    else:
        p.print_help()

if __name__ == '__main__':
    main()
# Note: This script assumes the existence of a wandb_benchmark.py file with the required functions.
# The script is designed to be run from the command line with various subcommands for different tasks.
# The full pipeline automates the process of splitting data, selecting the best model, training, indexing, and retrieving.
