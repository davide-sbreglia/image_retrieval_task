import os, json, random, argparse
import torch, faiss
import numpy as np
from torch.utils.data import DataLoader
from train_ft import main as train_main
from data import make_transforms, RetrievalDataset
from model import build_model, extract_embedding_model
from retrieve import build_index  # assumes build_index in retrieve.py

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
    cls = build_model(args.num_classes, model_name=args.model_name).to(device)
    cls.load_state_dict(torch.load(args.ft_model, map_location='cpu'))
    cls.eval()
    gallery_ds = RetrievalDataset(args.gallery_dir, make_transforms(args.img_size, False))
    idx, keys = build_index(cls, gallery_ds, device, args.model_name)
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
    with torch.no_grad():
        for X, fns in dl:
            X = X.to(device)
            e = embed(X)
            e = torch.nn.functional.normalize(e, p=2, dim=1).cpu().numpy().astype('float32')
            D, I = idx.search(e, args.k)
            for fn, nbrs in zip(fns, I):
                out.append({'filename': fn, 'samples': [gallery[i] for i in nbrs]})
    json.dump(out, open(args.out_json, 'w'), indent=2)
    print(f"✔ wrote {args.out_json} for {len(out)} queries")

def main():
    p = argparse.ArgumentParser(description="Unified pipeline CLI")
    sub = p.add_subparsers(dest='cmd')
    sp = sub.add_parser('split')
    sp.add_argument('--train_dir', required=True)
    sp.add_argument('--train_json', required=True)
    sp.add_argument('--val_json', required=True)
    sp.add_argument('--split_ratio', type=float, default=0.8)

    tp = sub.add_parser('train')
    for a in ['train_dir','train_json','val_dir','val_json','num_classes','img_size','bs','lr','epochs','out_model']:
        tp.add_argument(f'--{a}', required=a.endswith('dir') or a.endswith('json') or a=='out_model')
    tp.add_argument('--model_name', default='efficientnet_b0')

    ip = sub.add_parser('index')
    ip.add_argument('--ft_model', required=True)
    ip.add_argument('--gallery_dir', required=True)
    ip.add_argument('--img_size', type=int, default=224)
    ip.add_argument('--idx_out', default='gallery_idx.faiss')
    ip.add_argument('--keys_out', default='gallery_keys.json')
    ip.add_argument('--num_classes', type=int, default=10)
    ip.add_argument('--model_name', default='efficientnet_b0')

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

    fp = sub.add_parser('full')
    fp.add_argument('--train_dir', required=True)
    fp.add_argument('--val_dir', required=True)
    fp.add_argument('--test_dir', required=True)
    fp.add_argument('--train_json', default='train_split.json')
    fp.add_argument('--val_json', default='val_split.json')
    fp.add_argument('--split_ratio', type=float, default=0.8)
    fp.add_argument('--num_classes', type=int, default=3)
    fp.add_argument('--img_size', type=int, default=224)
    fp.add_argument('--bs', type=int, default=32)
    fp.add_argument('--lr', type=float, default=1e-4)
    fp.add_argument('--epochs', type=int, default=10)
    fp.add_argument('--k', type=int, default=5)
    fp.add_argument('--out_model', default='best_ft.pth')
    fp.add_argument('--idx_out', default='gallery_idx.faiss')
    fp.add_argument('--keys_out', default='gallery_keys.json')
    fp.add_argument('--out_json', default='submission.json')
    fp.add_argument('--model_name', default='efficientnet_b0')

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
        do_split(args)
        train_args = argparse.Namespace(**{k: getattr(args,k) for k in ['train_dir','train_json','val_dir','val_json','num_classes','img_size','bs','lr','epochs','out_model','model_name']})
        do_train(train_args)
        idx_args = argparse.Namespace(ft_model=args.out_model, gallery_dir=os.path.join(args.test_dir,'gallery'), img_size=args.img_size, idx_out=args.idx_out, keys_out=args.keys_out, num_classes=args.num_classes, model_name=args.model_name)
        do_index(idx_args)
        ret_args = argparse.Namespace(ft_model=args.out_model, query_dir=os.path.join(args.test_dir,'query'), idx=args.idx_out, keys=args.keys_out, k=args.k, img_size=args.img_size, out_json=args.out_json, num_classes=args.num_classes, model_name=args.model_name)
        do_retrieve(ret_args)
    else:
        p.print_help()

if __name__ == '__main__':
    main()

