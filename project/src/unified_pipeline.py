#!/usr/bin/env python3
"""
Unified Retrieval Pipeline
Commands:
  split       - create train/val JSON splits
  train       - fine-tune a classification head
  index       - build FAISS index on gallery
  retrieve    - run retrieval with single model
  ensemble    - run ensemble retrieval
  benchmark   - benchmark models/ensembles on val set with W&B
"""
import os
import json
import random
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
import wandb
import faiss
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Data utils
# ----------------------------
def make_transforms(image_size=224, train=True):
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ]
    if train:
        base.insert(0, transforms.RandomResizedCrop(image_size))
        base.insert(1, transforms.RandomHorizontalFlip())
    return transforms.Compose(base)

class ClassifyDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, json_path, transform):
        with open(json_path) as f:
            raw = json.load(f)
        classes = sorted(d for d in os.listdir(img_dir)
                         if os.path.isdir(os.path.join(img_dir,d)))
        class2idx = {c:i for i,c in enumerate(classes)}
        self.samples = []
        for cls in classes:
            folder = os.path.join(img_dir, cls)
            for fn in os.listdir(folder):
                if fn in raw and raw[fn]==cls:
                    self.samples.append((os.path.join(folder,fn), class2idx[cls]))
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path,label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, os.path.basename(path)

class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform):
        self.filenames = sorted(os.listdir(img_dir))
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self): return len(self.filenames)
    def __getitem__(self, idx):
        fn = self.filenames[idx]
        img = Image.open(os.path.join(self.img_dir, fn)).convert("RGB")
        return self.transform(img), fn

# ----------------------------
# Model utils
# ----------------------------
class L2Norm(nn.Module):
    def forward(self, x): return F.normalize(x, p=2, dim=1)

def build_model(num_classes, model_name='efficientnet_b0'):
    if model_name=='efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for p in m.features.parameters(): p.requires_grad=False
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f,num_classes))
    elif model_name=='resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for p in m.parameters(): p.requires_grad=False
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f,num_classes)
    elif model_name=='vit_b_16':
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        for p in m.parameters(): p.requires_grad=False
        in_f = m.heads.head.in_features
        m.heads.head = nn.Linear(in_f,num_classes)
    else:
        raise ValueError(model_name)
    return m

def extract_embedding_model(model, model_name='efficientnet_b0'):
    if model_name=='efficientnet_b0':
        return nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(), L2Norm())
    elif model_name=='resnet50':
        return nn.Sequential(*list(model.children())[:-1], nn.Flatten(), L2Norm())
    elif model_name=='vit_b_16':
        model.heads = nn.Identity()
        return nn.Sequential(model, L2Norm())
    else:
        raise ValueError(model_name)

# ----------------------------
# Pipeline commands
# ----------------------------

def cmd_split(args):
    classes=os.listdir(args.train_dir)
    train,val={},{}
    for cls in classes:
        p=os.path.join(args.train_dir,cls)
        if not os.path.isdir(p): continue
        files=os.listdir(p)
        random.shuffle(files)
        split=int(len(files)*args.split_ratio)
        for fn in files[:split]: train[fn]=cls
        for fn in files[split:]: val[fn]=cls
    with open(args.train_json,'w') as f: json.dump(train,f,indent=2)
    with open(args.val_json,'w') as f: json.dump(val,f,indent=2)
    print(f"Split done: train {len(train)} | val {len(val)}")


def train_one_epoch(model, loader, loss_fn, opt, device):
    model.train().to(device)
    l,a=0,0
    for X,y,_ in loader:
        X,y=X.to(device),y.to(device)
        logits=model(X); loss=loss_fn(logits,y)
        opt.zero_grad(); loss.backward(); opt.step()
        l+=loss.item(); a+= (logits.argmax(1)==y).float().mean().item()
    return l/len(loader),a/len(loader)

def validate(model, loader, loss_fn, device):
    model.eval().to(device)
    l,a=0,0
    with torch.no_grad():
        for X,y,_ in loader:
            X,y=X.to(device),y.to(device)
            logits=model(X)
            l+=loss_fn(logits,y).item()
            a+= (logits.argmax(1)==y).float().mean().item()
    return l/len(loader),a/len(loader)

def cmd_train(args):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr_ds=ClassifyDataset(args.train_dir,args.train_json,make_transforms(args.img_size,True))
    va_ds=ClassifyDataset(args.val_dir,args.val_json,make_transforms(args.img_size,False))
    tr_dl,va_dl=DataLoader(tr_ds,args.bs,True,4),DataLoader(va_ds,args.bs,False,4)
    model=build_model(args.num_classes,args.model_name).to(device)
    loss_fn=nn.CrossEntropyLoss()
    opt=optim.Adam(filter(lambda p:p.requires_grad, model.parameters()),lr=args.lr)
    best=float('inf')
    for ep in range(1,args.epochs+1):
        l1,a1=train_one_epoch(model,tr_dl,loss_fn,opt,device)
        l2,a2=validate(model,va_dl,loss_fn,device)
        print(f"[{ep}] tr {l1:.3f}/{a1:.3f}  va {l2:.3f}/{a2:.3f}")
        if l2<best:
            best=l2; torch.save(model.state_dict(),args.out_model)
    print(f"Saved {args.out_model}")


def build_index(model, ds, device):
    dl=DataLoader(ds,64,False,4)
    emb_mod=extract_embedding_model(model, args.model_name).to(device).eval()
    embs,keys=[],[]
    with torch.no_grad():
        for X,fns in dl:
            e=emb_mod(X.to(device)).cpu().numpy()
            embs.append(e); keys+=fns
    embs=np.vstack(embs).astype('float32')
    idx=faiss.IndexFlatIP(embs.shape[1]); faiss.normalize_L2(embs); idx.add(embs)
    return idx,keys

def cmd_index(args):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(args.num_classes,args.model_name).to(device)
    model.load_state_dict(torch.load(args.ft_model,map_location='cpu'))
    model.eval()
    ds=RetrievalDataset(args.gallery_dir,make_transforms(args.img_size,False))
    idx,keys=build_index(model,ds,device)
    faiss.write_index(idx,args.idx_out)
    json.dump(keys,open(args.keys_out,'w'),indent=2)
    print("Index saved")

def extract_embeddings(model, loader, device):
    model.to(device).eval()
    feats,names=[],[]; t0=time.time()
    with torch.no_grad():
        for X,fns in loader:
            e=model(X.to(device)).cpu().numpy()
            feats.append(e); names+=fns
    return np.vstack(feats).astype('float32'), names, (time.time()-t0)/len(names)


def cmd_retrieve(args):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(args.num_classes,args.model_name).to(device)
    model.load_state_dict(torch.load(args.ft_model,map_location='cpu'))
    emb_mod=extract_embedding_model(model,args.model_name).to(device).eval()
    idx=faiss.read_index(args.idx)
    gallery=json.load(open(args.keys,'r'))
    dl=DataLoader(RetrievalDataset(args.query_dir,make_transforms(args.img_size,False)),args.k,False,4)
    out=[]
    with torch.no_grad():
        for X,fns in dl:
            q=emb_mod(X.to(device)); faiss.normalize_L2(q.cpu().numpy())
            D,I=idx.search(q.cpu().numpy().astype('float32'),args.k)
            for fn,row in zip(fns,I): out.append({'filename':fn,'samples':[gallery[i] for i in row]})
    json.dump(out,open(args.out_json,'w'),indent=2)
    print("Retrieval done")

def cmd_ensemble(args):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tfm=make_transforms(args.img_size,False)
    gallery_dl=DataLoader(RetrievalDataset(args.gallery_dir,tfm),64,False,4)
    query_dl=DataLoader(RetrievalDataset(args.query_dir,tfm),64,False,4)
    cache={}
    for name in [args.model1,args.model2]:
        m,_=get_backbone(name); embs,keys,_=extract_embeddings(m, gallery_dl,device)
        qembs, qfns,_=extract_embeddings(m, query_dl,device)
        cache[name]=(embs,keys,qembs,qfns)
    e1,e2=cache[args.model1],cache[args.model2]
    avg_g,avg_q=average_embeddings(e1[0],e2[0]),average_embeddings(e1[2],e2[2])
    idx=faiss.IndexFlatIP(avg_g.shape[1]); faiss.normalize_L2(avg_g); idx.add(avg_g)
    out=[]
    for qfn,q in zip(e1[3],avg_q):
        D,I=idx.search(q.reshape(1,-1),args.k)
        out.append({'filename':qfn,'samples':[e1[1][i] for i in I[0]]})
    json.dump(out,open(args.out_json,'w'),indent=2)
    print(f"Ensemble {args.model1}+{args.model2} done")


def cmd_benchmark(args):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tfm=make_transforms(args.img_size,False)
    loader=DataLoader(RetrievalDataset(args.val_dir,tfm),args.bs,False,4)
    with open(args.val_json) as f: labels=json.load(f)
    wandb.init(project=args.wb_project, entity=args.wb_entity, name=args.wb_run)
    cache={};
    for m in args.models:
        mod,_=get_backbone(m); embs,fns,tm=extract_embeddings(extract_embedding_model(mod,m),loader,device)
        acc=evaluate_topk(embs,embs,fns,fns,labels,args.k)
        wandb.log({"model":m,"topk_acc":acc,"time/img":tm})
        cache[m]=(embs,fns)
    for e in args.ensembles:
        m1,m2=e.split('+')
        g1,k1=cache[m1]; g2,_=cache[m2]
        avg_g=average_embeddings(g1,g2)
        acc=evaluate_topk(avg_g,avg_g,k1,k1,labels,args.k)
        wandb.log({"model":e,"topk_acc":acc})
    wandb.finish()

# ----------------------------
# CLI
# ----------------------------
def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='cmd')
    sp=sub.add_parser('split'); sp.add_argument(...) # omitted for brevity
    # Add args for each cmd similarly...
    # Finally parse and dispatch:
    args=p.parse_args()
    globals()[f"cmd_{args.cmd}"](args)

if __name__=='__main__':
    main()

