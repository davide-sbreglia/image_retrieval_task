# wandb_benchmark.py

import os, time, json
import torch
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import models
from model import L2Norm
from data import RetrievalDataset, make_transforms
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Define models
# ----------------------------
def get_backbone(name):
    if name == 'resnet50':
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        net = torch.nn.Sequential(*(list(net.children())[:-1]))
        dim = 2048
    elif name == 'efficientnet_b0':
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).features
        net = torch.nn.Sequential(net, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        dim = 1280
    elif name == 'vit_b_16':
        net = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        net.heads = torch.nn.Identity()
        dim = 768
    else:
        raise ValueError(f"Unknown model {name}")
    return torch.nn.Sequential(net, L2Norm()), dim

# ----------------------------
# Extract embeddings
# ----------------------------
def extract_embeddings(model, loader, device):
    model.to(device).eval()
    feats, names = [], []
    start = time.time()
    with torch.no_grad():
        for x, fn in loader:
            x = x.to(device)
            emb = model(x).cpu().numpy()
            feats.append(emb)
            names.extend(fn)
    end = time.time()
    return np.vstack(feats).astype("float32"), names, (end - start) / len(names)

# ----------------------------
# Evaluate retrieval accuracy
# ----------------------------
def evaluate_topk(query_embs, gallery_embs, query_names, gallery_names, labels, k=5):
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(query_embs, gallery_embs)
    indices = np.argsort(sim, axis=1)[:, -k:][:, ::-1]
    correct = 0
    total = 0
    for q_idx, row in enumerate(indices):
        query_label = labels.get(query_names[q_idx])
        retrieved_labels = [labels.get(gallery_names[i]) for i in row]
        correct += sum(1 for l in retrieved_labels if l == query_label)
        total += len(retrieved_labels)
    return correct / total if total else 0.0

# ----------------------------
# Average of two embeddings
# ----------------------------
def average_embeddings(e1, e2):
    return (e1 + e2) / 2

# ----------------------------
# Main benchmark script
# ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224
    data_dir = "./project/training"
    val_json = "val_split.json"
    batch_size = 64
    k = 3

    with open(val_json) as f:
        labels = json.load(f)

    tfm = make_transforms(img_size, train=False)
    loader = DataLoader(RetrievalDataset(data_dir, tfm), batch_size=batch_size, shuffle=False)

    wandb.init(project="retrieval-benchmark", name="full-model-comparison")

    # Singoli modelli
    single_models = ["resnet50", "efficientnet_b0", "vit_b_16"]
    embedding_cache = {}

    for name in single_models:
        model, _ = get_backbone(name)
        embs, fns, avg_time = extract_embeddings(model, loader, device)
        acc = evaluate_topk(embs, embs, fns, fns, labels, k=k)
        wandb.log({"model": name, "topk_acc": acc, "avg_time_per_image": avg_time, "type": "single"})
        embedding_cache[name] = (embs, fns)
        print(f"✅ {name}: acc={acc:.3f}  time/img={avg_time:.4f}s")

    # Ensemble resnet + vit
    r_embs, fns = embedding_cache["resnet50"]
    v_embs, _   = embedding_cache["vit_b_16"]
    avg = average_embeddings(r_embs, v_embs)
    acc = evaluate_topk(avg, avg, fns, fns, labels, k=k)
    wandb.log({"model": "resnet+vit", "topk_acc": acc, "avg_time_per_image": None, "type": "ensemble"})

    # Ensemble eff + vit
    e_embs, fns = embedding_cache["efficientnet_b0"]
    v_embs, _   = embedding_cache["vit_b_16"]
    avg = average_embeddings(e_embs, v_embs)
    acc = evaluate_topk(avg, avg, fns, fns, labels, k=k)
    wandb.log({"model": "eff+vit", "topk_acc": acc, "avg_time_per_image": None, "type": "ensemble"})

    wandb.finish()
