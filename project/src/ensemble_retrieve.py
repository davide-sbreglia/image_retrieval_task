# ensemble_retrieve.py

import torch, faiss, json, os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from data import RetrievalDataset, make_transforms
from model import L2Norm
from torchvision import transforms
from tqdm import tqdm


def _flatten(embs: np.ndarray) -> np.ndarray:
    """
    Collapse any trailing dimensions so that the result is always
    shape (n_samples, feature_dim).
    """
    embs = np.asarray(embs, dtype='float32')
    if embs.ndim > 2:
        # e.g. (batch, C, 1, 1) -> (batch, C)
        return embs.reshape(embs.shape[0], -1)
    return embs
# ----------------------------
# Build backbone models
# ----------------------------
def get_backbone(name):
    if name == 'resnet50':
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        net = torch.nn.Sequential(*(list(net.children())[:-1]))  # remove fc
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
    feats = []
    fns = []
    with torch.no_grad():
        for x, fn in tqdm(loader):
            x = x.to(device)
            emb = model(x).cpu().numpy()
            emb = _flatten(emb)
            feats.append(emb)
            fns.extend(fn)
    return np.vstack(feats).astype('float32'), fns


# ----------------------------
# Retrieval
# ----------------------------
def do_retrieve(query_embs, gallery_embs, gallery_keys, k):
    gallery_embs = _flatten(gallery_embs)
    query_embs = _flatten(query_embs)
    idx = faiss.IndexFlatIP(gallery_embs.shape[1])
    faiss.normalize_L2(gallery_embs)
    faiss.normalize_L2(query_embs)
    idx.add(gallery_embs)
    D, I = idx.search(query_embs, k)
    return [[gallery_keys[i] for i in row] for row in I]

# ----------------------------
# Save submission
# ----------------------------
def save_submission(query_fns, retrieved_lists, out_json):
    results = []
    for qf, samples in zip(query_fns, retrieved_lists):
        results.append({"filename": qf, "samples": samples})
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224
    k = 5
    query_dir = "./test/query"
    gallery_dir = "./test/gallery"
    out_dir = "submissions"
    os.makedirs(out_dir, exist_ok=True)

    # Dataset loader
    tfm = make_transforms(img_size, train=False)
    query_dl = DataLoader(RetrievalDataset(query_dir, tfm), batch_size=64, shuffle=False)
    gallery_dl = DataLoader(RetrievalDataset(gallery_dir, tfm), batch_size=64, shuffle=False)

    # Singoli modelli
    for model_name in ["resnet50", "efficientnet_b0", "vit_b_16"]:
        model, _ = get_backbone(model_name)
        g_embs, g_keys = extract_embeddings(model, gallery_dl, device)
        q_embs, q_keys = extract_embeddings(model, query_dl, device)
        retrieved = do_retrieve(q_embs, g_embs, g_keys, k)
        save_submission(q_keys, retrieved, f"{out_dir}/submission_{model_name}.json")

    def l2norm_np(x): return x / np.linalg.norm(x, axis=1, keepdims=True)
    # Ensemble resnet + vit
    model1, _ = get_backbone("resnet50")
    model2, _ = get_backbone("vit_b_16")
    g1, gk = extract_embeddings(model1, gallery_dl, device)
    g2, _  = extract_embeddings(model2, gallery_dl, device)
    q1, qk = extract_embeddings(model1, query_dl, device)
    q2, _  = extract_embeddings(model2, query_dl, device)
    g1, g2 = l2norm_np(g1), l2norm_np(g2)
    q1, q2 = l2norm_np(q1), l2norm_np(q2)
    sim1 = np.dot(q1, g1.T)
    sim2 = np.dot(q2, g2.T)
    sim_ensemble = (sim1 +sim2)/2
    topk_indices = np.argsort(-sim_ensemble, axis=1)[:, :k]
    retrieved = [[gk[i] for i in row] for row in topk_indices]
    save_submission(qk, retrieved, f"{out_dir}/submission_resnet_vit_similarity.json")
    

    # Ensemble efficientnet + vit
    model1, _ = get_backbone("efficientnet_b0")
    model2, _ = get_backbone("vit_b_16")
    g1, gk = extract_embeddings(model1, gallery_dl, device)
    g2, _  = extract_embeddings(model2, gallery_dl, device)
    q1, qk = extract_embeddings(model1, query_dl, device)
    q2, _  = extract_embeddings(model2, query_dl, device)
    g1, g2 = l2norm_np(g1), l2norm_np(g2)
    q1, q2 = l2norm_np(q1),l2norm_np(q2)
    sim1 = np.dot(q1, g1.T)
    sim2 = np.dot(q2, g2.T)
    sim_ensemble = (sim1+sim2)/2
    topk_indices = np.argsort(-sim_ensemble, axis=1)[:, :k]
    retrieved = [[gk[i] for i in row] for row in topk_indices]
    save_submission(qk, retrieved, f"{out_dir}/submission_eff_vit_similarity.json")
