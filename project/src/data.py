from pathlib import Path
import json, os, torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def make_transforms(image_size=224, train=True):
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ]
    if train:
        base.insert(0, transforms.RandomResizedCrop(image_size))
        base.insert(1, transforms.RandomHorizontalFlip())
    return transforms.Compose(base)

class ClassifyDataset(Dataset):
    def __init__(self, img_dir: str, json_path: str, transform: callable):
        # 1) load the JSON of filename → class_name
        with open(json_path) as f:
            raw_labels = json.load(f)      # e.g. { "img001.jpg": "American_chameleon", … }

        # 2) build a class2idx mapping from the subfolders under img_dir
        class_names = sorted(
            d for d in os.listdir(img_dir)
            if os.path.isdir(os.path.join(img_dir, d))
        )
        class2idx = {cn: i for i, cn in enumerate(class_names)}

        self.img_dir   = img_dir
        self.transform = transform
        self.samples   = []

        # 3) for every image in each class-folder, look up its class_name in the JSON,
        #    then map that name to an integer
        for class_name in class_names:
            class_folder = os.path.join(img_dir, class_name)
            for fn in os.listdir(class_folder):
                if fn not in raw_labels:
                    continue
                label_name = raw_labels[fn]
                if label_name != class_name:
                    # if your JSON ever disagrees with the folder structure you can skip or warn
                    continue
                class_id = class2idx[label_name]
                self.samples.append((class_name, fn, class_id))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        class_name, fn, class_id = self.samples[idx]
        path = os.path.join(self.img_dir, class_name, fn)
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        label = torch.tensor(class_id, dtype=torch.long)
        return tensor, label, fn

class RetrievalDataset(Dataset):
    def __init__(self, img_dir: str, transform: callable):
        self.img_dir   = img_dir
        self.transform = transform
        # recursively collect all image files under img_dir
        p = Path(img_dir)
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        self.filepaths = []
        for pat in patterns:
            self.filepaths += list(p.rglob(pat))
        # sort so ordering is stable
        self.filepaths = sorted(self.filepaths)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        # make filename relative to img_dir for your fns list:
        fn   = str(path.relative_to(self.img_dir))
        img  = Image.open(path).convert("RGB")
        return self.transform(img), fn