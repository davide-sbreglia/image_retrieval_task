import json, os
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
    def __init__(self, img_dir, labels_json, transform):
        """
        img_dir: folder of images
        labels_json: map {"filename":class_id}
        """
        with open(labels_json) as f:
            self.labels = json.load(f)
        self.images = list(self.labels.keys())
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        fn = self.images[i]
        img = Image.open(os.path.join(self.img_dir, fn)).convert("RGB")
        return self.transform(img), self.labels[fn], fn

