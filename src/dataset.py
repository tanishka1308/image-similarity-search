import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

class RobustSyntheticTripletDataset(Dataset):
    def __init__(self, image_dir):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(valid_exts)
        ]
        if len(self.paths) < 2:
            raise ValueError(f"Directory {image_dir} must have at least 2 images.")

        self.augmenter = T.Compose([
            T.Resize((224, 224)),
            T.RandomPerspective(distortion_scale=0.35, p=0.7),
            T.RandomRotation(degrees=(-15, 15)),
            T.RandomResizedCrop(size=(224, 224), scale=(0.75, 1.0), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.RandomErasing(p=0.6, scale=(0.05, 0.25), ratio=(0.3, 3.3), value="random"),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.clean_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        base_img = Image.open(self.paths[idx]).convert('RGB')
        anchor = self.clean_transform(base_img)
        positive = self.augmenter(base_img)

        # Pick a different image for the negative sample
        neg_idx = random.choice([i for i in range(len(self.paths)) if i != idx])
        neg_img = Image.open(self.paths[neg_idx]).convert('RGB')
        negative = self.clean_transform(neg_img)

        return anchor, positive, negative