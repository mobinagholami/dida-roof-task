import torch
from torch.utils.data import Dataset
from src.utils.io import load_image, load_mask
import cv2

class RoofDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, size=256):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx])
        img = cv2.resize(img, (self.size, self.size))
        img = img / 255.0
        img = torch.tensor(img).permute(2, 0, 1).float()

        if self.mask_paths is None:
            return img

        mask = load_mask(self.mask_paths[idx])
        mask = cv2.resize(mask, (self.size, self.size))
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask
