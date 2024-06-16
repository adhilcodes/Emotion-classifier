from torchvision import transforms
from PIL import Image
import torch

class EmotionData:
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
