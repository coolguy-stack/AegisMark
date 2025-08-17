import os, glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class ImageFolderDataset(Dataset):
    def __init__(self, root, image_size=256):
        self.paths = []
        for ext in ('*.jpg','*.jpeg','*.png'):
            self.paths += glob.glob(os.path.join(root, '**', ext), recursive=True)
        self.tx = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        return self.tx(img)