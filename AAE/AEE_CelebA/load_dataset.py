"""To load the Celeba Dataset downloaded form kaggle """
import torch
from torch.utils.data import Dataset
# from torchvision.utils import make_grid
# from torchvision import transforms

import pandas as pd

class CelebA(Dataset):
    """
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        self.attr_frame =  pd.read_csv(csv_file, low_memory=False)

    def __len__(self):
        return len(self.attr_frame)

    def __getitem__(self, index):
        attributes = self.attr_frame.iloc[index, 1:]
        img_name = self.attr_frame.iloc[index, 0]
        img_path = os.path.join(self.root + "/" + img_name)
        image = io.imread(img_path)

        attr_tensor = torch.tensor(attributes)

        if self.transform:
            image = self.transform(image)

        return (image, attr_tensor)
