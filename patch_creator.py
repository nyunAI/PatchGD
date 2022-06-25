# Importing required modules
from random import seed, shuffle
import warnings
import gc
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch
import cv2
import pandas as pd
import numpy as np
import pathlib
import os
warnings.filterwarnings('ignore')

# Seeding to help make results reproduceable
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Building custom dataset
class CustomDataset(Dataset):
    def __init__(self, root_dir, X_train, y_train):
        self.root_dir = root_dir
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, index):
        label = self.y_train.iloc[index]
        image_path = f"{self.root_dir}/{self.X_train.iloc[index]}.jpeg"
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, torch.tensor(label)

class CustomPatchset(Dataset):
    def __init__(self, image, targets, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.image = image
        self.targets = targets

    def __len__(self):
        return num_patches*num_patches

    def __getitem__(self, index):
        label = self.targets
        patch_img = self.image
        # return torch.tensor(label), torch.tensor(np.array(patch_id))
        return patch_img, torch.tensor(label), torch.tensor(index)

def run():
    torch.cuda.empty_cache()
    seed_everything(SEED)
    train_df = pd.read_csv(train_csv_path)
    train_df = train_df.sample(1)
    X_train = train_df['id']
    y_train = train_df['digit_sum']
    # print('Data lengths: ', len(X_train), len(y_train))

    # Data transforms
    train_transforms = transforms.Compose([transforms.ToPILImage()])

    # DataLoader
    train_dataset = CustomDataset(root_dir,X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

    for batch_idx, data in enumerate(tqdm(train_loader, total=len(train_loader))):
        images, targets = data
        patch_dataset = CustomPatchset(images, targets, train_transforms)
        patch_loader = DataLoader(patch_dataset, batch_size = PATCH_BATCH_SIZE, shuffle=True, num_workers = num_workers)
        for batch_idx2, data2 in enumerate(tqdm(patch_loader, total=num_patches*num_patches)):
            print("Worked")
            patch_images, patch_labels, patch_ind = data2
            print(patch_labels)
            print(patch_ind)
        gc.collect()

if __name__ == "__main__":
    base_path = pathlib.Path().absolute()
    image_size = 1024
    BATCH_SIZE = 1
    SEED = 42
    PATCH_SIZE = 256
    PATCH_BATCH_SIZE = 1
    stride = 512
    num_patches = ((image_size-PATCH_SIZE)//stride)+1
    num_workers = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_dir = f"{base_path}/dataset/ultra-mnist_{image_size}/train"
    train_csv_path = f'{base_path}/dataset/ultra-mnist_{image_size}/train.csv'
    run()