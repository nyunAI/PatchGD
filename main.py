# Importing required modules
from calendar import EPOCH
from random import seed
import warnings
import gc
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import torch.optim as optim
import torch.nn as nn
import torch
import cv2
import pandas as pd
import numpy as np
import pathlib
import logging
import os
import sys
from torchvision.models import resnet50
import yaml
import wandb
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
    def __init__(self, root_dir, X_train, y_train, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, index):
        label = self.y_train.iloc[index]
        image_path = f"{self.root_dir}/{self.X_train.iloc[index]}.jpeg"
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image, torch.tensor(label)

def run():
    torch.cuda.empty_cache()
    seed_everything(SEED)

    train_df = pd.read_csv(train_csv_path)
    # train_df = train_df.sample(10)

    # building training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(train_df['id'], train_df['digit_sum'], test_size=0.1, random_state=SEED)
    print('Data lengths: ', len(X_train), len(X_valid), len(y_train), len(y_valid))

    # Data transforms
    train_transforms = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize((image_size, image_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])
    test_transforms = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize((image_size, image_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])

    # DataLoader
    train_dataset = CustomDataset(root_dir,X_train, y_train,train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    valid_dataset = CustomDataset(root_dir,X_valid, y_valid,test_transforms)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)


if __name__ == "__main__":
    # This helps make all other paths relative
    base_path = pathlib.Path().absolute()

    # Input of the required hyperparameters
    BATCH_SIZE = 4
    image_size = 1024
    # Fixed hyperparameters
    SEED = 42
    num_workers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_dir = f"{base_path}/dataset/ultra-mnist_{image_size}/train"
    if not os.path.exists(root_dir):
        print("Dataset missing.")
    train_csv_path = f'{base_path}/dataset/ultra-mnist_{image_size}/train.csv'

    run()
