# Importing required modules
from random import seed, shuffle
import warnings
import gc
from tqdm import tqdm
import torchvision
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
from torchvision.models import resnet18
from sklearn import preprocessing
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
        self.X_train = X_train
        self.y_train = y_train
        self.transform = transform

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, index):
        label = self.y_train.iloc[index]
        image_path = f"{self.root_dir}/{self.X_train.iloc[index]}.jpeg"
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        image_id = self.X_train.iloc[index]
        return image, torch.tensor(label), torch.tensor(le.transform([image_id]).item())

class CustomPatchset(Dataset):
    def __init__(self, image, targets, image_id):
        self.root_dir = root_dir
        self.image = image
        self.targets = targets
        self.image_id = image_id

    def __len__(self):
        return num_patches*num_patches

    def __getitem__(self, index):
        label = self.targets
        patch_img = self.image
        images_id = self.image_id
        patch_ind = index
        patch_img = torchvision.transforms.functional.crop(self.image, (patch_ind//num_patches)*stride,(patch_ind%num_patches)*stride, PATCH_SIZE, PATCH_SIZE)
        return patch_img, torch.tensor(label), torch.tensor(images_id), torch.tensor(patch_ind).repeat(BATCH_SIZE)
def run():
    torch.cuda.empty_cache()
    seed_everything(SEED)
    train_df = pd.read_csv(train_csv_path)
    # train_df = train_df.sample(8)
    X_train = train_df['id']
    y_train = train_df['digit_sum']
    le.fit(X_train)

    # Data transforms
    train_transforms = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize((image_size, image_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])

    model = resnet18(pretrained = True)
    print(model)
    model.fc = nn.Sequential(
        nn.Linear(512, 256)
    )
    # DataLoader
    train_dataset = CustomDataset(root_dir,X_train, y_train, train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    model.to(device)
    for epoch in range(num_epoch):
        print(f'Epoch: {epoch+1}/{num_epoch}')
        for batch_idx, data in enumerate(tqdm(train_loader, total=len(train_loader))):
            images, targets, image_id = data
            patch_dataset = CustomPatchset(images, targets, image_id)
            patch_loader = DataLoader(patch_dataset, batch_size = PATCH_BATCH_SIZE, shuffle=True, num_workers = num_workers)
            for batch_idx2, data2 in enumerate(tqdm(patch_loader, total=len(patch_loader), leave = False)):
                patch_images, patch_target, images_id, patch_ind = data2
                patch_images = torch.reshape(patch_images, (-1, 3, PATCH_SIZE, PATCH_SIZE))
                patch_target = torch.reshape(patch_target, (-1,))
                patch_ind = torch.reshape(patch_ind, (-1,))
                images_id = torch.reshape(images_id, (-1,))
                output = model(patch_images)
                # print(output.shape)
                gc.collect()

if __name__ == "__main__":
    base_path = pathlib.Path().absolute()
    image_size = 1024
    num_epoch = 1
    BATCH_SIZE = 4
    SEED = 42
    PATCH_SIZE = 256
    PATCH_BATCH_SIZE = 3
    stride = 64
    num_patches = ((image_size-PATCH_SIZE)//stride)+1
    num_workers = 2
    le = preprocessing.LabelEncoder()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_dir = f"{base_path}/dataset/ultra-mnist_{image_size}/train"
    train_csv_path = f'{base_path}/dataset/ultra-mnist_{image_size}/train.csv'
    print("Running")
    run()