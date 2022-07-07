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
import sys
from torchvision.models import resnet18
from sklearn import preprocessing
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Seeding to help make results reproduceable
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CNNModel(nn.Module):
    def __init__ (self):
        super(CNNModel,self).__init__()
        self.cnn0=resnet18(pretrained = True) 
        self.cnn0.fc = nn.Sequential(
            nn.Linear(512, 256)
        )
        self.cnn1=nn.Sequential(
            nn.Conv2d(256,256,(3,3),1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256,(3,3), 2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # nn.Conv2d(256,256,(3,3), 2),
            # nn.ReLU(),
            # nn.BatchNorm2d(256),
            nn.Flatten(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,28)
        )
    def forward(self, x, L1, images_id, patch_ind):
        out=self.cnn0(x)
        L1[images_id, :, (patch_ind%num_patches),(patch_ind//num_patches)] = out
        out=self.cnn1(L1)
        return out, L1
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
    train_df = train_df.sample(80)
    X_train = train_df['id']
    y_train = train_df['digit_sum']
    le.fit(X_train)

    # Data transforms
    train_transforms = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize((image_size, image_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])

    
    model1 = CNNModel()
    train_dataset = CustomDataset(root_dir,X_train, y_train, train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    model1 = model1.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_value)
    
    for epoch in range(num_epoch):
        print(f'Epoch: {epoch+1}/{num_epoch}')
        correct = 0
        total = 0
        losses = []
        for batch_idx, data in enumerate(tqdm(train_loader, total=len(train_loader))):
            images, targets, image_id = data
            le1.fit(image_id)
            L1c = torch.zeros(((len(image_id)),256, num_patches, num_patches), requires_grad = True)
            L1c = L1c.to(device)
            L1 = L1c.clone()
            patch_dataset = CustomPatchset(images, targets, image_id)
            patch_loader = DataLoader(patch_dataset, batch_size = PATCH_BATCH_SIZE, shuffle=True, num_workers = num_workers)

            for batch_idx2, data2 in enumerate(tqdm(patch_loader, total=len(patch_loader), leave = False)):

                for params in model1.parameters():
                    params.requires_grad = True
                print("Sum: ", (torch.sum(L1)))
                L1c.requires_grad = True
                optimizer.zero_grad()
                patch_images, patch_target, images_id, patch_ind = data2
                patch_images = torch.reshape(patch_images, (-1, 3, PATCH_SIZE, PATCH_SIZE))
                patch_target = torch.reshape(patch_target, (-1,))
                patch_ind = torch.reshape(patch_ind, (-1,))
                images_id = torch.reshape(images_id, (-1,))
                patch_images = patch_images.to(device)
                images_id = le1.transform(images_id)
                output, L1= model1(patch_images, L1, images_id, patch_ind)
                print(output.shape, patch_target.shape)
                print("Sum: ", (torch.sum(L1)))
                loss = criterion(output, patch_target)
                print("Requires grad:", targets.requires_grad)
                loss.backward(retain_graph = False)
                optimizer.step()
        scheduler.step()
        train_loss = np.mean(losses)
        train_acc = correct * 100.0         
        print(f'Train Loss: {train_loss}\tTrain Acc: {train_acc}\tLR: {scheduler.get_lr()}',end = '\r')
        del losses
if __name__ == "__main__":
    base_path = pathlib.Path().absolute()
    image_size = 1024
    num_epoch = 1
    BATCH_SIZE = 4
    SEED = 42
    PATCH_SIZE = 256
    PATCH_BATCH_SIZE = 32
    stride = 128
    learning_rate = 0.001
    gamma_value = 0.9
    num_patches = ((image_size-PATCH_SIZE)//stride)+1
    print(num_patches)
    num_workers = 0
    le = preprocessing.LabelEncoder()
    le1 = preprocessing.LabelEncoder()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_dir = f"{base_path}/dataset/ultra-mnist_{image_size}/train"
    train_csv_path = f'{base_path}/dataset/ultra-mnist_{image_size}/train.csv'
    print("Running")
    run()