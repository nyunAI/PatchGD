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
from sklearn.model_selection import train_test_split
import numpy as np
import pathlib
import os
import wandb
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
        self.cnn0 = resnet18(pretrained=True)
        self.cnn3 = nn.Sequential(
            nn.Linear(1000, 256)
        )
        self.cnn1 = nn.Sequential(
            nn.Conv2d(256,256,(3,3),2), # 13 - > 6
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256,(3,3), 1), # 6 - > 4 
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,256,(4,4), 1), #4 - > 1
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Flatten(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,28)
        )
    def forward(self,x = None, L1 = None, images_id = None, patch_ind = None, fill = False, val = False):
        L1cc = L1.clone()
        if val == False:
            out = self.cnn0(x)
            out = self.cnn3(out)
            L1cc[images_id, :, (patch_ind%num_patches),(patch_ind//num_patches)] = out
        if fill == False:
            out = self.cnn1(L1cc)
        return out, L1cc
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
    X_train, X_valid, y_train, y_valid = train_test_split(train_df['id'], train_df['digit_sum'], test_size=0.01, random_state=SEED)
    le.fit(train_df['id'])

    # Data transforms
    train_transforms = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize((image_size, image_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])

    model = CNNModel()
    train_dataset = CustomDataset(root_dir,X_train, y_train, train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    valid_dataset = CustomDataset(root_dir,X_valid, y_valid, train_transforms)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    for params in model.parameters():
        params.requires_grad = True

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_value)
    
    for epoch in range(num_epoch):
        print(f'Epoch: {epoch+1}/{num_epoch}')
        correct = 0
        total = 0
        losses = []
        counter = 0
        for batch_idx, data in enumerate(tqdm(train_loader, total=len(train_loader))):
            if counter % 500 < BATCH_SIZE:
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for batch_idx, valid_data in enumerate(tqdm(valid_loader, total=len(valid_loader), leave = False)):
                        images, targets, image_id = valid_data
                        le2.fit(image_id)
                        L1c = torch.zeros(((len(image_id)),256, num_patches, num_patches), requires_grad = False)
                        
                        patch_dataset = CustomPatchset(images, targets, image_id)
                        patch_loader = DataLoader(patch_dataset, batch_size = PATCH_BATCH_SIZE, shuffle=True, num_workers = num_workers)
                        for batch_idx2, data2 in enumerate(tqdm(patch_loader, total=len(patch_loader), leave = False)):
                            patch_images, patch_target, images_id, patch_ind = data2
                            patch_images = torch.reshape(patch_images, (-1, 3, PATCH_SIZE, PATCH_SIZE))
                            patch_target = torch.reshape(patch_target, (-1,))
                            patch_ind = torch.reshape(patch_ind, (-1,))
                            images_id = torch.reshape(images_id, (-1,))
                            patch_images = patch_images.to(device)
                            targets = targets.to(device)
                            patch_target = patch_target.to(device)
                            L1c = L1c.to(device)
                            L1 = L1c.clone()
                            images_id = le2.transform(images_id)
                            output,L1 = model(patch_images, L1, images_id, patch_ind, fill = True, val = False)
                            L1c = L1.detach().clone()

                            
                        targets = targets.to(device)
                        L1c = L1c.to(device)
                        L1 = L1c.clone()
                        output,_ = model(patch_images, L1, image_id, patch_ind, fill = False, val = True)
                        _, pred = torch.max(output, 1)
                        val_correct += (pred == targets).sum().item()
                        val_total += pred.size(0)
                        gc.collect()
                    print("Validation accuracy: ", (val_correct/val_total)*100)
                    wandb.log({"valid_acc": (val_correct/val_total)*100})
            counter += BATCH_SIZE
            images, targets, image_id = data
            le1.fit(image_id)
            L1c = torch.zeros(((len(image_id)),256, num_patches, num_patches), requires_grad = True)
            
            patch_dataset = CustomPatchset(images, targets, image_id)
            patch_loader = DataLoader(patch_dataset, batch_size = PATCH_BATCH_SIZE, shuffle=True, num_workers = num_workers)
            with torch.no_grad():
                for batch_idx2, data2 in enumerate(tqdm(patch_loader, total=len(patch_loader), leave = False)):
                    patch_images, patch_target, images_id, patch_ind = data2
                    patch_images = torch.reshape(patch_images, (-1, 3, PATCH_SIZE, PATCH_SIZE))
                    patch_target = torch.reshape(patch_target, (-1,))
                    patch_ind = torch.reshape(patch_ind, (-1,))
                    images_id = torch.reshape(images_id, (-1,))
                    patch_images = patch_images.to(device)
                    targets = targets.to(device)
                    patch_target = patch_target.to(device)
                    optimizer.zero_grad()
                    L1c = L1c.to(device)
                    L1 = L1c.clone()
                    images_id = le1.transform(images_id)
                    output,L1 = model(patch_images, L1, images_id, patch_ind, fill = True, val = False)
                    L1c = L1.detach().clone()
            for batch_idx2, data2 in enumerate(tqdm(patch_loader, total=len(patch_loader), leave = False)):
                patch_images, patch_target, images_id, patch_ind = data2
                patch_images = torch.reshape(patch_images, (-1, 3, PATCH_SIZE, PATCH_SIZE))
                patch_target = torch.reshape(patch_target, (-1,))
                patch_ind = torch.reshape(patch_ind, (-1,))
                images_id = torch.reshape(images_id, (-1,))
                patch_images = patch_images.to(device)
                targets = targets.to(device)
                patch_target = patch_target.to(device)
                optimizer.zero_grad()
                L1c = L1c.to(device)
                L1 = L1c.clone()
                images_id = le1.transform(images_id)
                output,L1 = model(patch_images, L1, images_id, patch_ind, fill = False, val = False)
                L1c = L1.detach().clone()
                loss = criterion(output, targets)
                loss.backward(retain_graph = False)
                optimizer.step()
                _, pred = torch.max(output, 1)
                correct += (pred == targets).sum().item()
                total += pred.size(0)
                losses.append(loss.item())
                loss.detach()
                del loss
                gc.collect()
            wandb.log({"loss": np.mean(losses[-6:-1]), "correct": correct * 100.0 / total})
        scheduler.step()
        del losses
if __name__ == "__main__":
    wandb.init(project="ultracnn", entity="gakash2001")
    
    base_path = pathlib.Path().absolute()
    image_size = 1024
    num_epoch = 20
    BATCH_SIZE = 8
    SEED = 42
    PATCH_SIZE = 256
    PATCH_BATCH_SIZE = 32
    stride = 64
    learning_rate = 0.00001
    gamma_value = 0.9
    num_patches = ((image_size-PATCH_SIZE)//stride)+1
    num_workers = 2
    wandb.config = {
    "learning_rate": learning_rate,
    "epochs": num_epoch,
    "batch_size": BATCH_SIZE
    }
    le = preprocessing.LabelEncoder()
    le1 = preprocessing.LabelEncoder()
    run()
    le2 = preprocessing.LabelEncoder()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    root_dir = f"{base_path}/dataset/ultra-mnist_{image_size}/train"
    train_csv_path = f'{base_path}/dataset/ultra-mnist_{image_size}/train.csv'
    print("Running")