from random import seed, shuffle
import warnings
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch
import cv2
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import wandb
import pathlib
import os
import time
import timm
from torchvision.models import resnet18, resnet50
import matplotlib.pyplot as plt
from datetime import datetime
import transformers
warnings.filterwarnings('ignore')
 

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transforms():
    return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN,STD)])


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class UltraMNISTDataset(Dataset):
    def __init__(self,df,root_dir,transforms=None):
        self.df = df
        self.root_dir = root_dir
        self.transforms = transforms
    def __len__(self):
        return len(self.df)
    def __getitem__(self,index):
        image_id = self.df.iloc[index].image_id
        digit_sum = self.df.iloc[index].digit_sum
        image = cv2.imread(f"{self.root_dir}/{image_id}.jpeg")
        if self.transforms is not None:
            image = self.transforms(image)
        return image, torch.tensor(digit_sum)

def get_train_val_dataset(print_lengths=True):
    transforms_dataset = get_transforms()
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = pd.read_csv(VAL_CSV_PATH)
    val_df = val_df.sample(frac=1).reset_index(drop=True)
    if SANITY_CHECK:
        train_df = train_df[:SANITY_DATA_LEN]
        val_df = val_df[:SANITY_DATA_LEN]
    if print_lengths:
        print(f"Train set length: {len(train_df)}, validation set length: {len(val_df)}")
    train_dataset = UltraMNISTDataset(train_df,TRAIN_ROOT_DIR,transforms_dataset)
    validation_dataset = UltraMNISTDataset(val_df,VAL_ROOT_DIR,transforms_dataset)
    return train_dataset, validation_dataset


def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim))).data.shape

class Backbone(nn.Module):
    def __init__(self,baseline):
        super(Backbone,self).__init__()
        # self.encoder = timm.create_model('resnet10t',pretrained=True)
        self.encoder = resnet50(pretrained=True)
        if baseline:
            self.encoder.fc = nn.Linear(2048,NUM_CLASSES)
        else:
            self.encoder.fc = nn.Linear(2048,LATENT_DIMENSION)
    def forward(self,x):
        return self.encoder(x)
 
 
 
if __name__ == "__main__":

    MONITOR_WANDB = True
    BASELINE = True
    SANITY_CHECK = False
    EPOCHS = 100
    LEARNING_RATE = 0.001
    ACCELARATOR = 'cuda:5' if torch.cuda.is_available() else 'cpu'
    RUN_NAME = f'5-2048-script-resnet50-baseline-bs=1'
    BATCH_SIZE = 2

    
    SCALE_FACTOR = 4
    IMAGE_SIZE = int(SCALE_FACTOR * 512)
    LATENT_DIMENSION = 256
    NUM_CLASSES = 28
    SEED = 42
    LEARNING_RATE_BACKBONE = LEARNING_RATE
    LEARNING_RATE_HEAD = LEARNING_RATE
    WARMUP_EPOCHS = 2
    NUM_WORKERS = 4
    TRAIN_ROOT_DIR = f'../data/ultramnist_{IMAGE_SIZE//SCALE_FACTOR}/train'
    VAL_ROOT_DIR = f'../data/ultramnist_{IMAGE_SIZE//SCALE_FACTOR}/val'
    MNIST_ROOT_DIR = '../data/mnist'
    TRAIN_CSV_PATH = f'../data/ultramnist_{IMAGE_SIZE//SCALE_FACTOR}/train.csv'
    VAL_CSV_PATH = f'../data/ultramnist_{IMAGE_SIZE//SCALE_FACTOR}/valid.csv'
    MEAN = [0.1307,0.1307,0.1307]
    STD = [0.3081,0.3081,0.3081]
    now = datetime.now() # current date and time
    date_time = now.strftime("%d_%m_%Y__%H_%M")
    SANITY_DATA_LEN = 512
    EXPERIMENT = "ultracnn-shared-runs-gowreesh" if not SANITY_CHECK else 'ultracnn-sanity-gowreesh'
    MODEL_SAVE_DIR = f'../models/4_5/{date_time}_{RUN_NAME}_{IMAGE_SIZE}' if not SANITY_CHECK else f'../models/4_5/sanity/{date_time}_{RUN_NAME}_{IMAGE_SIZE}'
    DECAY_FACTOR = 1

   
    if MONITOR_WANDB:
        run = wandb.init(project=EXPERIMENT, entity="gowreesh", reinit=True)
        wandb.run.name = RUN_NAME
        wandb.run.save()
    
 
    seed_everything(SEED)
 
    
    train_dataset, val_dataset = get_train_val_dataset()
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
    validation_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
 
    print("Train loader check!")
    for i in train_loader:
        print(i[0].shape,i[1].shape)
        break
 
    print("Validation loader check!")
    for i in validation_loader:
        print(i[0].shape,i[1].shape)
        break
 
    print(f"Length of train loader: {len(train_loader)},Validation loader: {(len(validation_loader))}")
 
    model1 = Backbone(baseline=BASELINE)
    model1.to(ACCELARATOR)
    for param in model1.parameters():
        param.requires_grad = True
 
 
 
    print(f"Baseline model:")
    
    
    criterion = nn.CrossEntropyLoss()
    lrs = {
        'head': LEARNING_RATE_HEAD,
        'backbone': LEARNING_RATE_BACKBONE
    }
    parameters = [{'params': model1.parameters(),
                    'lr': lrs['backbone']},
                    ]
    optimizer = optim.Adam(parameters)
    steps_per_epoch = len(train_dataset)//(BATCH_SIZE)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,WARMUP_EPOCHS*steps_per_epoch,EPOCHS*steps_per_epoch)
    
  
 
    TRAIN_ACCURACY = []
    TRAIN_LOSS = []
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []
 
    best_validation_loss = float('inf')
    best_validation_accuracy = 0
 
 
    for epoch in range(EPOCHS):
        print("="*31)
        print(f"{'-'*10} Epoch {epoch+1}/{EPOCHS} {'-'*10}")

        running_loss_train = 0.0
        running_loss_val = 0.0
        train_correct = 0
        val_correct  = 0
        num_train = 0
        num_val = 0
       
 
        model1.train()
        print("Train Loop!")
        for images,labels in tqdm(train_loader):
            images = images.to(ACCELARATOR)
            labels = labels.to(ACCELARATOR)
            batch_size = labels.shape[0]
            num_train += labels.shape[0]
            optimizer.zero_grad()
            outputs = model1(images)
            if torch.isnan(outputs).any():
                print("output has nan")
            _,preds = torch.max(outputs,1)
            train_correct += (preds == labels).sum().item()
            correct = (preds == labels).sum().item()
            loss = criterion(outputs,labels)
            l = loss.item()
            running_loss_train += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr = get_lr(optimizer)
            if MONITOR_WANDB:
                wandb.log({'lr':lr,'train_accuracy_step':correct/batch_size,"train_loss_step":l,'epoch':epoch})

        print(f"Train Loss: {running_loss_train/num_train} Train Accuracy: {train_correct/num_train}")
        TRAIN_LOSS.append(running_loss_train/num_train)
        TRAIN_ACCURACY.append(train_correct/num_train)
 
 
        # Evaluation Loop!
        val_accr = 0.0
        val_lossr = 0.0
        if (epoch+1)%1 == 0:
 
            model1.eval()
       
            with torch.no_grad():
                print("Validation Loop!")
                for images,labels in tqdm(validation_loader):
                    images = images.to(ACCELARATOR)
                    labels = labels.to(ACCELARATOR)
                    batch_size = labels.shape[0]

                    outputs = model1(images)
                    if torch.isnan(outputs).any():
                        print("L1 has nan")
                    num_val += labels.shape[0]
                    _,preds = torch.max(outputs,1)
                    val_correct += (preds == labels).sum().item()
                    correct = (preds == labels).sum().item()
                    loss = criterion(outputs,labels)
                    l = loss.item()
                    running_loss_val += loss.item()
                    if MONITOR_WANDB:
                        wandb.log({'lr':lr,'val_accuracy_step':correct/batch_size,"val_loss_step":l,"epoch":epoch})
                print(f"Validation Loss: {running_loss_val/num_val} Validation Accuracy: {val_correct/num_val}")
 
                VALIDATION_LOSS.append(running_loss_val/num_val)  
                VALIDATION_ACCURACY.append(val_correct/num_val)
        if MONITOR_WANDB:
             wandb.log({"training_loss": running_loss_train/num_train, "training_accuracy": train_correct/num_train, "validation_loss": running_loss_val/num_val, "validation_accuracy": val_correct/num_val})
        
