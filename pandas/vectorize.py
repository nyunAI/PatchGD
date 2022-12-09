from random import seed, shuffle
from tabnanny import check
from tracemalloc import start
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
    
class PatchDataset(Dataset):
    def __init__(self,images,num_patches,stride,patch_size):
        self.images = images
        self.num_patches = num_patches
        self.stride = stride
        self.patch_size = patch_size
    def __len__(self):
        return self.num_patches ** 2
    def __getitem__(self,choice):
        i = choice%self.num_patches
        j = choice//self.num_patches
        return self.images[:,:,self.stride*i:self.stride*i+self.patch_size,self.stride*j:self.stride*j+self.patch_size], choice

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
 
class CNN_Block(nn.Module):
    def __init__(self,latent_dim,num_classes):
        super(CNN_Block,self).__init__()
        self.expected_dim = (1,latent_dim,NUM_PATCHES,NUM_PATCHES)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(latent_dim,128,3,2,2), 
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            nn.Conv2d(128,128,3,2,1), 
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            nn.Conv2d(128,64,3,1), 
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        s = get_output_shape(self.conv_layers,self.expected_dim)
        flatten_dim = np.prod(list(s[1:]))
        print(flatten_dim)
        self.linear = nn.Linear(flatten_dim,num_classes)

    def forward(self,x,print_shape=False):
        for layer in self.conv_layers:
            x = layer(x)
            if print_shape:
                print(x.size())
        x = x.reshape(x.shape[0],-1)
        x = self.linear(x)
        if print_shape:
            print(x.size())
        return x
 
 
if __name__ == "__main__":

    MONITOR_WANDB = True
    SANITY_CHECK = False
    EPOCHS = 100
    LEARNING_RATE = 0.001
    ACCELARATOR = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    PERCENT_SAMPLING = 0.3
    BATCH_SIZE = 6
    RUN_NAME = f'2-2048-script-resnet50+L1-{PERCENT_SAMPLING}-{BATCH_SIZE}-16gb-repeated_patch=stride=256-modified_large_head'
    SAVE_MODELS = False
    REPEATED_SAMPLING = False
    SCALE_FACTOR = 4
    IMAGE_SIZE = int(SCALE_FACTOR * 512)
    PATCH_SIZE = 256


    
    LATENT_DIMENSION = 256
    NUM_CLASSES = 28
    SEED = 42
    LEARNING_RATE_BACKBONE = LEARNING_RATE
    LEARNING_RATE_HEAD = LEARNING_RATE
    WARMUP_EPOCHS = 2
    STRIDE = PATCH_SIZE
    NUM_PATCHES = ((IMAGE_SIZE-PATCH_SIZE)//STRIDE) + 1
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
    MODEL_SAVE_DIR = f"../models/{'sanity' if SANITY_CHECK else 'runs'}/{date_time}_{RUN_NAME}"
    DECAY_FACTOR = 1
    VALIDATION_EVERY = 1
    BASELINE = False
    CONINUE_FROM_LAST = False

    if MONITOR_WANDB:
        run = wandb.init(project=EXPERIMENT, entity="gowreesh", reinit=True)
        wandb.run.name = RUN_NAME
        wandb.run.save()
 
    seed_everything(SEED)
    if SAVE_MODELS:
        os.makedirs(MODEL_SAVE_DIR,exist_ok=True)
    
    train_dataset, val_dataset = get_train_val_dataset()
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
    validation_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
 
 
    print(f"Length of train loader: {len(train_loader)},Validation loader: {(len(validation_loader))}")
 
    model1 = Backbone(baseline=False)
    model1.to(ACCELARATOR)
    for param in model1.parameters():
        param.requires_grad = True
 
    
    model2 = CNN_Block(LATENT_DIMENSION,NUM_CLASSES)
    model2.to(ACCELARATOR)
 
    for param in model2.parameters():
        param.requires_grad = True
 
    print(f"Number of patches in one dimenstion: {NUM_PATCHES}, percentage sampling is: {PERCENT_SAMPLING}")
    print(RUN_NAME)
    print(ACCELARATOR)
    
    criterion = nn.CrossEntropyLoss()
    lrs = {
        'head': LEARNING_RATE_HEAD,
        'backbone': LEARNING_RATE_BACKBONE
    }
    parameters = [{'params': model1.parameters(),
                    'lr': lrs['backbone']},
                    {'params': model2.parameters(),
                    'lr': lrs['head']}]
    optimizer = optim.Adam(parameters)
    steps_per_epoch = len(train_dataset)//(BATCH_SIZE)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,WARMUP_EPOCHS*steps_per_epoch,DECAY_FACTOR*EPOCHS*steps_per_epoch)
    
  
 
    TRAIN_ACCURACY = []
    TRAIN_LOSS = []
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []
 
    best_validation_loss = float('inf')
    best_validation_accuracy = 0
    
    start_epoch = 0
    if CONINUE_FROM_LAST:
        checkpoint = torch.load(f"{MODEL_SAVE_DIR}/last_epoch.pt")
        start_epoch = checkpoint['epoch']
        print(f"Model already trained for {start_epoch} epochs.")
        model1.load_state_dict(checkpoint['model1_weights'])
        model2.load_state_dict(checkpoint['model2_weights'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    for epoch in range(start_epoch,start_epoch+EPOCHS):
        print("="*31)
        print(f"{'-'*10} Epoch {epoch+1}/{EPOCHS} {'-'*10}")

        running_loss_train = 0.0
        running_loss_val = 0.0
        train_correct = 0
        val_correct  = 0
        num_train = 0
        num_val = 0
       
 
        model1.train()
        model2.train()
        print("Train Loop!")
        for images,labels in tqdm(train_loader):
            images = images.to(ACCELARATOR)
            labels = labels.to(ACCELARATOR)
            batch_size = labels.shape[0]
            num_train += labels.shape[0]
            optimizer.zero_grad()

            L1 = torch.zeros((batch_size,LATENT_DIMENSION,NUM_PATCHES,NUM_PATCHES))
            L1 = L1.to(ACCELARATOR)

            patch_dataset = PatchDataset(images,NUM_PATCHES,STRIDE,PATCH_SIZE)
            patch_loader = DataLoader(patch_dataset,batch_size=int(len(patch_dataset)*PERCENT_SAMPLING),shuffle=True)

            # Initial filling without gradient engine:
            
            with torch.no_grad():
                for patches, idxs in patch_loader:
                    patches = patches.to(ACCELARATOR)
                    patches = patches.reshape(-1,3,PATCH_SIZE,PATCH_SIZE)
                    out = model1(patches)
                    out = out.reshape(-1,batch_size, LATENT_DIMENSION)
                    out = torch.permute(out,(1,2,0))
                    row_idx = idxs//NUM_PATCHES
                    col_idx = idxs%NUM_PATCHES
                    L1[:,:,row_idx,col_idx] = out
                    
            
            train_loss_sub_epoch = 0
            for patches,idxs in patch_loader:
                optimizer.zero_grad()
                L1 = L1.detach()
                patches = patches.to(ACCELARATOR)
                patches = patches.reshape(-1,3,PATCH_SIZE,PATCH_SIZE)
                out = model1(patches)
                out = out.reshape(-1,batch_size, LATENT_DIMENSION)
                out = torch.permute(out,(1,2,0))
                row_idx = idxs//NUM_PATCHES
                col_idx = idxs%NUM_PATCHES
                L1[:,:,row_idx,col_idx] = out
                outputs = model2(L1)
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                train_loss_sub_epoch += loss.item()
                if not REPEATED_SAMPLING:
                    break
            scheduler.step()


            # Adding all the losses... Can be modified??
            running_loss_train += train_loss_sub_epoch
            
            # Using the final L1 to make the final set of predictions for accuracy reporting 
            with torch.no_grad():
                outs = model2(L1)
                _,preds = torch.max(outputs,1)
                correct = (preds == labels).sum().item()
                train_correct += correct
            
            lr = get_lr(optimizer)
            if MONITOR_WANDB:
                wandb.log({'lr':lr,'train_accuracy_step':correct/batch_size,"train_loss_step":train_loss_sub_epoch,'epoch':epoch})

        print(f"Train Loss: {running_loss_train/num_train} Train Accuracy: {train_correct/num_train}")
        TRAIN_LOSS.append(running_loss_train/num_train)
        TRAIN_ACCURACY.append(train_correct/num_train)
 
 
        # Evaluation Loop!
        if (epoch+1)%VALIDATION_EVERY == 0:
 
            model1.eval()
            model2.eval()
       
            with torch.no_grad():
                print("Validation Loop!")
                for images,labels in tqdm(validation_loader):
                    images = images.to(ACCELARATOR)
                    labels = labels.to(ACCELARATOR)
                    batch_size = labels.shape[0]

                    patch_dataset = PatchDataset(images,NUM_PATCHES,STRIDE,PATCH_SIZE)
                    patch_loader = DataLoader(patch_dataset,int(len(patch_dataset)*PERCENT_SAMPLING),shuffle=True)
                    
                    L1 = torch.zeros((batch_size,LATENT_DIMENSION,NUM_PATCHES,NUM_PATCHES))
                    L1 = L1.to(ACCELARATOR)
 
                    # Filling once to get the final set of predictions
                    with torch.no_grad():
                        for patches, idxs in patch_loader:
                            patches = patches.to(ACCELARATOR)
                            patches = patches.reshape(-1,3,PATCH_SIZE,PATCH_SIZE)
                            out = model1(patches)
                            out = out.reshape(-1,batch_size, LATENT_DIMENSION)
                            out = torch.permute(out,(1,2,0))
                            row_idx = idxs//NUM_PATCHES
                            col_idx = idxs%NUM_PATCHES
                            L1[:,:,row_idx,col_idx] = out
                    
                    outputs = model2(L1)
                    num_val += labels.shape[0]
                    _,preds = torch.max(outputs,1)
                    val_correct += (preds == labels).sum().item()
                    correct = (preds == labels).sum().item()
                    loss = criterion(outputs,labels)
                    l = loss.item()
                    running_loss_val += loss.item()
                    if MONITOR_WANDB:
                        wandb.log({'lr':lr,
                        'val_accuracy_step':correct/batch_size,
                        "val_loss_step":l,
                        'epoch':epoch})
                
                print(f"Validation Loss: {running_loss_val/num_val} Validation Accuracy: {val_correct/num_val}")
                VALIDATION_LOSS.append(running_loss_val/num_val)  
                VALIDATION_ACCURACY.append(val_correct/num_val)
                
                if (running_loss_val/num_val) < best_validation_loss:
                    best_validation_loss = running_loss_val/num_val
                    if SAVE_MODELS:
                        torch.save({
                        'model1_weights': model1.state_dict(),
                        'model2_weights': model2.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'epoch' : epoch+1,
                        }, f"{MODEL_SAVE_DIR}/best_val_loss.pt")
            
                if (val_correct/num_val) > best_validation_accuracy:
                    best_validation_accuracy = val_correct/num_val
                    if SAVE_MODELS:
                        torch.save({
                        'model1_weights': model1.state_dict(),
                        'model2_weights': model2.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'epoch' : epoch+1,
                        }, f"{MODEL_SAVE_DIR}/best_val_accuracy.pt")
        if MONITOR_WANDB:
             wandb.log({"training_loss": running_loss_train/num_train, 
             "training_accuracy": train_correct/num_train, 
             "validation_loss": running_loss_val/num_val, 
             "validation_accuracy": val_correct/num_val,
             'epoch':epoch,
             'best_loss':best_validation_loss,
             'best_accuracy':best_validation_accuracy})
        
        if SAVE_MODELS:
            torch.save({
                    'model1_weights': model1.state_dict(),
                    'model2_weights': model2.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch' : epoch+1,
                    }, f"{MODEL_SAVE_DIR}/last_epoch.pt")
