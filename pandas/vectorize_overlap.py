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
import math
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
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    return lrs

class PandasDataset(Dataset):
    def __init__(self,df,root_dir,transforms=None):
        self.df = df
        self.root_dir = root_dir
        self.transforms = transforms
    def __len__(self):
        return len(self.df)
    def __getitem__(self,index):
        image_id = self.df.iloc[index].image_id
        label = self.df.iloc[index].isup_grade
        image = cv2.imread(f"{self.root_dir}/{image_id}.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, torch.tensor(label)
    
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
    df = pd.read_csv(TRAIN_CSV_PATH)
    train_df = df[df['kfold']!=0]
    val_df = df[df['kfold']==0]
    if SANITY_CHECK:
        train_df = train_df[:SANITY_DATA_LEN]
        val_df = val_df[:SANITY_DATA_LEN]
    if print_lengths:
        print(f"Train set length: {len(train_df)}, validation set length: {len(val_df)}")
    train_dataset = PandasDataset(train_df,TRAIN_ROOT_DIR,transforms_dataset)
    validation_dataset = PandasDataset(val_df,VAL_ROOT_DIR,transforms_dataset)
    return train_dataset, validation_dataset

def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim))).data.shape

class Backbone(nn.Module):
    def __init__(self,baseline,latent_dim):
        super(Backbone,self).__init__()
        # self.encoder = timm.create_model('resnet10t',pretrained=True)
        self.encoder = resnet50(pretrained=True)
        if baseline:
            self.encoder.fc = nn.Linear(2048,NUM_CLASSES)
        else:
            self.encoder.fc = nn.Linear(2048,latent_dim)
    def forward(self,x):
        return self.encoder(x)
 
class CNN_Block(nn.Module):
    def __init__(self,latent_dim,num_classes,num_patches):
        super(CNN_Block,self).__init__()
        self.expected_dim = (BATCH_SIZE,latent_dim,num_patches,num_patches)
        self.layer1 = nn.Sequential(
            nn.Conv2d(latent_dim,latent_dim,3,1,1), 
            nn.ReLU(),
            nn.BatchNorm2d(latent_dim)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(latent_dim,latent_dim,3,2,1), 
            nn.ReLU(),
            nn.BatchNorm2d(latent_dim)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(latent_dim,latent_dim,3,2,1), 
            nn.ReLU(),
            nn.BatchNorm2d(latent_dim)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(latent_dim,latent_dim,3,2,1), 
            nn.ReLU(),
            nn.BatchNorm2d(latent_dim)
        )
        self.dropout = nn.Dropout2d(p=0.2)
        flatten_dim = self.get_final_out_dimension(self.expected_dim)
        self.linear = nn.Linear(flatten_dim,num_classes)

    def get_output_shape(self, model, image_dim):
        return model(torch.rand(*(image_dim))).data.shape

    def get_final_out_dimension(self,shape):
        s = shape
        s = self.get_output_shape(self.layer1,s)
        s = self.get_output_shape(self.layer2,s)
        s = self.get_output_shape(self.layer3,s)
        s = self.get_output_shape(self.layer4,s)
        return np.prod(list(s[1:]))

    def forward(self,x,print_shape=False):
        x = self.layer1(x)
        if print_shape:
            print(x.size())
        x = self.dropout(x)
        x = self.layer2(x)
        if print_shape:
            print(x.size())
        x = self.dropout(x)
        x = self.layer3(x)
        if print_shape:
            print(x.size())
        x = self.dropout(x)
        x = self.layer4(x)
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
    LEARNING_RATE = 1e-3
    ACCELARATOR = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    PERCENT_SAMPLING = 0.1
    BATCH_SIZE = 15
    PATCH_SIZE = 128
    SAVE_MODELS = MONITOR_WANDB
    SCALE_FACTOR = 4
    IMAGE_SIZE = 512*SCALE_FACTOR
    WARMUP_EPOCHS = 2
    EXPERIMENT = "pandas-shared-runs" if not SANITY_CHECK else 'pandas-sanity-gowreesh'
    # EXPERIMENT = "ultracnn-lr-tuning" if not SANITY_CHECK else 'ultracnn-sanity'
    PATCH_BATCHES = math.ceil(1/PERCENT_SAMPLING)
    INNER_ITERATION = PATCH_BATCHES
    LEARNING_RATE_BACKBONE = LEARNING_RATE
    LEARNING_RATE_HEAD = LEARNING_RATE
    RUN_NAME = f"{ACCELARATOR[-1]}-{IMAGE_SIZE}-heirarchical-resnet50+L1-{PERCENT_SAMPLING}-{BATCH_SIZE}-inner_iteration={INNER_ITERATION}_patch=stride={PATCH_SIZE}-linear_warmup={WARMUP_EPOCHS}-16-GB"
    # RUN_NAME = f'{ACCELARATOR[-1]}_cyclic_different'
    
    LATENT_DIMENSION = 256
    NUM_CLASSES = 6
    SEED = 42
    STRIDE = PATCH_SIZE
    NUM_PATCHES = ((IMAGE_SIZE-PATCH_SIZE)//STRIDE) + 1
    NUM_WORKERS = 4
    TRAIN_ROOT_DIR = f'..\\data\\pandas_dataset\\training_images_{IMAGE_SIZE}'
    VAL_ROOT_DIR = TRAIN_ROOT_DIR
    TRAIN_CSV_PATH = f'..\\data\\pandas_dataset\\train_kfold.csv'
    MEAN = [0.9770, 0.9550, 0.9667]
    STD = [0.0783, 0.1387, 0.1006]
    now = datetime.now() # current date and time
    date_time = now.strftime("%d_%m_%Y__%H_%M")
    SANITY_DATA_LEN = None
    MODEL_SAVE_DIR = f"../models/{'sanity' if SANITY_CHECK else 'runs'}/{date_time}_{RUN_NAME}"
    DECAY_FACTOR = 1
    VALIDATION_EVERY = 1
    BASELINE = False
    CONINUE_FROM_LAST = True
    MODEL_LOAD_DIR = '../models/runs/13_12_2022__13_32_4-512-heirarchical-resnet50+L1-0.3-105-inner_iteration=4_patch=stride=128-linear_warmup=2'

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
 
    model1 = Backbone(baseline=False,latent_dim=256)
    model2 = CNN_Block(LATENT_DIMENSION,NUM_CLASSES,NUM_PATCHES)
 
    if CONINUE_FROM_LAST:
        checkpoint = torch.load(f"{MODEL_LOAD_DIR}/last_epoch.pt")
        start_epoch = checkpoint['epoch']
        print(f"Model already trained for {start_epoch} epochs on 512 size images.")
        model1.load_state_dict(checkpoint['model1_weights'])
        # model1.encoder.fc = nn.Linear(2048,LATENT_DIMENSION)
        # print("512 weights backbone loaded with latent modified!!")

    model1.to(ACCELARATOR)
    for param in model1.parameters():
        param.requires_grad = True

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
    
    # optimizer_backbone = optim.Adam(model1.parameters())
    # optimizer_head = optim.Adam(model2.parameters())


    steps_per_epoch = len(train_dataset)//(BATCH_SIZE)
    if len(train_dataset)%BATCH_SIZE!=0:
        steps_per_epoch+=1

    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,WARMUP_EPOCHS*steps_per_epoch,DECAY_FACTOR*EPOCHS*steps_per_epoch)
    
    # scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 
    #                                                      num_warmup_steps=WARMUP_EPOCHS*steps_per_epoch, 
    #                                                      num_training_steps=DECAY_FACTOR*EPOCHS*steps_per_epoch)
    
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3,step_size_up=steps_per_epoch//2,mode="triangular",cycle_momentum=False)
    
    
    # scheduler_backbone = torch.optim.lr_scheduler.CyclicLR(optimizer_backbone, base_lr=1e-5, max_lr=1e-4,step_size_up=steps_per_epoch//2,mode="triangular",cycle_momentum=False)
    # scheduler_head = torch.optim.lr_scheduler.CyclicLR(optimizer_head, base_lr=1e-4, max_lr=1e-3,step_size_up=steps_per_epoch//2,mode="triangular",cycle_momentum=False)

    TRAIN_ACCURACY = []
    TRAIN_LOSS = []
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []
 
    best_validation_loss = float('inf')
    best_validation_accuracy = 0
    
    start_epoch = 0


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

            # optimizer_backbone.zero_grad()
            # optimizer_head.zero_grad()
            
            L1 = torch.zeros((batch_size,LATENT_DIMENSION,NUM_PATCHES,NUM_PATCHES))
            L1 = L1.to(ACCELARATOR)

            patch_dataset = PatchDataset(images,NUM_PATCHES,STRIDE,PATCH_SIZE)
            patch_loader = DataLoader(patch_dataset,batch_size=int(len(patch_dataset)*PERCENT_SAMPLING/2),shuffle=True)

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
            prev_patches = None
            prev_idxs = None
            for inner_iteration, (patches,idxs) in enumerate(patch_loader):
                patches = patches.to(ACCELARATOR)
                patches = patches.reshape(-1,3,PATCH_SIZE,PATCH_SIZE)

                if prev_patches is None and prev_idxs is None:
                    prev_patches = patches
                    prev_idxs = idxs
                    continue
                else:
                    optimizer.zero_grad()

                    # optimizer_backbone.zero_grad()
                    # optimizer_head.zero_grad()
                    pass_patches = torch.cat([prev_patches,patches])
                    pass_idxs = torch.cat([prev_idxs,idxs])
                    prev_patches = patches
                    prev_idxs = idxs
                    L1 = L1.detach()
                    out = model1(pass_patches)
                    out = out.reshape(-1,batch_size, LATENT_DIMENSION)
                    out = torch.permute(out,(1,2,0))
                    row_idx = pass_idxs//NUM_PATCHES
                    col_idx = pass_idxs%NUM_PATCHES
                    L1[:,:,row_idx,col_idx] = out
                    outputs = model2(L1)
                    loss = criterion(outputs,labels)
                    loss.backward()
                    optimizer.step()

                    # optimizer_backbone.step()
                    # optimizer_head.step()

                    train_loss_sub_epoch += loss.item()
                    if inner_iteration + 1 >= INNER_ITERATION:
                        break
            scheduler.step()
            # scheduler_backbone.step()
            # scheduler_head.step()


            # Adding all the losses... Can be modified??
            running_loss_train += train_loss_sub_epoch
            
            # Using the final L1 to make the final set of predictions for accuracy reporting 
            with torch.no_grad():
                outs = model2(L1)
                _,preds = torch.max(outputs,1)
                correct = (preds == labels).sum().item()
                train_correct += correct
            
            lr = get_lr(optimizer)


            # lr = get_lr(optimizer_backbone)
            # lr.extend(get_lr(optimizer_head))
            
            if MONITOR_WANDB:
                wandb.log({f"lrs/lr-{ii}":learning_rate for ii,learning_rate in enumerate(lr)})
                wandb.log({'train_accuracy_step':correct/batch_size,"train_loss_step":train_loss_sub_epoch,'epoch':epoch})

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
                        wandb.log({f"lrs/lr-{ii}":learning_rate for ii,learning_rate in enumerate(lr)})
                        wandb.log({'val_accuracy_step':correct/batch_size,
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