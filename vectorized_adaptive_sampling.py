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
from torchvision.models import resnet50
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
        return image_id, image, torch.tensor(digit_sum)
    
class PatchDatasetValidation(Dataset):
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


class PatchDatasetTrain(Dataset):
    def __init__(self,images,image_ids,num_patches,stride,patch_size):
        self.images = images
        self.num_patches = num_patches
        self.stride = stride
        self.patch_size = patch_size
        self.num_images = self.images.shape[0]
        self.image_ids = image_ids
        self.refill()
    
    def __len__(self):
        return self.max_list_length
    
    def __getitem__(self,index):
        choices = self.padded_valid_index[:,index]
        patches = []
        for image_num,choice in enumerate(choices):
          i = choice%self.num_patches
          j = choice//self.num_patches
          patches.append(self.images[image_num,:,self.stride*i:self.stride*i+self.patch_size,self.stride*j:self.stride*j+self.patch_size])
        return torch.stack(patches),torch.tensor(choices)


    def findMaxList(self,valid_index=None):
      valid_index = self.valid_idxs if valid_index is None else valid_index
      list_len = [len(i) for i in valid_index]
      return max(list_len)

    def refill(self,print_outs=False):
        self.idxs = self.getIndexes()
        if print_outs:
            print(self.idxs)
        self.forbidden_idxs = self.getForbiddenBatch()
        if print_outs:
            print(self.forbidden_idxs)
        self.valid_idxs = self.setDiff()
        if print_outs:
            print(self.valid_idxs)
        self.max_list_length = self.findMaxList()
        self.padded_valid_index = self.shuffleAndPadIndex()
        if print_outs:
            print(self.padded_valid_index)

    def getIndexes(self,num_images=None,num_patches=None):
        num_images = self.num_images if num_images is None else num_images
        num_patches = self.num_patches ** 2 if num_patches is None else num_patches
        x = np.tile(np.arange(num_patches),(num_images,1))
        return x
    

    def isInvalidPatch(self,patch,threshold=1e-2):
        if patch.max() - patch.min() < threshold:
            return True
        else:
            return False
     
    def getForbiddenImage(self,image=None):
        forbidden_patches = []
        for choice in range(self.num_patches**2):
            i = choice%self.num_patches
            j = choice//self.num_patches
            patch = image[:,self.stride*i:self.stride*i+self.patch_size,self.stride*j:self.stride*j+self.patch_size]
            if self.isInvalidPatch(patch):
                forbidden_patches.append(choice)
        return forbidden_patches
        
        
    def getForbiddenBatch(self,num_images=None,num_patches=None):
        num_images = self.num_images if num_images is None else num_images
        x = []
        for i in range(num_images):
            # if self.image_ids[i] not in FORBIDDEN_INDEXES.keys():
            #     # Find the invalid patches from the image
            #     forbidden_index_image = self.getForbiddenImage(self.images[i])
            #     FORBIDDEN_INDEXES[self.image_ids[i]] = forbidden_index_image

            # x.append(FORBIDDEN_INDEXES[self.image_ids[i]])
            x.append([])
        return x

    def setDiff(self,idx=None,forbidden=None):
        idx = self.idxs if idx is None else idx
        forbidden = self.forbidden_idxs if forbidden is None else forbidden
        assert len(idx) == len(forbidden)
        final_indexes = []
        for i in range(len(idx)):
            final_indexes.append(np.setdiff1d(idx[i],forbidden[i]))
        return final_indexes
    
    def shuffleAndPadIndex(self,valid_index=None):
        valid_index = self.valid_idxs if valid_index is None else valid_index
        x = []
        for i in range(len(valid_index)):
            np.random.shuffle(valid_index[i])
            rem_patches = self.max_list_length - len(valid_index[i])
            x.append(np.pad(valid_index[i], (0,rem_patches), 'reflect'))
        return np.array(x)

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
    ACCELARATOR = 'cuda:7' if torch.cuda.is_available() else 'cpu'
    PERCENT_SAMPLING = 0.3
    BATCH_SIZE = 16
    PATCH_SIZE = 128
    SAVE_MODELS = False #MONITOR_WANDB
    SCALE_FACTOR = 4
    IMAGE_SIZE = 512*SCALE_FACTOR
    WARMUP_EPOCHS = 6
    EXPERIMENT = "ultracnn-shared-runs-gowreesh" if not SANITY_CHECK else 'ultracnn-sanity'
    # EXPERIMENT = "ultracnn-lr-tuning" if not SANITY_CHECK else 'ultracnn-sanity'
    PATCH_BATCHES = math.ceil(1/PERCENT_SAMPLING)
    INNER_ITERATION = PATCH_BATCHES
    LEARNING_RATE_BACKBONE = LEARNING_RATE
    LEARNING_RATE_HEAD = LEARNING_RATE
    RUN_NAME = f"{ACCELARATOR[-1]}-{IMAGE_SIZE}-adaptive-sampling-removed-sanity-heirarchical-resnet50+L1-{PERCENT_SAMPLING}-{BATCH_SIZE}-inner_iteration={INNER_ITERATION}_patch=stride={PATCH_SIZE}-larger_head_-cosine_annealing_more_warmup_{WARMUP_EPOCHS}"
    
    
    FORBIDDEN_INDEXES = {}
    LATENT_DIMENSION = 256
    NUM_CLASSES = 28
    SEED = 42
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
    SANITY_DATA_LEN = 64
    MODEL_SAVE_DIR = f"../models/{'sanity' if SANITY_CHECK else 'runs'}/{date_time}_{RUN_NAME}"
    DECAY_FACTOR = 2
    VALIDATION_EVERY = 1
    BASELINE = False
    CONINUE_FROM_LAST = True
    MODEL_LOAD_DIR = '../models/runs/30_10_2022__06_01_3-512-modified_head-resnet50+L1-0.3-256-repeated_patch=stride=128-linear-lr-0.001'

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
        checkpoint = torch.load(f"{MODEL_LOAD_DIR}/best_val_accuracy.pt")
        start_epoch = checkpoint['epoch']
        print(f"Model already trained for {start_epoch} epochs on 512 size images.")
        model1.load_state_dict(checkpoint['model1_weights'])
        if LATENT_DIMENSION!=256:
            model1.encoder.fc = nn.Linear(2048,LATENT_DIMENSION)
            print("512 weights backbone loaded with latent modified!!")

    model1.to(ACCELARATOR)
    for param in model1.parameters():
        param.requires_grad = True

    model2.to(ACCELARATOR)
    for param in model2.parameters():
        param.requires_grad = True
        
    print(f"Number of patches in one dimenstion: {NUM_PATCHES}, percentage sampling is: {PERCENT_SAMPLING}")
    print(RUN_NAME)
    print(ACCELARATOR)
    # print(train_dataset[0][1].shape)
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

    # scheduler = transformers.get_linear_schedule_with_warmup(optimizer,WARMUP_EPOCHS*steps_per_epoch,DECAY_FACTOR*EPOCHS*steps_per_epoch)
    
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 
                                                         num_warmup_steps=WARMUP_EPOCHS*steps_per_epoch, 
                                                         num_training_steps=DECAY_FACTOR*EPOCHS*steps_per_epoch)
    
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
        for image_ids,images,labels in tqdm(train_loader):
            images = images.to(ACCELARATOR)
            labels = labels.to(ACCELARATOR)
            batch_size = labels.shape[0]
            num_train += labels.shape[0]
            optimizer.zero_grad()

            # optimizer_backbone.zero_grad()
            # optimizer_head.zero_grad()
            
            L1 = torch.zeros((batch_size,LATENT_DIMENSION,NUM_PATCHES,NUM_PATCHES))
            L1 = L1.to(ACCELARATOR)

            patch_dataset_train = PatchDatasetTrain(images,image_ids,NUM_PATCHES,STRIDE,PATCH_SIZE)
            patch_loader_train = DataLoader(patch_dataset_train,batch_size=int(NUM_PATCHES*NUM_PATCHES*PERCENT_SAMPLING))

            patch_dataset_val = PatchDatasetValidation(images,NUM_PATCHES,STRIDE,PATCH_SIZE)
            patch_loader_val = DataLoader(patch_dataset_val,batch_size=int(NUM_PATCHES*NUM_PATCHES*PERCENT_SAMPLING))


            # Initial filling without gradient engine:
            
            with torch.no_grad():
                for patches, idxs in patch_loader_val:
                    patches = patches.to(ACCELARATOR)
                    patches = patches.reshape(-1,3,PATCH_SIZE,PATCH_SIZE)
                    out = model1(patches)
                    out = out.reshape(-1,batch_size, LATENT_DIMENSION)
                    out = torch.permute(out,(1,2,0))
                    row_idx = idxs//NUM_PATCHES
                    col_idx = idxs%NUM_PATCHES
                    L1[:,:,row_idx,col_idx] = out
                    
                    
            
            train_loss_sub_epoch = 0
            for inner_iteration, (patches,idxs) in enumerate(patch_loader_train):
                optimizer.zero_grad()

                # optimizer_backbone.zero_grad()
                # optimizer_head.zero_grad()

                L1 = L1.detach()
                patches = patches.to(ACCELARATOR)
                patches = patches.reshape(-1,3,PATCH_SIZE,PATCH_SIZE)
                out = model1(patches)
                out = out.reshape(-1,batch_size, LATENT_DIMENSION)
                out = torch.permute(out,(1,2,0))
                row_idx = idxs//NUM_PATCHES
                col_idx = idxs%NUM_PATCHES
                for image_num in range(batch_size):
                    L1[image_num,:,row_idx[:,image_num].long(),col_idx[:,image_num].long()] = out[image_num,:,:]
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
            
            del patch_dataset_train,patch_loader_train

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
            if SANITY_CHECK:
                print("Number of Forbidden indexes: ",len(FORBIDDEN_INDEXES))
                
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
                for image_ids,images,labels in tqdm(validation_loader):
                    images = images.to(ACCELARATOR)
                    labels = labels.to(ACCELARATOR)
                    batch_size = labels.shape[0]

                    patch_dataset = PatchDatasetValidation(images,NUM_PATCHES,STRIDE,PATCH_SIZE)
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