import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from constants import *
import pandas as pd
from sklearn.model_selection import KFold
from utils import get_transforms

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

 
 
    
def get_train_val_loaders(print_lengths=False):
    df = pd.read_csv(TRAIN_CSV_PATH)
    if SANITY_CHECK:
        df = df[:50]
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    kf = KFold(n_splits=SPLITS)
    for fold, (trn_,val_) in enumerate(kf.split(X=df.image_id,y=df.digit_sum)):
        df.loc[val_,'kfold'] = fold
    train_df = df[(df['kfold']!=0) & (df['kfold']!=1)].reset_index(drop=True)
    validation_df = df[(df['kfold']==0) | (df['kfold']==1)].reset_index(drop=True)
    if print_lengths:
        print(f"Train set length: {len(train_df)}, validation set length: {len(validation_df)}")
    transforms_dataset = get_transforms()
    train_dataset = UltraMNISTDataset(train_df,ROOT_DIR,transforms_dataset)
    train_loader = DataLoader(train_dataset,batch_size=TRAIN_BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
 
    validation_dataset = UltraMNISTDataset(validation_df,ROOT_DIR,transforms_dataset)
    validation_loader = DataLoader(validation_dataset,batch_size=INFER_BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
    
    if print_lengths:
        print("Train loader check!")
        for i in train_loader:
            print(i[0].shape,i[1].shape)
            break
    
        print("Validation loader check!")
        for i in validation_loader:
            print(i[0].shape,i[1].shape)
            break
 
        print(f"Length of train loader: {len(train_loader)},Validation loader: {(len(validation_loader))}")
 
    return train_loader, validation_loader