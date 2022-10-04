import torch
import cv2
from torch.utils.data import Dataset
from utils import *
from constants import *
import pandas as pd
from sklearn.model_selection import KFold

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



def get_train_val_dataset(print_lengths=False):
    transforms_dataset = get_transforms()
    df = pd.read_csv(TRAIN_CSV_PATH)
    df = df.sample(frac=1).reset_index(drop=True)
    if SANITY_CHECK:
        df = df[:300]
    df['kfold'] = -1
    kf = KFold(n_splits=SPLITS)
    for fold, (trn_,val_) in enumerate(kf.split(X=df.image_id,y=df.digit_sum)):
        df.loc[val_,'kfold'] = fold
    train_df = df[(df['kfold']!=0) & (df['kfold']!=1)].reset_index(drop=True)
    validation_df = df[(df['kfold']==0) | (df['kfold']==1)].reset_index(drop=True)
    if print_lengths:
        print(f"Train set length: {len(train_df)}, validation set length: {len(validation_df)}")
    
    train_dataset = UltraMNISTDataset(train_df,ROOT_DIR,transforms_dataset)
    validation_dataset = UltraMNISTDataset(validation_df,ROOT_DIR,transforms_dataset)
    return train_dataset, validation_dataset