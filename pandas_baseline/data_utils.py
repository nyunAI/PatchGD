import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
from utils import get_transforms


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


def get_train_val_dataset(train_csv_path,
                        sanity_check,
                        sanity_data_len,
                        train_root_dir,
                        val_root_dir,
                        image_size,
                        mean,
                        std,
                        print_lengths=True):
    transforms_dataset = get_transforms(image_size=image_size,mean=mean,std=std)
    df = pd.read_csv(train_csv_path)
    train_df = df[df['kfold']!=0]
    val_df = df[df['kfold']==0]
    if sanity_check:
        train_df = train_df[:sanity_data_len]
        val_df = val_df[:sanity_data_len]
    if print_lengths:
        print(f"Train set length: {len(train_df)}, validation set length: {len(val_df)}")
    train_dataset = PandasDataset(train_df,train_root_dir,transforms_dataset)
    validation_dataset = PandasDataset(val_df,val_root_dir,transforms_dataset)
    return train_dataset, validation_dataset
