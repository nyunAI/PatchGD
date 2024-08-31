# Import modules
import cv2
import pandas as pd
import os
import shutil
import sys
from tqdm import tqdm

image_size = int(sys.argv[1])
base_path = '../data'
print(base_path)
source_path = f'{base_path}/ultramnist'
destination_path = f'{base_path}/ultramnist_{image_size}'
if os.path.exists(destination_path):
    print("Directory already exists.")
elif not os.path.exists(source_path):
    print("Ultramnist dataset missing.")
else:
    os.mkdir(destination_path)
    os.mkdir(f'{destination_path}/train')
    shutil.copy2(f'{source_path}/train.csv', f'{destination_path}/train.csv') 
    os.mkdir(f'{destination_path}/val')
    shutil.copy2(f'{source_path}/valid.csv', f'{destination_path}/valid.csv') 
    
    train = pd.read_csv(f'{destination_path}/train.csv')
    train.head()
    num_train = len(train)
    print("Number of training sample: ", num_train)
    for i in tqdm(range(num_train)):
        pth = f'{source_path}/train/{train.iloc[i]["image_id"]}.jpeg'
        image = cv2.imread(pth)
        resized_down = cv2.resize(image, (image_size,image_size), interpolation= cv2.INTER_AREA)
        cv2.imwrite(f'{destination_path}/train/{train.iloc[i]["image_id"]}.jpeg', resized_down)
    
    val = pd.read_csv(f'{destination_path}/valid.csv')
    val.head()
    num_val = len(val)
    print("Number of validation sample: ", num_val)
    for i in tqdm(range(num_val)):
        pth = f'{source_path}/val/{val.iloc[i]["image_id"]}.jpeg'
        image = cv2.imread(pth)
        resized_down = cv2.resize(image, (image_size,image_size), interpolation= cv2.INTER_AREA)
        cv2.imwrite(f'{destination_path}/val/{val.iloc[i]["image_id"]}.jpeg', resized_down)