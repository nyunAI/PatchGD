# Import modules
import cv2
import pandas as pd
import os
import shutil
import sys

image_size = int(sys.argv[1])
base_path = '../data'
print(base_path)
source_path = f'{base_path}/ultramnist_sample'
destination_path = f'{base_path}/ultra-mnist_{image_size}'
if os.path.exists(destination_path):
    print("Directory already exists.")
elif not os.path.exists(source_path):
    print("Ultramnist dataset missing.")
else:
    os.mkdir(destination_path)
    os.mkdir(f'{destination_path}/train')
    shutil.copy2(f'{source_path}/train.csv', f'{destination_path}/train.csv') 
    train = pd.read_csv(f'{destination_path}/train.csv')
    train.head()
    num_train = len(train)
    print("Number of training sample: ", num_train)
    for i in range(num_train):
        pth = f'{source_path}/train/{train.iloc[i]["image_id"]}.jpeg'
        print((i+1), end="\r")
        image = cv2.imread(pth)
        resized_down = cv2.resize(image, (image_size,image_size), interpolation= cv2.INTER_AREA)
        cv2.imwrite(f'{destination_path}/train/{train.iloc[i]["image_id"]}.jpeg', resized_down)