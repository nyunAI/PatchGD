from datetime import datetime
import torch
from utils import DatasetName

num_classes = {
    DatasetName.PANDAS: 6,
    DatasetName.UMNIST: 10
}
mean = {
    DatasetName.PANDAS: [0.9770, 0.9550, 0.9667],
    DatasetName.UMNIST: [0.1307,0.1307,0.1307]
}

std = {
    DatasetName.PANDAS: [0.0783, 0.1387, 0.1006],
    DatasetName.UMNIST: [0.3081,0.3081,0.3081]
}
DATASET_NAME = DatasetName.PANDAS #umnist
DEVICE_ID = 0 
now = datetime.now() 
date_time = now.strftime("%d_%m_%Y__%H_%M")

MONITOR_WANDB = True 
SAVE_MODELS =  MONITOR_WANDB
BASELINE = True
SANITY_CHECK = False
EPOCHS = 100
LEARNING_RATE = 1e-3
ACCELARATOR = f'cuda:0' if torch.cuda.is_available() else 'cpu'
SCALE_FACTOR = 1
IMAGE_SIZE = int(SCALE_FACTOR * 512)
BATCH_SIZE = 128 

RUN_NAME = f'{DEVICE_ID}-{IMAGE_SIZE}-{BATCH_SIZE}-resnet50-baseline-datetime_{date_time}' 

NUM_CLASSES = num_classes[DATASET_NAME]  
SEED = 42
LEARNING_RATE_BACKBONE = LEARNING_RATE
LEARNING_RATE_HEAD = LEARNING_RATE
WARMUP_EPOCHS = 2
NUM_WORKERS = 4
TRAIN_ROOT_DIR = f''
VAL_ROOT_DIR = TRAIN_ROOT_DIR
TRAIN_CSV_PATH = f'./train_kfold.csv'
VAL_CSV_PATH = f'./train_kfold.csv'
MEAN = mean[DATASET_NAME]
STD = std[DATASET_NAME]
SANITY_DATA_LEN = None
MODEL_SAVE_DIR = f"../{RUN_NAME}"
EXPERIMENT = ''
DECAY_FACTOR = 2
