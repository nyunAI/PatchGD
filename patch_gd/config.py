from datetime import datetime
import torch
import math
from enum import Enum

class model_config(Enum):
    ORIGINAL = 0
    SMALLER = 1
    LARGER = 2
    SMALLER_FEAT = 3
    LARGER_FEAT = 4

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
MAIN_RUN = True

MONITOR_WANDB = False ######################################
SANITY_CHECK = False
EPOCHS = 100
LEARNING_RATE = 1e-3
ACCELARATOR = 'cuda:0' if torch.cuda.is_available() else 'cpu'
PERCENT_SAMPLING = 1/8 
GRAD_ACCUM =  True
BATCH_SIZE = 196 
MEMORY = '16' 
PATCH_SIZE = 32
SAVE_MODELS = False
SCALE_FACTOR = 1
IMAGE_SIZE = int(SCALE_FACTOR * 512)
WARMUP_EPOCHS = 2
EXPERIMENT = "EXPERIMENT_NAME" 
PATCH_BATCHES = math.ceil(1/PERCENT_SAMPLING)
INNER_ITERATION = PATCH_BATCHES
EPSILON = INNER_ITERATION if GRAD_ACCUM else 1 
LEARNING_RATE_BACKBONE = LEARNING_RATE
LEARNING_RATE_HEAD = LEARNING_RATE
FEATURE = f"{'grad_accumulation' if EPSILON == INNER_ITERATION else ''}"
RUN_NAME = f'{DEVICE_ID}-{IMAGE_SIZE}_{PATCH_SIZE}-{PERCENT_SAMPLING}-bs-{BATCH_SIZE}-resnet50+head-{MEMORY}-{FEATURE}-datetime_{date_time}'


LATENT_DIMENSION = 256
NUM_CLASSES = 6
SEED = 42
STRIDE = PATCH_SIZE
NUM_PATCHES = ((IMAGE_SIZE-PATCH_SIZE)//STRIDE) + 1
NUM_WORKERS = 4
TRAIN_ROOT_DIR = f''
VAL_ROOT_DIR = TRAIN_ROOT_DIR
TRAIN_CSV_PATH = f'./train_kfold.csv'
VAL_CSV_PATH = f'./train_kfold.csv'
MEAN = mean[DATASET_NAME]
STD = std[DATASET_NAME]
SANITY_DATA_LEN = None
MODEL_SAVE_DIR = f"../{'models_icml' if MAIN_RUN else 'models'}/{'sanity' if SANITY_CHECK else 'runs'}/{RUN_NAME}"
DECAY_FACTOR = 2
VALIDATION_EVERY = 1
BASELINE = False
CONINUE_FROM_LAST = False 
MODEL_LOAD_DIR = '' 
GROUP = "512_32_runs"
