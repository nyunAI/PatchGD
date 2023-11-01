import torch
from datetime import datetime

BASELINE = False
SANITY_CHECK = False
SCALE_FACTOR = 1
IMAGE_SIZE = int(SCALE_FACTOR * 512)
EPOCHS = 50 
LATENT_DIMENSION = 256
BATCH_SIZE = 128
NUM_CLASSES = 28
SEED = 42
PATCH_SIZE = int(SCALE_FACTOR * 128)
STRIDE = int(SCALE_FACTOR * 64)
LEARNING_RATE_BACKBONE = 1e-3
LEARNING_RATE_HEAD = 1e-3
WARMUP_EPOCHS = 2
NUM_PATCHES = ((IMAGE_SIZE-PATCH_SIZE)//STRIDE) + 1
NUM_WORKERS = 4
ACCELARATOR = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_ROOT_DIR = f'../data/ultramnist_{IMAGE_SIZE}/train'
VAL_ROOT_DIR = f'../data/ultramnist_{IMAGE_SIZE}/val'
MNIST_ROOT_DIR = '../data/mnist'
TRAIN_CSV_PATH = f'../data/ultramnist_{IMAGE_SIZE}/train.csv'
VAL_CSV_PATH = f'../data/ultramnist_{IMAGE_SIZE}/valid.csv'
PERCENT_SAMPLING = 0.3
MEAN = [0.1307,0.1307,0.1307]
STD = [0.3081,0.3081,0.3081]
now = datetime.now() # current date and time
date_time = now.strftime("%d_%m_%Y__%H_%M")
RUN_NAME = f'resnet10t+L1-{STRIDE}-{PERCENT_SAMPLING}'
SANITY_DATA_LEN = 512
EXPERIMENT = "ultracnn-shared-runs-gowreesh" if not SANITY_CHECK else 'ultracnn-sanity-gowreesh'
MODEL_SAVE_DIR = f'../models/4_5/{date_time}_{RUN_NAME}_{IMAGE_SIZE}' if not SANITY_CHECK else f'../models/4_5/sanity/{date_time}_{RUN_NAME}_{IMAGE_SIZE}'
SPLITS = 10


DEVICES = [4,5]
PRECISION = 16
DEVICE_TUNE =[4]
FIND_BATCH_SIZE = False
