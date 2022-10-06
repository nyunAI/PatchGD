import torch

SANITY_CHECK = False
SCALE_FACTOR = 1
IMAGE_SIZE = int(SCALE_FACTOR * 512)
EPOCHS = 100
LATENT_DIMENSION = 256
BATCH_SIZE = 128
NUM_CLASSES = 28
SEED = 42
PATCH_SIZE = int(SCALE_FACTOR * 128)
STRIDE = int(SCALE_FACTOR * 32)
LEARNING_RATE_BACKBONE = 0.001
LEARNING_RATE_HEAD = 0.005
WARMUP_EPOCHS = 2
NUM_PATCHES = ((IMAGE_SIZE-PATCH_SIZE)//STRIDE) + 1
NUM_WORKERS = 8
ACCELARATOR = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_ROOT_DIR = f'../data/ultramnist_{IMAGE_SIZE}/train'
VAL_ROOT_DIR = f'../data/ultramnist_{IMAGE_SIZE}/val'
MNIST_ROOT_DIR = '../data/mnist'
TRAIN_CSV_PATH = f'../data/ultramnist_{IMAGE_SIZE}/train.csv'
VAL_CSV_PATH = f'../data/ultramnist_{IMAGE_SIZE}/valid.csv'
PERCENT_SAMPLING = 0.3
MEAN = [0.1307,0.1307,0.1307]
STD = [0.3081,0.3081,0.3081]
RUN_NAME = 'resnet18+L1'
SANITY_DATA_LEN = 300
EXPERIMENT = "ultracnn-shared-runs-gowreesh" if not SANITY_CHECK else 'ultracnn-sanity-gowreesh'
MODEL_SAVE_DIR = f'../models/6_7/{RUN_NAME}_{IMAGE_SIZE}'
SPLITS = 10


DEVICES = [6,7]
PRECISION = 16
DEVICE_TUNE =[6]
FIND_BATCH_SIZE = False
