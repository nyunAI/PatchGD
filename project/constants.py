import torch
import datetime

SANITY_CHECK = False
SCALE_FACTOR = 1
IMAGE_SIZE = int(SCALE_FACTOR * 512)
EPOCHS = 50
LATENT_DIMENSION = 256
BATCH_SIZE = 16
NUM_CLASSES = 28
SEED = 42
PATCH_SIZE = int(SCALE_FACTOR * 128)
STRIDE = int(SCALE_FACTOR * 32)
LEARNING_RATE_BACKBONE = 0.0005
LEARNING_RATE_HEAD = 0.001
WARMUP_STEPS = 2
NUM_PATCHES = ((IMAGE_SIZE-PATCH_SIZE)//STRIDE) + 1
NUM_WORKERS = 4
ACCELARATOR = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT_DIR = f'../data/ultra-mnist_{IMAGE_SIZE}/train'
MNIST_ROOT_DIR = '../data/mnist'
TRAIN_CSV_PATH = f'../data/ultra-mnist_{IMAGE_SIZE}/train.csv'
PERCENT_SAMPLING = 0.3
MEAN = [0.1307,0.1307,0.1307]
STD = [0.3081,0.3081,0.3081]
today = datetime.datetime.now()
EXPERIMENT = "ultracnn-shared-runs-gowreesh" if not SANITY_CHECK else 'ultracnn-sanity-gowreesh'
MODEL_SAVE_DIR = f'./models_{IMAGE_SIZE}'
SPLITS = 10


DEVICES = [6,7]
PRECISION = 16
DEVICE_TUNE =[7]
FIND_BATCH_SIZE = False
