import torch
import datetime

SANITY_CHECK = False
SCALE_FACTOR = 1
IMAGE_SIZE = int(SCALE_FACTOR * 512)
EPOCHS = 5
LATENT_DIMENSION = 256
BATCH_SIZE = 16
NUM_CLASSES = 28
SEED = 42
PATCH_SIZE = int(SCALE_FACTOR * 128)
STRIDE = int(SCALE_FACTOR * 32)
LEARNING_RATE_BACKBONE = 0.0005
LEARNING_RATE_HEAD = 0.001
GAMMA = 0.9
NUM_PATCHES = ((IMAGE_SIZE-PATCH_SIZE)//STRIDE) + 1
NUM_WORKERS = 4
ACCELARATOR = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT_DIR = '../data/ultramnist_sample/train'
MNIST_ROOT_DIR = '../data/mnist'
TRAIN_CSV_PATH = '../data/ultramnist_sample/train.csv'
PERCENT_SAMPLING = 0.3
MEAN = [0.1307,0.1307,0.1307]
STD = [0.3081,0.3081,0.3081]
today = datetime.datetime.now()
EXPERIMENT = "ultracnn-shared-runs-gowreesh"
MODEL_SAVE_DIR = './models'
SPLITS = 10


DEVICES = [4,5]
DEVICE_TUNE =[7]
FIND_BATCH_SIZE = False
