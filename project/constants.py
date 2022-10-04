import torch
import datetime

SANITY_CHECK = True
SCALE_FACTOR = 1
IMAGE_SIZE = int(SCALE_FACTOR * 512)
EPOCHS = 5
LATENT_DIMENSION = 256
BATCH_SIZE = 4
NUM_CLASSES = 28
SEED = 42
PATCH_SIZE = int(SCALE_FACTOR * 128)
STRIDE = int(SCALE_FACTOR * 32)
LEARNING_RATE_BACKBONE = 0.00005
LEARNING_RATE_HEAD = 0.0005
GAMMA = 0.9
NUM_PATCHES = ((IMAGE_SIZE-PATCH_SIZE)//STRIDE) + 1
NUM_WORKERS = 2
ACCELARATOR = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT_DIR = '../data/ultramnist_sample/train'
MNIST_ROOT_DIR = '../data/mnist'
TRAIN_CSV_PATH = '../data/ultramnist_sample/train.csv'
PERCENT_SAMPLING = 0.3
MEAN = [0.1307,0.1307,0.1307]
STD = [0.3081,0.3081,0.3081]
today = datetime.datetime.now()
EXPERIMENT = today.strftime("%d_%m_%Y__%H_%M_")+f"effnetb0_{PERCENT_SAMPLING}_{EPOCHS}"
MODEL_SAVE_DIR = './models'
SPLITS = 10


DEVICES = -1
