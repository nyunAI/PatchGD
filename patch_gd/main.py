import argparse
from config import *
from data_utils import *
from train_utils import *
from utils import *
from models import *
import numpy as np
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for PatchGD")
    parser.add_argument('--head',default='original',type=str)
    parser.add_argument('--grad_accumulation',default=GRAD_ACCUM,type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--latent_size',default=LATENT_DIMENSION,type=int)
    parser.add_argument('--sampling',default=PERCENT_SAMPLING,type=float)
    parser.add_argument('--batch_size',default=BATCH_SIZE,type=int)
    parser.add_argument('--monitor_wandb',default=MONITOR_WANDB,type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--save_models',default=SAVE_MODELS,type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--gpu_id',default=0,type=int)
    args = parser.parse_args()
    
    HEAD = args.head
    GRAD_ACCUM = args.grad_accumulation
    LATENT_DIMENSION = args.latent_size
    MONITOR_WANDB = args.monitor_wandb
    BATCH_SIZE = args.batch_size 
    PERCENT_SAMPLING = args.sampling
    SAVE_MODELS = args.save_models
    DEVICE_ID = args.gpu_id
    FEATURE = f"{'grad_accumulation' if GRAD_ACCUM else ''}_head={HEAD}_latent_dim={LATENT_DIMENSION}" ######################################
    
    RUN_NAME = f'{DEVICE_ID}-{IMAGE_SIZE}_{PATCH_SIZE}-{PERCENT_SAMPLING}-bs-{BATCH_SIZE}-resnet50+head-{MEMORY}-{FEATURE}-datetime_{date_time}' 

    
    train_dataset,val_dataset = get_train_val_dataset(TRAIN_CSV_PATH,
                                                    SANITY_CHECK,
                                                    SANITY_DATA_LEN,
                                                    TRAIN_ROOT_DIR,
                                                    VAL_ROOT_DIR,
                                                    IMAGE_SIZE,
                                                    MEAN,
                                                    STD)


    print(DEVICE_ID)
    print(RUN_NAME)

    trainer = Trainer(SEED,
                EXPERIMENT,
                train_dataset,
                val_dataset,
                BATCH_SIZE,
                NUM_WORKERS,
                NUM_CLASSES,
                ACCELARATOR,
                RUN_NAME,
                HEAD,
                LATENT_DIMENSION,
                NUM_PATCHES,
                PATCH_SIZE,
                STRIDE,
                PERCENT_SAMPLING,
                LEARNING_RATE,
                EPOCHS,
                INNER_ITERATION,
                GRAD_ACCUM,
                WARMUP_EPOCHS,
                DECAY_FACTOR,
                MONITOR_WANDB,
                SAVE_MODELS,
                MODEL_SAVE_DIR)
    logs = trainer.run()
    del trainer
