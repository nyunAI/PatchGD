from random import seed, shuffle
from torchvision.datasets.folder import ImageFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import warnings
from sklearn import metrics
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch
import cv2
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import wandb
import pathlib
import os
import time
from PIL import Image
from torchvision.models import resnet50
import matplotlib.pyplot as plt
from datetime import datetime
import transformers
warnings.filterwarnings('ignore')
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import yaml
import random
import argparse
import timm
from timm.data.config import resolve_data_config
from timm.data import Mixup
from timm.optim import create_optimizer_v2,optimizer_kwargs
from timm.data.transforms_factory import create_transform
from timm.loss import BinaryCrossEntropy
from timm.scheduler import create_scheduler
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='./config.yaml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    # print(cfg)
    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class IMAGENET100(Dataset):    
    def __init__(self,df,transform=None):
        self.df = df
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self,index):
        path = self.df.iloc[index].file_paths
        label = self.df.iloc[index].class_num
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label)

def get_train_val_dataset(print_lengths=True):
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_dataset = IMAGENET100(train_df)
    val_df = pd.read_csv(VAL_CSV_PATH)
    print(train_df.head())
    print(val_df.head())
    validation_dataset = IMAGENET100(val_df)
    return train_dataset, validation_dataset

def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        no_aug=False,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_split=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        num_aug_repeats=0,
        num_aug_splits=0,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        crop_pct=None,
        collate_fn=None,
        pin_memory=False,
        fp16=False,
        tf_preprocessing=False,
        use_multi_epochs_loader=False,
        persistent_workers=True,
        worker_seeding='all',
):
    re_num_splits = 0
    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    dataset.transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        no_aug=no_aug,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )
    print(dataset.transform)
    
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=is_training,num_workers=num_workers)

    return loader
def get_metrics(predictions,actual,isTensor=False):
    if isTensor:
        p = predictions.detach().cpu().numpy()
        a = actual.detach().cpu().numpy()
    else:
        p = predictions
        a = actual
    kappa_score = metrics.cohen_kappa_score(a, p, labels=None, weights= 'quadratic', sample_weight=None)
    accuracy = metrics.accuracy_score(y_pred=p,y_true=a)
    return {
        "kappa":  kappa_score,
        "accuracy": accuracy
    }

def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim))).data.shape

class Backbone(nn.Module):
    def __init__(self,args):
        super(Backbone,self).__init__()
        self.encoder = timm.create_model(
            args.model,
            pretrained=args.pretrained,
            in_chans=3,
            num_classes=NUM_CLASSES,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript,
            checkpoint_path=args.initial_checkpoint,
        )
    def forward(self,x):
        return self.encoder(x)


if __name__ == "__main__":

    DEVICE_ID = 2 ######################################
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    now = datetime.now() 
    date_time = now.strftime("%d_%m_%Y__%H_%M")
    MAIN_RUN = True ######################################
    CONINUE_FROM_LAST = False ######################################
    args, args_text = _parse_args()
    # print(args.model)
    args.prefetcher = False
    

    MONITOR_WANDB = False ######################################
    SAVE_MODELS =  MONITOR_WANDB
    BASELINE = True
    SANITY_CHECK = False
    EPOCHS = 100
    ACCELARATOR = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    IMAGE_SIZE = args.img_size
    BATCH_SIZE = args.batch_size ######################################
    FEATURE = '' ######################################
    MEMORY = 48 ######################################
    MODEL_LOAD_DIR = '' ######################################
    RUN_NAME = f'full_100_timm_strategy_scheduler_included_mixed_precision_{DEVICE_ID}-{IMAGE_SIZE}-{BATCH_SIZE}-resnet50-baseline-{MEMORY}GB-{FEATURE}-datetime_{date_time}' ######################################

    NUM_CLASSES = 100
    SEED = 42
    WARMUP_EPOCHS = args.warmup_epochs
    NUM_WORKERS = 4
    TRAIN_CSV_PATH = './train_100_full.csv'
    VAL_CSV_PATH = './val_100_10_1.csv'
    MEAN = IMAGENET_DEFAULT_MEAN
    STD = IMAGENET_DEFAULT_STD
    SANITY_DATA_LEN = None
    MODEL_SAVE_DIR = f"./{'models_icml' if MAIN_RUN else 'models'}/{'sanity' if SANITY_CHECK else 'runs'}/{RUN_NAME}"
    EXPERIMENT = "ImageNet100-10" if not SANITY_CHECK else 'imagenet-sanity-gowreesh' 
    DECAY_FACTOR = 1

    args.num_classes=NUM_CLASSES

    if SAVE_MODELS:
        os.makedirs(MODEL_SAVE_DIR,exist_ok=True)
   
    if MONITOR_WANDB:
        run = wandb.init(project=EXPERIMENT, entity="gowreesh", reinit=True)
        wandb.run.name = RUN_NAME
        wandb.run.save()
    
 
    seed_everything(SEED)
 
    
    model1 = Backbone(args)
    model1.to(ACCELARATOR)
    for param in model1.parameters():
        param.requires_grad = True
    data_config = resolve_data_config(vars(args), model=model1.encoder, verbose=True)
    print(ACCELARATOR)
    print(RUN_NAME)
 
 
    print(f"Baseline model:")
    
    train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh).to()
    train_loss_fn = train_loss_fn.to(device=ACCELARATOR)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=ACCELARATOR)
    
   
    # optimizer = optim.Adam(parameters)
    optimizer = create_optimizer_v2(
        model1,
        **optimizer_kwargs(cfg=args),
    )

    train_dataset, val_dataset = get_train_val_dataset()
    train_loader = create_loader(
        train_dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=0,
        interpolation=args.train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=None,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        
    )

    eval_workers = args.workers
    if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
        # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
        eval_workers = min(2, args.workers)
    validation_loader = create_loader(
        val_dataset,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=eval_workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )
    
    # for i in train_loader:
    #     print(i[0].shape,i[1].shape)
    #     break
        
    # for i in validation_loader:
    #     print(i[0].shape,i[1].shape)
    #     break
    print(f"Length of train loader: {len(train_loader)},Validation loader: {(len(validation_loader))}")

    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )
    mixup_fn = Mixup(**mixup_args)
    steps_per_epoch = len(train_dataset)//(BATCH_SIZE)

    if len(train_dataset)%BATCH_SIZE!=0:
        steps_per_epoch+=1
    # scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,WARMUP_EPOCHS*steps_per_epoch,DECAY_FACTOR*EPOCHS*steps_per_epoch)
    scheduler, num_epochs = create_scheduler(
        args=args,
        optimizer=optimizer,
    )
    if CONINUE_FROM_LAST:
        checkpoint = torch.load(f"{MODEL_LOAD_DIR}/best_val_accuracy.pt")
        start_epoch = checkpoint['epoch']
        print(f"Model already trained for {start_epoch} epochs on 512 size images.")
        model1.load_state_dict(checkpoint['model1_weights'])
    
    best_validation_loss = float('inf')
    best_validation_accuracy = 0
    best_validation_metric = -float('inf')
    
    use_amp=True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(EPOCHS):
        print("="*31)
        print(f"{'-'*10} Epoch {epoch+1}/{EPOCHS} {'-'*10}")

        running_loss_train = 0.0
        running_loss_val = 0.0
        train_correct = 0
        val_correct  = 0
        num_train = 0
        num_val = 0
       
        train_predictions = np.array([])
        train_labels = np.array([])

        val_predictions = np.array([])
        val_labels = np.array([])
 
        model1.train()
        print("Train Loop!")
        for images,labels in tqdm(train_loader):
            images = images.to(ACCELARATOR)
            labels = labels.to(ACCELARATOR)
            images, labels = mixup_fn(images,labels)

            batch_size = labels.shape[0]
            num_train += labels.shape[0]
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                outputs = model1(images)
            # if torch.isnan(outputs).any():
            #     print("output has nan")
            # print(outputs.shape,labels.shape)
            # _,preds = torch.max(outputs,1)
            # import pdb; pdb.set_trace()
            # train_correct += (preds == labels).sum().item()
            # correct = (preds == labels).sum().item()

            # train_metrics_step = get_metrics(preds,labels,True)
            # train_predictions = np.concatenate((train_predictions,preds.detach().cpu().numpy()))
            # train_labels = np.concatenate((train_labels,labels.detach().cpu().numpy()))

                loss = train_loss_fn(outputs,labels)
                l = loss.item()
                running_loss_train += loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            lr = get_lr(optimizer)
            if MONITOR_WANDB:
                wandb.log({'lr':lr,"train_loss_step":l/batch_size,'epoch':epoch,})
                    #    'train_accuracy_step_metric':train_metrics_step['accuracy'],'train_kappa_step_metric':train_metrics_step['kappa']})
        # train_metrics = get_metrics(train_predictions,train_labels)
        print(f"Train Loss: {running_loss_train/num_train}")
        # print(f"Train Accuracy Metric: {train_metrics['accuracy']} Train Kappa Metric: {train_metrics['kappa']}")    
        scheduler.step(epoch+1)
 
        # Evaluation Loop!
        val_accr = 0.0
        val_lossr = 0.0
        if (epoch+1)%1 == 0:
 
            model1.eval()
       
            with torch.no_grad():
                print("Validation Loop!")
                for images,labels in tqdm(validation_loader):
                    images = images.to(ACCELARATOR)
                    labels = labels.to(ACCELARATOR)
                    batch_size = labels.shape[0]

                    outputs = model1(images)
                    # if torch.isnan(outputs).any():
                    #     print("L1 has nan")
                    num_val += labels.shape[0]
                    _,preds = torch.max(outputs,1)
                    val_correct += (preds == labels).sum().item()
                    correct = (preds == labels).sum().item()

                    val_metrics_step = get_metrics(preds,labels,True)
                    val_predictions = np.concatenate((val_predictions,preds.detach().cpu().numpy()))
                    val_labels = np.concatenate((val_labels,labels.detach().cpu().numpy()))


                    loss = validate_loss_fn(outputs,labels)
                    l = loss.item()
                    running_loss_val += loss.item()
                    if MONITOR_WANDB:
                        wandb.log({'lr':lr,"val_loss_step":l/batch_size,"epoch":epoch,'val_accuracy_step_metric':val_metrics_step['accuracy'],'val_kappa_step_metric':val_metrics_step['kappa']})
                
                val_metrics = get_metrics(val_predictions,val_labels)
                print(f"Validation Loss: {running_loss_val/num_val} Validation Accuracy: {val_correct/num_val}")
                print(f"Val Accuracy Metric: {val_metrics['accuracy']} Val Kappa Metric: {val_metrics['kappa']}")    

                if (running_loss_val/num_val) < best_validation_loss:
                    best_validation_loss = running_loss_val/num_val
                    if SAVE_MODELS:
                        torch.save({
                        'model1_weights': model1.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'epoch' : epoch+1,
                        }, f"{MODEL_SAVE_DIR}/best_val_loss.pt")
            
                if val_metrics['accuracy'] > best_validation_accuracy:
                    best_validation_accuracy = val_metrics['accuracy']
                    if SAVE_MODELS:
                        torch.save({
                        'model1_weights': model1.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'epoch' : epoch+1,
                        }, f"{MODEL_SAVE_DIR}/best_val_accuracy.pt")
                
                if val_metrics['kappa'] > best_validation_metric:
                    best_validation_metric = val_metrics['kappa']
                    if SAVE_MODELS:
                        torch.save({
                        'model1_weights': model1.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'epoch' : epoch+1,
                        }, f"{MODEL_SAVE_DIR}/best_val_metric.pt")
                
        if MONITOR_WANDB:
             wandb.log({"training_loss": running_loss_train/num_train,  
             "validation_loss": running_loss_val/num_val, 
            #  'training_accuracy_metric': train_metrics['accuracy'],
            #  'training_kappa_metric': train_metrics['kappa'],
             'validation_accuracy_metric': val_metrics['accuracy'],
             'validation_kappa_metrics': val_metrics['kappa'],
             'epoch':epoch,
             'best_loss':best_validation_loss,
             'best_accuracy':best_validation_accuracy,
             'best_metric': best_validation_metric})
        
        if SAVE_MODELS:
            torch.save({
                    'model1_weights': model1.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'epoch' : epoch+1,
                    }, f"{MODEL_SAVE_DIR}/last_epoch.pt")
