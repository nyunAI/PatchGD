import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
from PIL import Image
import os
from torchvision import transforms
from torchvision.models import resnet50
import random
from sklearn import metrics
import torch.optim as optim
from tqdm import tqdm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from glob import glob

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transform(image_size,mean,std):
    return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)])

class IMAGENET100(Dataset):    
    def __init__(self,df,img_dir,transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self,index):
        image_id = self.df.iloc[index]['image_id']
        label = int(self.df.iloc[index]['class'])
        path = os.path.join(self.img_dir,f'{image_id}.jpeg')
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label)
    
class IMAGENET100_test(Dataset):    
    def __init__(self,img_paths,transform=None):
        self.img_paths = img_paths
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self,index):
        image_id = self.img_paths[index].split('/')[-1].split('.')[0]
        path = self.img_paths[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image,image_id

class BaselineModel(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.encoder = resnet50()
        self.encoder.fc = nn.Linear(2048,num_classes)
    def forward(self,x):
        return self.encoder(x)


class Trainer():
    def __init__(self,
                train_dataset,
                val_dataset,
                test_dataset,
                batch_size,
                num_workers,
                device,
                learning_rate,
                epochs,
                model,
                test_output_dir):
        
        self.epochs = epochs
        self.epoch = 0
        self.accelarator = device
        self.model = model
        self.test_output_dir = test_output_dir

        self.train_dataset = train_dataset 
        self.val_dataset = val_dataset 
        self.test_dataset = test_dataset

        self.train_loader = DataLoader(self.train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
        self.validation_loader = DataLoader(self.val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

        print(f"Length of train loader: {len(self.train_loader)}, validation loader: {(len(self.validation_loader))}, test loader: {len(self.test_loader)}")
    
        self.criterion = nn.CrossEntropyLoss()
        self.lrs = {
            'backbone': learning_rate
        }
        parameters = [{'params': self.model.parameters(),
                        'lr': self.lrs['backbone']},
                        ]
        self.optimizer = optim.Adam(parameters)
    
    def get_metrics(self,predictions,actual,isTensor=False):
        if isTensor:
            p = predictions.detach().cpu().numpy()
            a = actual.detach().cpu().numpy()
        else:
            p = predictions
            a = actual
        accuracy = metrics.accuracy_score(y_pred=p,y_true=a)
        return {
            "accuracy": accuracy
        }

    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train_step(self):
        self.model.train()
        print("Train Loop!")
        running_loss_train = 0.0
        num_train = 0
        train_predictions = np.array([])
        train_labels = np.array([])
        for images,labels in tqdm(self.train_loader):
            images = images.to(self.accelarator)
            labels = labels.to(self.accelarator)
            num_train += labels.shape[0]
            self.optimizer.zero_grad()
            outputs = self.model(images)
            _,preds = torch.max(outputs,1)
            train_predictions = np.concatenate((train_predictions,preds.detach().cpu().numpy()))
            train_labels = np.concatenate((train_labels,labels.detach().cpu().numpy()))

            loss = self.criterion(outputs,labels)
            running_loss_train += loss.item()
            loss.backward()
            self.optimizer.step()
            self.lr = self.get_lr(self.optimizer)

        train_metrics = self.get_metrics(train_predictions,train_labels)
        print(f"Train Loss: {running_loss_train/num_train}")
        print(f"Train Accuracy Metric: {train_metrics['accuracy']}")    
        return {
                'loss': running_loss_train/num_train,
                'accuracy': train_metrics['accuracy'],
            }


    def val_step(self):
        val_predictions = np.array([])
        val_labels = np.array([])
        running_loss_val = 0.0
        num_val = 0
        self.model.eval()
        with torch.no_grad():
            print("Validation Loop!")
            for images,labels in tqdm(self.validation_loader):
                images = images.to(self.accelarator)
                labels = labels.to(self.accelarator)
                outputs = self.model(images)
                num_val += labels.shape[0]
                _,preds = torch.max(outputs,1)
                val_predictions = np.concatenate((val_predictions,preds.detach().cpu().numpy()))
                val_labels = np.concatenate((val_labels,labels.detach().cpu().numpy()))


                loss = self.criterion(outputs,labels)
                running_loss_val += loss.item()
            val_metrics = self.get_metrics(val_predictions,val_labels)
            print(f"Validation Loss: {running_loss_val/num_val}")
            print(f"Val Accuracy Metric: {val_metrics['accuracy']} ")    
            return {
                'loss': running_loss_val/num_val,
                'accuracy': val_metrics['accuracy'],
            }
    def test_step(self):
        test_image_ids = np.array([])
        test_predictions = np.array([])
        self.model.eval()
        with torch.no_grad():
            print("Test Loop!")
            for images,image_ids in tqdm(self.test_loader):
                images = images.to(self.accelarator)
                outputs = self.model(images)
                _,preds = torch.max(outputs,1)
                test_image_ids = np.concatenate((test_image_ids,image_ids))
                test_predictions = np.concatenate((test_predictions,preds.detach().cpu().numpy()))
            
#             with open(os.path.join(self.test_output_dir , 'submission.txt'), 'w') as f:
#                 for x, y in zip(test_image_ids[:-1], test_predictions[:-1]):
#                     f.write(f'{x},{int(y)}\n')
#                 f.write(f'{test_image_ids[-1]},{int(test_predictions[-1])}')
            
#             f.close()
            pd.DataFrame({
                'image_id':test_image_ids,
                'label':test_predictions.astype(int)
            }).to_csv(os.path.join(self.test_output_dir , 'submission.csv'), index=False)

    def run(self,run_test=True):
        best_validation_loss = float('inf')
        best_validation_accuracy = 0

        for epoch in range(self.epochs):
            print("="*31)
            print(f"{'-'*10} Epoch {epoch+1}/{self.epochs} {'-'*10}")
            train_logs = self.train_step()
            val_logs = self.val_step()
            self.epoch = epoch
            if val_logs["loss"] < best_validation_loss:
                best_validation_loss = val_logs["loss"]
            if val_logs['accuracy'] > best_validation_accuracy:
                best_validation_accuracy = val_logs['accuracy']
        if run_test:
            self.test_step()

        return {
            'best_accuracy':best_validation_accuracy,
            'best_loss': best_validation_loss,
        }    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for training baseline on ImageNet100")
    parser.add_argument('--root_dir',default='./',type=str)
    parser.add_argument('--epochs',default=2,type=int)
    parser.add_argument('--batch_size',default=32,type=int)
    parser.add_argument('--image_size',default=160,type=int)
    parser.add_argument('--seed',default=42,type=int)
    parser.add_argument('--num_classes',default=100,type=int)
    parser.add_argument('--num_workers',default=2,type=int)
    parser.add_argument('--lr',default=1e-3,type=float)
    parser.add_argument('--output_dir',default='./',type=str)
    args = parser.parse_args()
    
    ROOT_DIR = args.root_dir
    IMAGE_SIZE = args.image_size
    EPOCHS = args.epochs
    SEED = args.seed
    BATCH_SIZE = args.batch_size 
    NUM_CLASSES = args.num_classes
    NUM_WORKERS =args.num_workers
    LEARNING_RATE = args.lr
    TEST_OUTPUT_DIR = args.output_dir
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed_everything(SEED)
    model = BaselineModel(NUM_CLASSES)
    model.to(DEVICE)

    for param in model.parameters():
        param.requires_grad = True

    transform = get_transform(IMAGE_SIZE,
                            IMAGENET_DEFAULT_MEAN,
                            IMAGENET_DEFAULT_STD)

    train_df = pd.read_csv(os.path.join(ROOT_DIR,'train.csv'))
    val_df = pd.read_csv(os.path.join(ROOT_DIR,'val.csv'))
    test_files = glob(f'{ROOT_DIR}/test/*')

    train_dataset = IMAGENET100(train_df,os.path.join(ROOT_DIR,"train"),transform)
    val_dataset = IMAGENET100(val_df,os.path.join(ROOT_DIR,"val"),transform)
    test_dataset = IMAGENET100_test(test_files,transform)

    trainer = Trainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        model=model,
        test_output_dir=TEST_OUTPUT_DIR
    )
    
    trainer.run(run_test=True)
    

