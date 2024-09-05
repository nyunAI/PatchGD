from sklearn import metrics
from data_utils import *
from train_utils import *
from utils import *
from models import Backbone
import torch.optim as optim
import transformers
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
import torch.nn as nn


class Trainer():
    def __init__(self,
                seed,
                run_number,
                train_dataset,
                val_dataset,
                batch_size,
                num_workers,
                num_classes,
                accelarator,
                run_name,
                learning_rate,
                epochs,
                warmup_epochs,
                decay_factor,
                monitor_wandb,
                save_models,
                model_save_dir):

        seed_everything(seed)
        self.epochs = epochs
        self.num_classes = num_classes
        self.epoch = 0
        self.monitor_wandb = monitor_wandb
        self.save_models = save_models
        self.model_save_dir = model_save_dir
        self.run_number = run_number
        self.accelarator = accelarator

        self.train_dataset = train_dataset 
        self.val_dataset = val_dataset 
        self.train_loader = DataLoader(self.train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
        self.validation_loader = DataLoader(self.val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
        
        print(f"Length of train loader: {len(self.train_loader)},Validation loader: {(len(self.validation_loader))}")
    
        self.model1 = Backbone(self.num_classes)
        self.model1.to(accelarator)
        for param in self.model1.parameters():
            param.requires_grad = True
    
        print(f"Baseline model:")
        
        self.criterion = nn.CrossEntropyLoss()
        self.lrs = {
            'head': learning_rate,
            'backbone': learning_rate
        }
        parameters = [{'params': self.model1.parameters(),
                        'lr': self.lrs['backbone']},
                        ]
        self.optimizer = optim.Adam(parameters)
        steps_per_epoch = len(self.train_dataset)//(batch_size)

        if len(self.train_dataset)%batch_size!=0:
            steps_per_epoch+=1
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,warmup_epochs*steps_per_epoch,decay_factor*self.epochs*steps_per_epoch)
        
    
    def get_metrics(self,predictions,actual,isTensor=False):
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

    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']


    def train_step(self):
        self.model1.train()
        print("Train Loop!")
        running_loss_train = 0.0
        train_correct = 0
        num_train = 0
        train_predictions = np.array([])
        train_labels = np.array([])
        for images,labels in tqdm(self.train_loader):
            images = images.to(self.accelarator)
            labels = labels.to(self.accelarator)
            batch_size = labels.shape[0]
            num_train += labels.shape[0]
            self.optimizer.zero_grad()
            outputs = self.model1(images)
            # if torch.isnan(outputs).any():
            #     print("output has nan")
            _,preds = torch.max(outputs,1)
            train_correct += (preds == labels).sum().item()
            correct = (preds == labels).sum().item()

            train_metrics_step = self.get_metrics(preds,labels,True)
            train_predictions = np.concatenate((train_predictions,preds.detach().cpu().numpy()))
            train_labels = np.concatenate((train_labels,labels.detach().cpu().numpy()))

            loss = self.criterion(outputs,labels)
            l = loss.item()
            running_loss_train += loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.lr = self.get_lr(self.optimizer)
            if self.monitor_wandb:
                wandb.log({'lr':self.lr,"train_loss_step":l/batch_size,'epoch':self.epoch,'train_accuracy_step_metric':train_metrics_step['accuracy'],'train_kappa_step_metric':train_metrics_step['kappa']})
        train_metrics = self.get_metrics(train_predictions,train_labels)
        print(f"Train Loss: {running_loss_train/num_train} Train Accuracy: {train_correct/num_train}")
        print(f"Train Accuracy Metric: {train_metrics['accuracy']} Train Kappa Metric: {train_metrics['kappa']}")    
        return {
                'loss': running_loss_train/num_train,
                'accuracy': train_metrics['accuracy'],
                'kappa': train_metrics['kappa']
            }


    def val_step(self):
        val_predictions = np.array([])
        val_labels = np.array([])
        running_loss_val = 0.0
        val_correct  = 0
        num_val = 0
        self.model1.eval()
        with torch.no_grad():
            print("Validation Loop!")
            for images,labels in tqdm(self.validation_loader):
                images = images.to(self.accelarator)
                labels = labels.to(self.accelarator)
                batch_size = labels.shape[0]

                outputs = self.model1(images)
                # if torch.isnan(outputs).any():
                #     print("L1 has nan")
                num_val += labels.shape[0]
                _,preds = torch.max(outputs,1)
                val_correct += (preds == labels).sum().item()
                correct = (preds == labels).sum().item()

                val_metrics_step = self.get_metrics(preds,labels,True)
                val_predictions = np.concatenate((val_predictions,preds.detach().cpu().numpy()))
                val_labels = np.concatenate((val_labels,labels.detach().cpu().numpy()))


                loss = self.criterion(outputs,labels)
                l = loss.item()
                running_loss_val += loss.item()
                if self.monitor_wandb:
                    wandb.log({'lr':self.lr,"val_loss_step":l/batch_size,"epoch":self.epoch,'val_accuracy_step_metric':val_metrics_step['accuracy'],'val_kappa_step_metric':val_metrics_step['kappa']})
            
            val_metrics = self.get_metrics(val_predictions,val_labels)
            print(f"Validation Loss: {running_loss_val/num_val} Validation Accuracy: {val_correct/num_val}")
            print(f"Val Accuracy Metric: {val_metrics['accuracy']} Val Kappa Metric: {val_metrics['kappa']}")    
            return {
                'loss': running_loss_val/num_val,
                'accuracy': val_metrics['accuracy'],
                'kappa': val_metrics['kappa']
            }

    def run(self):
        best_validation_loss = float('inf')
        best_validation_accuracy = 0
        best_validation_metric = -float('inf')

        for epoch in range(self.epochs):
            print("="*31)
            print(f"{'-'*10} Epoch {epoch+1}/{self.epochs} {'-'*10}")
            train_logs = self.train_step()
            val_logs = self.val_step()
            self.epoch = epoch
            if val_logs["loss"] < best_validation_loss:
                best_validation_loss = val_logs["loss"]
                if self.save_models:
                    torch.save({
                    'model1_weights': self.model1.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                    'epoch' : epoch+1,
                    }, f"{self.model_save_dir}/best_val_loss.pt")

            if val_logs['accuracy'] > best_validation_accuracy:
                best_validation_accuracy = val_logs['accuracy']
                if self.save_models:
                    torch.save({
                    'model1_weights': self.model1.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                    'epoch' : epoch+1,
                    }, f"{self.model_save_dir}/best_val_accuracy.pt")

            if val_logs['kappa'] > best_validation_metric:
                best_validation_metric = val_logs['kappa']
                if self.save_models:
                    torch.save({
                    'model1_weights': self.model1.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                    'epoch' : epoch+1,
                    }, f"{self.model_save_dir}/best_val_metric.pt")

            if self.monitor_wandb:
                wandb.log({"training_loss": train_logs['loss'],  
                "validation_loss": val_logs['loss'], 
                'training_accuracy_metric': train_logs['accuracy'],
                'training_kappa_metric': train_logs['kappa'],
                'validation_accuracy_metric': val_logs['accuracy'],
                'validation_kappa_metrics': val_logs['kappa'],
                'epoch':self.epoch,
                'best_loss':best_validation_loss,
                'best_accuracy':best_validation_accuracy,
                'best_metric': best_validation_metric})
                    
            if self.save_models:
                torch.save({
                'model1_weights': self.model1.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'epoch' : epoch+1,
                }, f"{self.model_save_dir}/last_epoch.pt")
        
        return {
            'best_accuracy':best_validation_accuracy,
            'best_loss': best_validation_loss,
            'best_metric': best_validation_metric
        }