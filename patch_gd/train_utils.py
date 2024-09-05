from sklearn import metrics
from data_utils import *
from train_utils import *
from utils import *
from models import *
import torch.optim as optim
import transformers
from tqdm import tqdm
import wandb
import torch.nn as nn
import os
from config import GROUP
import math
from torch.utils.data import DataLoader

class Trainer():
    def __init__(self,
                seed,
                experiment,
                train_dataset,
                val_dataset,
                batch_size,
                num_workers,
                num_classes,
                accelarator,
                run_name,
                head,
                latent_dimension,
                num_patches,
                patch_size,
                stride,
                percent_sampling,
                learning_rate,
                epochs,
                inner_iteration,
                grad_accumulation,
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
        self.accelarator = accelarator
        self.batch_size = batch_size 
        self.head = head
        self.inner_iteration = inner_iteration
        self.grad_accumulation = grad_accumulation
        if self.grad_accumulation:
            self.epsilon = self.inner_iteration
        else:
            self.epsilon = 1
        self.latent_dimension = latent_dimension
        self.num_patches = num_patches
        self.stride = stride
        self.patch_size = patch_size
        self.percent_sampling = percent_sampling
        self.train_dataset = train_dataset 
        self.val_dataset = val_dataset 
        self.train_loader = DataLoader(self.train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
        self.validation_loader = DataLoader(self.val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
        
        if self.monitor_wandb:
            run = wandb.init(project=experiment, entity="gowreesh", group=GROUP, reinit=True)
            wandb.run.name = run_name
            wandb.run.save()
            wandb.log({
                "batch_size": batch_size,
                'head': head,
                'latent_dimension': latent_dimension,
                'percent_sampling': percent_sampling,
                'grad_accumulation': grad_accumulation
            })

    
        if self.save_models:
            os.makedirs(self.model_save_dir,exist_ok=True)

        #print(f"Length of train loader: {len(self.train_loader)},Validation loader: {(len(self.validation_loader))}")
    
        self.model1 = Backbone(self.latent_dimension)
        self.model1.to(self.accelarator)
        for param in self.model1.parameters():
            param.requires_grad = True

        self.model2 = get_head_from_name(self.head,self.latent_dimension)
        self.model2.to(self.accelarator)
        for param in self.model2.parameters():
            param.requires_grad = True
    
        
        #print(f"Number of patches in one dimenstion: {self.num_patches}, percentage sampling is: {self.percent_sampling}")
        self.criterion = nn.CrossEntropyLoss()
        self.lrs = {
            'head': learning_rate,
            'backbone': learning_rate
        }
        parameters = [{'params': self.model1.parameters(),
                    'lr': self.lrs['backbone']},
                    {'params': self.model2.parameters(),
                    'lr': self.lrs['head']}]
        self.optimizer = optim.Adam(parameters)
        steps_per_epoch = len(self.train_dataset)//(self.batch_size)

        if len(self.train_dataset)%self.batch_size!=0:
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
        lrs = []
        for param_group in optimizer.param_groups:
            lrs.append(param_group['lr'])
        return lrs

    def train_step(self):
        self.model1.train()
        self.model2.train()
        #print("Train Loop!")
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

            L1 = torch.zeros((batch_size,self.latent_dimension,self.num_patches,self.num_patches))
            L1 = L1.to(self.accelarator)

            patch_dataset = PatchDataset(images,self.num_patches,self.stride,self.patch_size)
            patch_loader = DataLoader(patch_dataset,batch_size=int(math.ceil(len(patch_dataset)*self.percent_sampling)),shuffle=True)


            with torch.no_grad():
                for patches, idxs in patch_loader:
                    patches = patches.to(self.accelarator)
                    patches = patches.reshape(-1,3,self.patch_size,self.patch_size)
                    out = self.model1(patches)
                    out = out.reshape(-1,batch_size, self.latent_dimension)
                    out = torch.permute(out,(1,2,0))
                    row_idx = idxs//self.num_patches
                    col_idx = idxs%self.num_patches
                    L1[:,:,row_idx,col_idx] = out
            
            train_loss_sub_epoch = 0
            self.optimizer.zero_grad()
            for inner_iteration, (patches,idxs) in enumerate(patch_loader):

                L1 = L1.detach()
                patches = patches.to(self.accelarator)
                patches = patches.reshape(-1,3,self.patch_size,self.patch_size)
                out = self.model1(patches)
                out = out.reshape(-1,batch_size, self.latent_dimension)
                out = torch.permute(out,(1,2,0))
                row_idx = idxs//self.num_patches
                col_idx = idxs%self.num_patches
                L1[:,:,row_idx,col_idx] = out
                outputs = self.model2(L1)
                loss = self.criterion(outputs,labels)
                loss = loss/self.epsilon
                loss.backward()
                train_loss_sub_epoch += loss.item()

                if (inner_iteration + 1)%self.epsilon==0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                if inner_iteration + 1 >= self.inner_iteration:
                    break
            self.scheduler.step()
            running_loss_train += train_loss_sub_epoch

            with torch.no_grad():
                _,preds = torch.max(outputs,1)
                correct = (preds == labels).sum().item()
                train_correct += correct

                train_metrics_step = self.get_metrics(preds,labels,True)
                train_predictions = np.concatenate((train_predictions,preds.detach().cpu().numpy()))
                train_labels = np.concatenate((train_labels,labels.detach().cpu().numpy()))

            self.lr = self.get_lr(self.optimizer)
            if self.monitor_wandb:
                wandb.log({f"lrs/lr-{ii}":learning_rate for ii,learning_rate in enumerate(self.lr)})
                wandb.log({
                "train_loss_step":train_loss_sub_epoch/batch_size,
                'epoch':self.epoch,
                'train_accuracy_step_metric':train_metrics_step['accuracy'],
                'train_kappa_step_metric':train_metrics_step['kappa']})

        train_metrics = self.get_metrics(train_predictions,train_labels)
        #print(f"Train Loss: {running_loss_train/num_train} Train Accuracy: {train_correct/num_train}")
        #print(f"Train Accuracy Metric: {train_metrics['accuracy']} Train Kappa Metric: {train_metrics['kappa']}")    
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
        self.model2.eval()
        with torch.no_grad():
            #print("Validation Loop!")
            for images,labels in self.validation_loader:
                images = images.to(self.accelarator)
                labels = labels.to(self.accelarator)
                batch_size = labels.shape[0]
                
                L1 = torch.zeros((batch_size,self.latent_dimension,self.num_patches,self.num_patches))
                L1 = L1.to(self.accelarator)

                patch_dataset = PatchDataset(images,self.num_patches,self.stride,self.patch_size)
                patch_loader = DataLoader(patch_dataset,batch_size=int(math.ceil(len(patch_dataset)*self.percent_sampling)),shuffle=True)


                with torch.no_grad():
                    for patches, idxs in patch_loader:
                        patches = patches.to(self.accelarator)
                        patches = patches.reshape(-1,3,self.patch_size,self.patch_size)
                        out = self.model1(patches)
                        out = out.reshape(-1,batch_size, self.latent_dimension)
                        out = torch.permute(out,(1,2,0))
                        row_idx = idxs//self.num_patches
                        col_idx = idxs%self.num_patches
                        L1[:,:,row_idx,col_idx] = out
                
                outputs = self.model2(L1)
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
                    wandb.log({f"lrs/lr-{ii}":learning_rate for ii,learning_rate in enumerate(self.lr)})
                    wandb.log({"val_loss_step":l/batch_size,
                            "epoch":self.epoch,
                            'val_accuracy_step_metric':val_metrics_step['accuracy'],
                            'val_kappa_step_metric':val_metrics_step['kappa']})
            
            val_metrics = self.get_metrics(val_predictions,val_labels)
            #print(f"Validation Loss: {running_loss_val/num_val} Validation Accuracy: {val_correct/num_val}")
            #print(f"Val Accuracy Metric: {val_metrics['accuracy']} Val Kappa Metric: {val_metrics['kappa']}")    
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
            #print("="*31)
            #print(f"{'-'*10} Epoch {epoch+1}/{self.epochs} {'-'*10}")
            train_logs = self.train_step()
            val_logs = self.val_step()
            self.epoch = epoch
            if val_logs["loss"] < best_validation_loss:
                best_validation_loss = val_logs["loss"]
                if self.save_models:
                    torch.save({
                        'model1_weights': self.model1.state_dict(),
                        'model2_weights': self.model2.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'scheduler_state': self.scheduler.state_dict(),
                        'epoch' : epoch+1,
                    }, f"{self.model_save_dir}/best_val_loss.pt")

            if val_logs['accuracy'] > best_validation_accuracy:
                best_validation_accuracy = val_logs['accuracy']
                if self.save_models:
                    torch.save({
                        'model1_weights': self.model1.state_dict(),
                        'model2_weights': self.model2.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'scheduler_state': self.scheduler.state_dict(),
                        'epoch' : epoch+1,
                    }, f"{self.model_save_dir}/best_val_accuracy.pt")

            if val_logs['kappa'] > best_validation_metric:
                best_validation_metric = val_logs['kappa']
                if self.save_models:
                    torch.save({
                        'model1_weights': self.model1.state_dict(),
                        'model2_weights': self.model2.state_dict(),
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
                'model2_weights': self.model2.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'epoch' : epoch+1,
                }, f"{self.model_save_dir}/last_epoch.pt")
        
        return {
            'best_accuracy':best_validation_accuracy,
            'best_loss': best_validation_loss,
            'best_metric': best_validation_metric
        }