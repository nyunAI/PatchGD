from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.tuner.tuning import Tuner
from constants import *
from dataset_utils import get_train_val_dataset
from glob import glob
from models import *

class CustomModel(pl.LightningModule):
    def __init__(self, 
    train_dataset,
    val_dataset,
    latent_dim=LATENT_DIMENSION, 
    sampling_fraction=PERCENT_SAMPLING, 
    image_size=IMAGE_SIZE, 
    patch_size=PATCH_SIZE, 
    stride=STRIDE, 
    num_classes=NUM_CLASSES,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    gamma=GAMMA):
        super().__init__()
        # Constants:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.latent_dim = latent_dim
        self.sampling_fraction = sampling_fraction
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.num_patches = ((self.image_size-self.patch_size)//self.stride) + 1
        # Backbone and Head
        self.backbone = Backbone()
        self.head = CNN_Block()
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.head.parameters():
            param.requires_grad = True
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []
        self.train_correct = 0
        self.num_train = 0
        self.val_correct = 0
        self.num_val = 0
        self.batch_size = batch_size
        self.save_hyperparameters()
        
    def forward_no_grad_fill(self, images, L1):
        with torch.no_grad():
            for i in range(self.num_patches):
                for j in range(self.num_patches):
                    patch = images[:,:,self.stride*i:self.stride*i+self.patch_size,self.stride*j:self.stride*j+self.patch_size]
                    out = self.backbone(patch)
                    L1[:,:,i,j] = out
        return L1
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def forward(self, x, print_shape=False):
        L1 = torch.zeros((x.shape[0],self.latent_dim,self.num_patches,self.num_patches),requires_grad=False)
        L1 = L1.type_as(x)
        L1 = self.forward_no_grad_fill(x,L1)
        with torch.no_grad():
            out = L1
            for layer in self.head:
                out = layer(out)
                if print_shape:
                    print(out.size())
        return out
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        L1 = torch.zeros((x.shape[0],self.latent_dim,self.num_patches,self.num_patches),requires_grad=False)
        L1 = L1.type_as(x)
        L1 = self.forward_no_grad_fill(x,L1)
        
        patches = self.num_patches**2
        sampled = np.random.choice(patches, int(self.sampling_fraction*patches),replace=False)

        for choice in sampled:
            i = choice%self.num_patches
            j = choice//self.num_patches
            patch = x[:,:,self.stride*i:self.stride*i+self.patch_size,self.stride*j:self.stride*j+self.patch_size]
            out = self.backbone(patch)
            L1[:,:,i,j] = out
        
        outputs = self.head(L1)
        loss = self.criterion(outputs,y)
        _,preds = torch.max(outputs,1)
        self.train_correct += (preds == y).sum().item()
        self.num_train += x.shape[0]
        self.train_losses.append(loss.item())
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        L1 = torch.zeros((x.shape[0],self.latent_dim,self.num_patches,self.num_patches),requires_grad=False)
        L1 = L1.type_as(x)
        L1 = self.forward_no_grad_fill(x,L1)
        outputs = self.head(L1)
        loss = self.criterion(outputs,y)
        _,preds = torch.max(outputs,1)
        self.val_correct += (preds == y).sum().item()
        self.num_val += x.shape[0]
        self.val_losses.append(loss.item())

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return [optimizer], [scheduler]
        
    def on_train_epoch_start(self):
        self.train_correct = 0
        self.num_train = 0
        self.train_losses = []
    
    def on_train_epoch_end(self):
        train_loss = np.mean(self.train_losses)
        self.log('train_loss', train_loss,)
        train_accuracy = self.train_correct/self.num_train
        self.log('train_accuracy', train_accuracy)

    def on_validation_epoch_start(self):
        self.val_correct = 0
        self.num_val = 0
        self.val_losses = []
    
    def on_validation_epoch_end(self):
        val_loss = np.mean(self.val_losses)
        self.log('val_loss', val_loss)
        val_accuracy = self.val_correct/self.num_val
        self.log('val_accuracy', val_accuracy)

if __name__ == '__main__':
    wandb.login()
    run = wandb.init(project=EXPERIMENT, entity="gowreesh", reinit=True)

    train_dataset, val_dataset = get_train_val_dataset()
    model = CustomModel(train_dataset,val_dataset)
    checkpoint_callback_accuracy = ModelCheckpoint(dirpath=MODEL_SAVE_DIR, filename='best_accuracy_{epoch}-{val_loss:.4f}-{val_accuracy:.4f}', monitor='val_accuracy',mode='max',save_last=True,save_top_k=3)
    checkpoint_callback_loss = ModelCheckpoint(dirpath=MODEL_SAVE_DIR, filename='best_loss_{epoch}-{val_loss:.4f}-{val_accuracy:.4f}', monitor='val_loss',mode='min',save_top_k=3)
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min",patience=10)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    wandb_logger = WandbLogger()

    # Find the max batch size that can be fitted
    print("Model hyperparams before batch_size tuning =", model.hparams)

    trainer = pl.Trainer(auto_scale_batch_size='binsearch',accelerator=ACCELARATOR,devices=DEVICES,log_every_n_steps=1)
    tuner = Tuner(trainer)

    # Invoke method
    new_batch_size = tuner.scale_batch_size(model)
    print("Max Batch size: ",new_batch_size)
    # Override old batch size (this is done automatically)
    model.batch_size = new_batch_size

    print("Model hyperparams after batch_size tuning =",model.hparams)

    trainer = pl.Trainer(auto_lr_find=True,accelerator=ACCELARATOR,devices=DEVICES,log_every_n_steps=1)
    lr_finder = trainer.tuner.lr_find(model, num_training=50)
    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    print("Learning rate found: ",model.learning_rate)
    # # update hparams of the model
    model.learning_rate = new_lr

    trainer = pl.Trainer(accelerator=ACCELARATOR,
                        devices=DEVICES,
                        log_every_n_steps=1,
                        max_epochs=EPOCHS,
                        logger=wandb_logger,
                        callbacks=[early_stopping_callback,
                                   checkpoint_callback_accuracy,
                                   checkpoint_callback_loss,
                                   lr_monitor])



    print(glob(MODEL_SAVE_DIR+'*'))

    run.finish()
