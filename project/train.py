from constants import *
from models import *
from dataset_utils import *
from utils import *

if __name__ == '__main__':
    # Train and Validation Dataset:
    train_loader, val_loader = get_train_val_loaders(print_lengths=True)
    model = CustomModel()

    trainer = pl.Trainer(accelerator='gpu', devices=GPUS, auto_scale_batch_size=True)
    trainer.fit(model, train_loader, val_loader)
    
