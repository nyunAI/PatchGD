import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import os
DEVICE_ID = 2 ######################################
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
import cv2
import PIL
from tensorflow import keras
from IPython.display import Image, display
from keras.applications.vgg16 import VGG16,preprocess_input
from sklearn.metrics import cohen_kappa_score
from checkmate.tf2 import get_keras_model
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model,load_model
from keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers import Dense,Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten,BatchNormalization,Activation,GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.models import Model
import tensorflow_addons as tfa
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical, load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import gc
import albumentations as A
from tqdm import tqdm
from sklearn.model_selection import KFold
import tensorflow as tf
import math
from tensorflow.python.keras import backend as K
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class WarmupLinearDecayLRScheduler(
      tf.keras.optimizers.schedules.LearningRateSchedule):

        def __init__(self, 
                      max_lr: float,
                      warmup_steps: int,
                      decay_steps: int) -> None:
            super(WarmupLinearDecayLRScheduler, self).__init__()

            self.name = 'WarmupLinearDecayLRScheduler'

            self.max_lr = max_lr
            self.last_step = 0

            self.warmup_steps = int(warmup_steps)
            self.linear_increase = self.max_lr / float(self.warmup_steps)

            self.decay_steps = decay_steps

        def _decay(self):
            rate = tf.subtract(self.last_step, self.warmup_steps) 
            rate = tf.divide(rate, 2*self.decay_steps)
            rate = tf.subtract(1.0, rate)
            rate = tf.cast(rate, tf.float32)
            decayed = rate
            return tf.multiply(self.max_lr, decayed)

        def __call__(self, step):
          self.last_step = step
          dtype = type(self.linear_increase)
          self.last_step = tf.cast(self.last_step,dtype)
          lr_s = tf.cond(
                        tf.less(self.last_step, self.warmup_steps),
                        lambda: tf.multiply(self.linear_increase, self.last_step),
                        lambda: self._decay())
          return lr_s

        def get_config(self) -> dict:
            config = {
                "max_lr": self.max_lr,
                "warmup_steps": self.warmup_steps,
                'decay_steps': self.decay_steps,
            }
            return config

def transform(image):
        aug = A.Compose([
            A.Normalize(mean=MEAN,std=STD,max_pixel_value=255.0),
        ])
        return aug(image=image)['image']

def resnet50_model(num_classes=None):

        model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE,IMAGE_SIZE, 3))
        
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        output=Dense(num_classes)(x)
        model=Model(model.input,output)
        return model


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*18)])
      except RuntimeError as e:
        print(e)


    now = datetime.now() 
    date_time = now.strftime("%d_%m_%Y__%H_%M")
    MAIN_RUN = True

    MONITOR_WANDB = False ######################################
    SANITY_CHECK = False
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    ACCELARATOR = 'cuda:0'
    BATCH_SIZE = 27 ######################################
    MEMORY = '16' ######################################
    SAVE_MODELS = False
    SCALE_FACTOR = 1 ######################################
    IMAGE_SIZE = 512*SCALE_FACTOR
    WARMUP_EPOCHS = 2
    EXPERIMENT = "pandas-shared-runs-icml-rebuttal" if not SANITY_CHECK else 'pandas-sanity-gowreesh'
    # RUN_NAME = f'{DEVICE_ID}-{IMAGE_SIZE}-{BATCH_SIZE}-resnet50-baseline-{MEMORY}GB-datetime_{date_time}' ######################################


    NUM_CLASSES = 6
    SEED = 42
    NUM_WORKERS = 4
    TRAIN_ROOT_DIR = f'./pandas_dataset/training_images_{IMAGE_SIZE}/'
    VAL_ROOT_DIR = TRAIN_ROOT_DIR
    TRAIN_CSV_PATH = f'./pandas_dataset/train_kfold.csv'
    MEAN = [0.9770, 0.9550, 0.9667]
    STD = [0.0783, 0.1387, 0.1006]
    SANITY_DATA_LEN = None
    # MODEL_SAVE_DIR = f"../{'models_icml' if MAIN_RUN else 'models'}/{'sanity' if SANITY_CHECK else 'runs'}/{RUN_NAME}"
    DECAY_FACTOR = 2
    VALIDATION_EVERY = 1
    BASELINE = True
    
    
    tf.keras.utils.set_random_seed(
        SEED
    )
    df = pd.read_csv(TRAIN_CSV_PATH)
    print(df.head())

    df['images'] = TRAIN_ROOT_DIR + df['image_id'].astype(str) + '.png'

    train_df = df[df['kfold']!=0]
    val_df = df[df['kfold']==0]

    train_df = train_df[['images','isup_grade']]
    val_df = val_df[['images','isup_grade']]

    train_df['isup_grade']=train_df['isup_grade'].astype(str)
    val_df['isup_grade']=val_df['isup_grade'].astype(str)

    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))


    
    train_datagen = ImageDataGenerator(preprocessing_function=transform)
    val_datagen = ImageDataGenerator(preprocessing_function=transform)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='images',
        y_col='isup_grade',
        target_size=(IMAGE_SIZE,IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse')

    validation_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col='images',
        y_col='isup_grade',
        target_size=(IMAGE_SIZE,IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse')


    element_spec = train_generator.__iter__().__next__()
    print(element_spec[0].shape,element_spec[1])
    
    model = resnet50_model(NUM_CLASSES)
    print(model.summary())
    for layer in (model.layers) :
        layer.trainable = True

    LR_schedule = WarmupLinearDecayLRScheduler(max_lr=LEARNING_RATE,
                                          decay_steps=int((EPOCHS- WARMUP_EPOCHS) * (len(train_df))/BATCH_SIZE),
                                          warmup_steps=int(WARMUP_EPOCHS * (len(train_df))/BATCH_SIZE))

    optimizer= tf.keras.optimizers.Adam(LR_schedule) 

    nb_train_steps = train_df.shape[0]/BATCH_SIZE
    nb_val_steps=val_df.shape[0]/BATCH_SIZE
    print("Number of training and validation steps: {} and {}".format(nb_train_steps,nb_val_steps))


    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    train_kappa_metric = tfa.metrics.CohenKappa(num_classes=NUM_CLASSES,
        weightage='quadratic',sparse_labels=True
    )
    val_kappa_metric = tfa.metrics.CohenKappa(num_classes=NUM_CLASSES,
        weightage='quadratic',sparse_labels=True
    )
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    best_validation_accuracy = 0
    best_validation_metric = -float('inf')

    for epoch in range(EPOCHS):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        batches = 0
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(tqdm(train_generator)):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            lr = optimizer.learning_rate.numpy()
            batches += 1
            if batches >= nb_train_steps:
                break
            train_acc_metric.update_state(y_batch_train, logits)
            train_kappa_metric.update_state(y_batch_train, logits)
        print(f'lr:{lr}')
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        train_kappa = train_kappa_metric.result()
        print("Training kappa over epoch: %.4f" % (float(train_kappa),))
        batches = 0
        train_acc_metric.reset_states()
        train_kappa_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in tqdm(validation_generator):
            val_logits = model(x_batch_val, training=False)
            val_acc_metric.update_state(y_batch_val, val_logits)
            val_kappa_metric.update_state(y_batch_val, val_logits)
            batches += 1
            if batches >= nb_val_steps:
                break
        val_acc = val_acc_metric.result()
        val_kappa = val_kappa_metric.result()
        val_kappa_metric.reset_states()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Validation kappa: %.4f" % (float(val_kappa),))


        if val_acc > best_validation_accuracy:
            best_validation_accuracy = val_acc
        if val_kappa > best_validation_metric:
            best_validation_metric = val_kappa

        print("Time taken: %.2fs" % (time.time() - start_time))

    print(f'Best Accuracy:{best_validation_accuracy}, Best Kappa: {best_validation_metric}')


    # model.compile(optimizer=optimizer,
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy',tfa.metrics.CohenKappa(num_classes=NUM_CLASSES, sparse_labels=True,weightage='quadratic')])

    # nb_epochs = EPOCHS
    # batch_size= BATCH_SIZE
    # nb_train_steps = train_df.shape[0]//batch_size
    # nb_val_steps=val_df.shape[0]//batch_size
    # print("Number of training and validation steps: {} and {}".format(nb_train_steps,nb_val_steps))
    # model.fit_generator(generator=train_generator,
    #     steps_per_epoch=nb_train_steps, epochs=nb_epochs,
    #     validation_data=validation_generator, validation_steps=nb_val_steps)