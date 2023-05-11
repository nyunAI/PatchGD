import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# load a single file as a numpy array


def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

# load a list of files into a 3D array of [samples, timesteps, features]


def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = np.dstack(loaded)
    return loaded

# load a dataset group, such as train or test


def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all NUM_TIME_SERIES files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' +
                  group + '.txt', 'total_acc_z_' + group + '.txt']
    # body acceleration
    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' +
                  group + '.txt', 'body_acc_z_' + group + '.txt']
    # body gyroscope
    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' +
                  group + '.txt', 'body_gyro_z_' + group + '.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_' + group + '.txt')
    return X, y


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

# load the dataset, returns train and test X and y elements


def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix)
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix)
    print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy


class ConvNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(NUM_TIME_SERIES, 64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.2))
        self.layer2 = nn.Flatten()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(LATENT_DIMENSION*NUM_CHUNKS, 100),
            nn.ReLU(),
            nn.Linear(100, NUM_CLASSES)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.layers(x)


class ChunkDataset(Dataset):
    def __init__(self, series, num_chunks, stride, chunk_size):
        self.series = series
        self.num_chunks = num_chunks
        self.stride = stride
        self.chunk_size = chunk_size

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, choice):
        i = choice
        return self.series[:, :, self.stride*i:self.stride*i+self.chunk_size], choice


if __name__ == '__main__':

    ACCELARATOR = 'cuda'
    EPOCHS = 20
    PERCENT_SAMPLING = 1/4
    GRAD_ACCUM = True
    BATCH_SIZE = 32
    SERIES_LEN = 128
    NUM_TIME_SERIES = 9
    CHUNK_SIZE = SERIES_LEN//8

    CHUNK_BATCHES = math.ceil(1/PERCENT_SAMPLING)
    INNER_ITERATION = CHUNK_BATCHES
    EPSILON = INNER_ITERATION if GRAD_ACCUM else 1
    STRIDE = CHUNK_SIZE

    LATENT_DIMENSION = 64*(CHUNK_SIZE-2)
    NUM_CLASSES = 6
    SEED = 42
    NUM_CHUNKS = ((SERIES_LEN-CHUNK_SIZE)//STRIDE) + 1
    NUM_WORKERS = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load data set and split into training and testing inputs (X) and outputs (y)
    trainX, trainy, testX, testy = load_dataset(
        '/workspace/data/HAR/UCI HAR Dataset/')

    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    total_step = len(trainX)

    # transformation of data into torch tensors
    trainXT = torch.from_numpy(trainX)
    # input is (N, Cin, Lin) = Ntimesteps, Nfeatures, 128
    trainXT = trainXT.transpose(1, 2).float()
    trainyT = torch.from_numpy(trainy).float()
    testXT = torch.from_numpy(testX)
    testXT = testXT.transpose(1, 2).float()
    testyT = torch.from_numpy(testy).float()

    model1 = ConvNet1D()
    model1.to(ACCELARATOR)
    model2 = ClassificationHead()
    model2.to(ACCELARATOR)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    parameters = [{'params': model1.parameters(),
                   'lr': 1e-3},
                  {'params': model2.parameters(),
                   'lr': 1e-3}]
    optimizer = torch.optim.Adam(parameters)

    # Train the model
    num_epochs = EPOCHS
    batch_size = BATCH_SIZE

    loss_list = []
    acc_list = []
    acc_list_epoch = []

    model1.train()
    model2.train()
    for epoch in range(num_epochs):
        correct_sum = 0
        running_loss_train = 0.0
        running_loss_val = 0.0
        train_correct = 0
        val_correct = 0
        num_train = 0
        num_val = 0

        for i in range(int(np.floor(total_step/batch_size))):  # split data into batches
            trainXT_seg = trainXT[i*batch_size:(i+1)*batch_size]
            labels = trainyT[i*batch_size:(i+1)*batch_size]
            trainXT_seg = trainXT_seg.to(device)
            labels = labels.to(device)

            bs = labels.shape[0]
            num_train += bs
            latent_vector = torch.zeros((bs, LATENT_DIMENSION, NUM_CHUNKS))
            latent_vector = latent_vector.to(ACCELARATOR)

            chunk_dataset = ChunkDataset(
                trainXT_seg, NUM_CHUNKS, STRIDE, CHUNK_SIZE)
            chunk_loader = DataLoader(chunk_dataset, batch_size=int(
                math.ceil(len(chunk_dataset)*PERCENT_SAMPLING)), shuffle=True)

            with torch.no_grad():
                for chunks, idxs in chunk_loader:
                    chunks = chunks.to(ACCELARATOR)
                    chunks = chunks.reshape(-1, NUM_TIME_SERIES, CHUNK_SIZE)
                    out = model1(chunks)
                    out = out.reshape(-1, bs, LATENT_DIMENSION)
                    out = torch.permute(out, (1, 2, 0))
                    latent_vector[:, :, idxs] = out

            train_loss_sub_epoch = 0
            optimizer.zero_grad()
            for inner_iteration, (chunks, idxs) in enumerate(chunk_loader):
                latent_vector = latent_vector.detach()
                chunks = chunks.to(ACCELARATOR)
                chunks = chunks.reshape(-1, NUM_TIME_SERIES, CHUNK_SIZE)
                out = model1(chunks)
                out = out.reshape(-1, bs, LATENT_DIMENSION)
                out = torch.permute(out, (1, 2, 0))
                latent_vector[:, :, idxs] = out
                outputs = model2(latent_vector)
                loss = criterion(outputs, labels)
                loss = loss/EPSILON
                loss.backward()
                train_loss_sub_epoch += loss.item()

                if (inner_iteration + 1) % EPSILON == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if inner_iteration + 1 >= INNER_ITERATION:
                    break

            # Adding all the losses... Can be modified??
            running_loss_train += train_loss_sub_epoch

            # Using the final L1 to make the final set of predictions for accuracy reporting
            with torch.no_grad():
                _, preds = torch.max(outputs, 1)
                _, actual = torch.max(labels, 1)
                correct = (preds == actual).sum().item()
                train_correct += correct

        print(
            f"Train Loss: {running_loss_train/num_train} Train Accuracy: {train_correct/num_train}")

    # Test the model

    model1.eval()
    model2.eval()

    correct_sum = 0
    running_loss_train = 0.0
    running_loss_val = 0.0
    train_correct = 0
    val_correct = 0
    num_train = 0
    num_val = 0

    testXT_seg = testXT
    labels = testyT
    testXT_seg = testXT_seg.to(device)
    labels = labels.to(device)

    bs = labels.shape[0]
    num_val += bs
    latent_vector = torch.zeros((bs, LATENT_DIMENSION, NUM_CHUNKS))
    latent_vector = latent_vector.to(ACCELARATOR)

    chunk_dataset = ChunkDataset(testXT_seg, NUM_CHUNKS, STRIDE, CHUNK_SIZE)
    chunk_loader = DataLoader(chunk_dataset, batch_size=int(
        math.ceil(len(chunk_dataset)*PERCENT_SAMPLING)), shuffle=True)

    with torch.no_grad():
        for chunks, idxs in chunk_loader:
            chunks = chunks.to(ACCELARATOR)
            chunks = chunks.reshape(-1, NUM_TIME_SERIES, CHUNK_SIZE)
            out = model1(chunks)
            out = out.reshape(-1, bs, LATENT_DIMENSION)
            out = torch.permute(out, (1, 2, 0))
            latent_vector[:, :, idxs] = out

        outputs = model2(latent_vector)
        _, preds = torch.max(outputs, 1)
        _, actual = torch.max(labels, 1)
        correct = (preds == actual).sum().item()
        val_correct += correct

    print(f"Test Accuracy: {val_correct/num_val}")
