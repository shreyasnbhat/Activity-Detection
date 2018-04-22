import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from keras import backend as K
from keras.layers import Conv2D, Dropout, LSTM, BatchNormalization, Input,Activation, MaxPool2D, Flatten, Dense,TimeDistributed
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers.convolutional import ZeroPadding2D
from keras import metrics
import h5py
from sklearn.metrics import confusion_matrix
from utils import *
from data_utils import *
from models import *
import math
import time

VIDEOS_DIR, IMAGES_DIR, classes, class_to_index, videos = get_global_variables()
print(VIDEOS_DIR)
print(IMAGES_DIR)
print(classes)
print(class_to_index)
print(videos)

X_train = np.load('../Numpy/End2End/X_aug.npy')
Y_train = np.load('../Numpy/End2End/Y_aug.npy')

def generate_batch_data(train_size,batch_size,suffix = 'aug'):

    while 1:
        i = 0
        for i in range(0,train_size,batch_size):
            X_batch = '../Numpy/End2End/batch' + str(batch_size) + '/X_' + suffix + '_' + str(i) + '.npy'
            Y_batch =  '../Numpy/End2End/batch' + str(batch_size) + '/Y_' + suffix + '_' + str(i) + '.npy'
            Y_batch_2 = '../Numpy/End2End/batch' + str(batch_size) + '/Y2_' + suffix + '_' + str(i) + '.npy'

            batch =  (np.load(X_batch),[np.load(Y_batch),np.load(Y_batch_2)])
            yield batch

batch_size = 80
train_size = X_train.shape[0]
steps = math.ceil(train_size/batch_size)
del X_train
del Y_train

gen = generate_batch_data(train_size,batch_size)

e2e = load_model('../models/End_End/gpu_then_cpu.h5')
e2e.loss_weights = [1,0.75]
epochs = 10
for j in range(epochs):
    ini = time.time()
    for i in range(steps):
        res = next(gen)
        print(res[0].shape,res[1][0].shape)
        e2e.train_on_batch(res[0],res[1])
        del res
    print("Epoch",j+1," Time:",str(time.time() - ini) + 's')
    e2e.save('temp_' + str(j) + '.h5')
