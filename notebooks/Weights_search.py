import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from keras import backend as K
from keras.layers import Conv2D, Dropout, LSTM, BatchNormalization, Input,Activation, MaxPool2D, Flatten, Dense,TimeDistributed
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers.convolutional import ZeroPadding2D
from keras import metrics
import h5py
from sklearn.metrics import confusion_matrix
from utils import *
from data_utils import *
from models import *

X_train, Y_train, X_test,Y_test = build_dataset_end_to_end((172, 172))

def create_model(X_train, Y_train, X_test, Y_test):
    Y_train = convert_to_one_hot(Y_train, 9)
    Y_train2 = np.tile(Y_train, (40, 1, 1))
    Y_train2 = Y_train2.transpose((1, 0, 2))
    print(Y_train2.shape)

    for weight in [0.2, 0.4, 0.6, 0.8]:
        model = load_model('../models/End_End/temp0_ke_baad_3_epoch_phir_ulta_weights.h5')
        model.loss_weights = [1, weight]
        model.fit(X_train, [Y_train, Y_train2],
                  batch_size=64,
                  epochs=1,
                  verbose=2)
        acc = evaluate(model, X_test, Y_test)
        print('Test accuracy:', acc)
        model.save('ulta_'+str(weight)+'.h5')

create_model(X_train, Y_train, X_test,Y_test)


