import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from keras import backend as K
from keras.layers import Conv2D, Dropout, LSTM, BatchNormalization, Input,Activation, MaxPool2D, Flatten, Dense,TimeDistributed
from keras.models import Model, load_model
from keras import metrics
import h5py

VIDEOS_DIR = './Videos/'
IMAGES_DIR = './Images/'
classes = []
class_to_index = {}
videos = []

def pre_compute():
    global classes
    global class_to_index
    global videos

    classes = list(os.listdir(VIDEOS_DIR))
    print(classes)

    for i in range(len(classes)):
        class_to_index[classes[i]] = i
    class_to_index

    for x in classes:
        videos.append(list(os.listdir(VIDEOS_DIR+x+'/')))
    print(videos)

def permute(X,Y):
    train_size = X.shape[0]
    permutation_train = np.random.permutation(train_size)
    X = X[permutation_train]
    Y = Y[permutation_train]
    return X,Y

def load_image(path,image_size):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    return image

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def pad_end_to_end(X_train_images_class,pad_len):
    length = X_train_images_class.shape[0]
    pad_arr = np.zeros((X_train_images_class.shape[1:4]),dtype=np.uint8)
    X_train_images_class = list(X_train_images_class)
    for i in range(pad_len-length):
        X_train_images_class.append(pad_arr)
    return np.array(X_train_images_class,dtype=np.uint8)

def build_dataset_end_to_end(image_size, pad_len = 170):
    global classes
    global videos

    X_train_images = []
    Y_train_images = []
    for i in range(len(classes)):
        cls = classes[i]
        for j in range(len(videos[i])):
            vid = videos[i][j]
            video_r = VIDEOS_DIR+cls+'/'+ vid +'/'
            image_r = IMAGES_DIR+cls+'/'+ vid +'/'
            filelist = sorted(list(os.listdir(image_r)))
            X_train_images_class = []
            for file in filelist:
                if file.endswith(".png"):
                    image = load_image(image_r+file,image_size)
                    X_train_images_class.append(image)
            X_train_images_class = pad_end_to_end(np.array(X_train_images_class,dtype=np.uint8),pad_len) # Pad till 170 frames
            assert(X_train_images_class.shape == (170,172, 172, 3))
            X_train_images.append(X_train_images_class)
            Y_train_images.append(i)
            print("Processed",videos[i][j],"of","class",classes[i])
    return np.array(X_train_images,dtype=np.uint8),np.array(Y_train_images,dtype=np.uint8)

def end_to_end(input_shape):
    X_input = Input(input_shape)
    X = TimeDistributed(BatchNormalization(name = 'BatchNorm_1'))(X_input)
    X = TimeDistributed(Conv2D(32, (7, 7), strides = (2, 2), activation='relu', name="Conv_1a", padding="same"))(X)
    X = TimeDistributed(Conv2D(32, (3, 3), strides = (2, 2), activation='relu', name="Conv_1b", padding="same"))(X)
    X = TimeDistributed(MaxPool2D((2, 2), name = "Pool_1"))(X)

    X = TimeDistributed(Conv2D(64, (3, 3), name ="Conv_2a", activation='relu', padding = "same"))(X)
    X = TimeDistributed(Conv2D(64, (3, 3), name ="Conv_2b", activation='relu', padding = "same"))(X)
    X = TimeDistributed(MaxPool2D((2, 2), name = "Pool_2"))(X)

    X = TimeDistributed(Conv2D(256,(3,3), name='Conv_3a'))(X)
    X = TimeDistributed(MaxPool2D((4, 4), name = "Pool_3"))(X)

    X = TimeDistributed(Flatten())(X)

    X = LSTM(32, return_sequences=True)(X)
    X = LSTM(32, return_sequences=False)(X)
    X = Dense(9, activation='softmax')(X)

    return Model(X_input,X)


if __name__ == '__main__' :

    pre_compute()
    print("Pre Computation Done")
    e2e = end_to_end((170, 172, 172, 3))
    print(e2e.summary())

    try:
        X = np.load('X_e2e.npy')
        Y = np.load('Y_e2e.npy')
    except FileNotFoundError:
        X,Y = build_dataset_end_to_end((172, 172))
        np.save('X_e2e.npy',X)
        np.save('Y_e2e.npy',Y)

    Y = convert_to_one_hot(Y,9)
    print("Shape of X" , X.shape)
    print("Shape of Y" , Y.shape)

    e2e.compile(loss='categorical_crossentropy',
            metrics=['accuracy'],
            optimizer='adam')

    X_train_e2e,Y_train_e2e = permute(X,Y)

    for i in range(epochs):
        e2e.fit(X_train_e2e, Y_train_e2e, epochs=1, batch_size = 8, validation_split=0.05)
        e2e.save('epoch_e2e'+str(i)+'.h5')
