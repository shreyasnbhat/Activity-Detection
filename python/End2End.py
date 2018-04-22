import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from keras import backend as K
from keras.layers import Conv2D, Dropout, LSTM, BatchNormalization, Input,Activation, MaxPool2D, Flatten, Dense,TimeDistributed
from keras.models import Model, load_model
from keras.layers.convolutional import ZeroPadding2D
from keras import metrics
import h5py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


VIDEOS_DIR = './Videos/'
IMAGES_DIR = './Images/'
classes = []
class_to_index = {}
videos = []

def pre_compute():
    global classes
    global class_to_index
    global videos

    classes = ['Kicking', 'Riding-Horse', 'Running', 'SkateBoarding', 'Swing-Bench', 'Lifting', 'Swing-Side', 'Walking', 'Golf-Swing']
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

def pad(X_train_images_class,max_len):
    length = len(X_train_images_class)
    pad_arr = np.zeros((X_train_images_class.shape[1:4]),dtype=np.uint8)
    X_train_images_class = list(X_train_images_class)
    for i in range(max_len-length):
        X_train_images_class.append(pad_arr)
    return np.array(X_train_images_class,dtype=np.uint8)

# Don't do one hot
def evaluate(model, X_test,Y_test,verbose = False):
    global classes
    count = 0
    for i in range(len(X_test)):
        pred = model.predict(X_test[i])[0]
        #print(pred[0].shape, pred[1].shape)
        #break
        max_pred = [np.argmax(i) for i in pred]
        counts = np.bincount(max_pred)
        class_pred = np.argmax(counts)
        #class_pred = max_pred
        actual = Y_test[i]
        if verbose:
            print("Max Preds time", max_pred)
            print("Pred",classes[class_pred],"Actual",classes[actual])
            print()
        if class_pred == actual:
            count += 1
    return count * 100 /float(len(Y_test)) 


if __name__ == '__main__' :
    pre_compute()
    print("Pre Computation Done")
    
    X_train = np.load('./Numpy/End2End/X_train_10_40.npy')
    Y_train = np.load('./Numpy/End2End/Y_train_10_40.npy')
    Y_train = convert_to_one_hot(Y_train, 9)
    print("Shapes: X_train " + str(X_train.shape) + " Y_train " + str(Y_train.shape))
    Y_train2 = np.tile(Y_train, (40, 1, 1))
    Y_train2 = Y_train2.transpose(1, 0, 2)
    print("Loaded dataset")
    
    model = load_model('./models/End_End/model_time_12.h5')
    
    for i in range(5):
        model.fit(X_train, [Y_train, Y_train2], epochs =1, batch_size = 64, validation_split = 0.2)
        model.save('temp' + str(i) + '.h5')
    
    X_test = np.load('./Numpy/End2End/X_test_10_40.npy')
    Y_test = np.load('./Numpy/End2End/Y_test_10_40.npy')
    
    print("{Test Accuracy : " + str(evaluate(model, X_test, Y_test, verbose=True)) +" }")


