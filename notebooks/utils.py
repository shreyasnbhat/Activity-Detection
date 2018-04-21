import os
import numpy as np
from sklearn.metrics import confusion_matrix

VIDEOS_DIR = '../Videos/'
IMAGES_DIR = '../Images/'

classes = ['Kicking', 'Riding-Horse', 'Running', 'SkateBoarding', 'Swing-Bench', 'Lifting', 'Swing-Side', 'Walking', 'Golf-Swing']

class_to_index = {}

videos = []

def get_global_variables():
    global classes, videos, class_to_index
    for i in range(len(classes)):
        class_to_index[classes[i]] = i
    for x in classes:
        videos.append(list(os.listdir(VIDEOS_DIR+x+'/')))
    return VIDEOS_DIR, IMAGES_DIR, classes, class_to_index, videos


def predict(model,X,verbose=True):
    pred = model.predict(X)[0]
    max_pred = [np.argmax(i) for i in pred]
    if verbose:
        print("Max Preds time", max_pred)
    counts = np.bincount(max_pred)
    class_pred = np.argmax(counts)
    return class_pred

def evaluate(model, X_test,Y_test,verbose = True):
    count = 0
    preds = []
    for i in range(len(X_test)):
        class_pred = predict(model,X_test[i],verbose=verbose)
        preds.append(class_pred)
        actual = Y_test[i]
        if verbose:
            print("Pred",classes[class_pred],"Actual",classes[actual])
            print()
        if class_pred == actual:
            count += 1
    if verbose:
        print("Confusion Matrix")
        print(confusion_matrix(Y_test,preds))
    return float(count)/float(len(Y_test)) * 100.0
