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
