import os
import numpy as np
import cv2

from utils import get_global_variables

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


UNSEEN_VIDEOS_DIR = '../UCF_Unseen/'
UNSEEN_IMAGES_DIR = '../UCF_Images/'

def build_dataset_end_to_end(image_size=(172, 172), max_len = 40, stride = 10):
    try:
        X_train = np.load('../Numpy/End2End/X_train_'+str(image_size)+'_'+str(max_len)+'_'+str(stride)+'.npy')
        Y_train = np.load('../Numpy/End2End/Y_train_'+str(image_size)+'_'+str(max_len)+'_'+str(stride)+'.npy')
        X_test = np.load('../Numpy/End2End/X_test_'+str(image_size)+'_'+str(max_len)+'_'+str(stride)+'.npy')
        Y_test = np.load('../Numpy/End2End/Y_test_'+str(image_size)+'_'+str(max_len)+'_'+str(stride)+'.npy')
        return X_train, Y_train, X_test, Y_test
    except FileNotFoundError:     
        X_train_images = []
        Y_train_images = []
        X_test_images = []
        Y_test_images = []
    
        test_videos = [['004', '011', '007'], ['006', '010'], ['007', '002'], \
                   ['003','001'], ['006', '012', '009'], ['004', '005'], ['008','002'], \
                   ['004', '012', '002'], ['001', '013', '006']]
        
        VIDEOS_DIR, IMAGES_DIR, classes, class_to_index, videos = get_global_variables()
        
        for i in range(len(classes)):
            cls = classes[i]
            test = test_videos[i] 
            for j in range(len(videos[i])):
                vid = videos[i][j]
                video_r = VIDEOS_DIR+cls+'/'+ vid +'/'
                image_r = IMAGES_DIR+cls+'/'+ vid +'/'
                image_jpeg = IMAGES_DIR+cls+'/'+ vid +'/jpeg/'
                filelist = sorted(list(os.listdir(image_r)))
                X_train_images_class = []
                for file in filelist:
                    if file.endswith(".png"):
                        image = load_image(image_r+file,image_size)
                        X_train_images_class.append(image)
                X_train_images_class = np.array(X_train_images_class)
                
                X_test_frames = []                                
                for k in range(0,len(X_train_images_class),stride):
                    lower = k
                    upper = min(len(X_train_images_class),k+max_len)
                    if upper == len(X_train_images_class):
                        if vid not in test:                
                            X_train_images.append(pad(X_train_images_class[lower:upper],max_len))
                            Y_train_images.append(i)
                        else:
                            X_test_frames.append(pad(X_train_images_class[lower:upper],max_len))
                            X_test_images.append(np.array(X_test_frames))        
                            Y_test_images.append(i)
                        #print("Padded frames" , lower , "to" , upper)
                        break
                    else:
                        if vid not in test:                
                            X_train_images.append(X_train_images_class[lower:upper])
                            Y_train_images.append(i)
                        else:
                            X_test_frames.append(X_train_images_class[lower:upper])
                        #print("Added frames" , lower , "to" , upper)
                X_train_jpeg_class = []
                try:
                    filelist = sorted(list(os.listdir(image_jpeg)))
                except FileNotFoundError:
                    print('Not found ' + str(image_jpeg))
                    continue
                for file in filelist:
                     if file.endswith(".jpg"):
                        image = load_image(image_jpeg+file,image_size)
                        X_train_jpeg_class.append(image)
                X_train_jpeg_class = np.array(X_train_jpeg_class)
                X_test_frames = [] 
                for k in range(0,len(X_train_jpeg_class),stride):
                    lower = k
                    upper = min(len(X_train_jpeg_class),k+max_len)
                    if upper == len(X_train_jpeg_class):
                        if vid not in test:                
                            X_train_images.append(pad(X_train_jpeg_class[lower:upper],max_len))
                            Y_train_images.append(i)
                        else:
                            X_test_frames.append(pad(X_train_jpeg_class[lower:upper],max_len))
                            X_test_images.append(np.array(X_test_frames))        
                            Y_test_images.append(i)
                        #print("Padded frames" , lower , "to" , upper)
                        break
                    else:
                        if vid not in test:                
                            X_train_images.append(X_train_jpeg_class[lower:upper])
                            Y_train_images.append(i)
                        else:
                            X_test_frames.append(X_train_jpeg_class[lower:upper])
                            
                print("Processed",videos[i][j],"of","class",classes[i])
        X_train = np.array(X_train_images,dtype=np.uint8)
        Y_train = np.array(Y_train_images,dtype=np.uint8)
        X_test = np.array(X_test_images)
        Y_test = np.array(Y_test_images)
        np.save('../Numpy/End2End/X_train_'+str(image_size)+'_'+str(max_len)+'_'+str(stride)+'.npy', X_train)
        np.save('../Numpy/End2End/Y_train_'+str(image_size)+'_'+str(max_len)+'_'+str(stride)+'.npy', Y_train)
        np.save('../Numpy/End2End/X_test_'+str(image_size)+'_'+str(max_len)+'_'+str(stride)+'.npy', X_test)
        np.save('../Numpy/End2End/Y_test_'+str(image_size)+'_'+str(max_len)+'_'+str(stride)+'.npy', Y_test)
        return X_train, Y_train, X_test, Y_test
        
        
def build_test_dataset(image_size=(172, 172), stride = 10, max_len = 40, train = False):
    global UNSEEN_VIDEOS_DIR, UNSEEN_IMAGES_DIR 
    
    VIDEOS_DIR = UNSEEN_VIDEOS_DIR
    IMAGES_DIR = UNSEEN_IMAGES_DIR
    
    X_test_images = []
    Y_test_images = []
    
    classes = ['Kicking', 'Riding-Horse', 'Running', 'SkateBoarding', 'Swing-Bench', 'Lifting', 'Swing-Side', 'Walking', 'Golf-Swing']
    videos = []
    for x in classes:
        videos.append(list(os.listdir(VIDEOS_DIR+x+'/')))
    if train:
        VIDEOS_DIR, IMAGES_DIR, classes, class_to_index, videos = get_global_variables()
        
    try:
        if train:
            X_test_full = np.load('../Numpy/End2End/X_test_full_training_'+str(image_size)+'_'+str(max_len)+'_'+str(stride)+'.npy')
            Y_test_full = np.load('../Numpy/End2End/Y_test_full_training_'+str(image_size)+'_'+str(max_len)+'_'+str(stride)+'.npy') 
            return X_test_full, Y_test_full
        else:
            X_test_unseen = np.load('../Numpy/End2End/X_test_unseen_'+str(image_size)+'_'+str(max_len)+'_'+str(stride)+'.npy')
            Y_test_unseen = np.load('../Numpy/End2End/Y_test_unseen_'+str(image_size)+'_'+str(max_len)+'_'+str(stride)+'.npy')
            return X_test_unseen, Y_test_unseen
    except FileNotFoundError:
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
                X_train_images_class = np.array(X_train_images_class)        
                X_test_frames = []                                
                for k in range(0,len(X_train_images_class),stride):
                    lower = k
                    upper = min(len(X_train_images_class),k+max_len)
                    if upper == len(X_train_images_class):             
                        X_test_frames.append(pad(X_train_images_class[lower:upper],max_len))
                        X_test_images.append(np.array(X_test_frames))        
                        Y_test_images.append(i)
                        #print("Padded frames" , lower , "to" , upper)
                        break
                    else:
                        X_test_frames.append(X_train_images_class[lower:upper])
                        #print("Added frames" , lower , "to" , upper)        
                print("Processed",videos[i][j],"of","class",classes[i])
        X_test = np.array(X_test_images)
        Y_test = np.array(Y_test_images)
        if train:
            np.save('../Numpy/End2End/X_test_full_training_'+str(image_size)+'_'+str(max_len)+'_'+str(stride)+'.npy', X_test)
            np.save('../Numpy/End2End/Y_test_full_training_'+str(image_size)+'_'+str(max_len)+'_'+str(stride)+'.npy', Y_test) 
        else:
            np.save('../Numpy/End2End/X_test_unseen_'+str(image_size)+'_'+str(max_len)+'_'+str(stride)+'.npy', X_test)
            np.save('../Numpy/End2End/Y_test_unseen_'+str(image_size)+'_'+str(max_len)+'_'+str(stride)+'.npy', Y_test)
        return X_test, Y_test
