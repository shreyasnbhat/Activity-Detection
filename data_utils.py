from utils import get_global_variables

UNSEEN_VIDEOS_DIR = '../UCF_Unseen/'
UNSEEN_IMAGES_DIR = '../UCF_Images/'

def build_dataset_end_to_end(image_size=(172, 172), max_len = 40, stride = 10):
    try:
        assert(max_len == 40 and stride == 10 and image_size=(172, 172))
        X_train = np.load('../Numpy/End2End/X_train.npy')
        Y_train = np.load('../Numpy/End2End/Y_train.npy')
        X_test = np.load('../Numpy/End2End/X_test.npy')
        Y_test = np.load('../Numpy/End2End/Y_test.npy')
        return X_train, Y_train, X_test, Y_test
    except FileNotFoundError, AssertionError:     
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
                        print("Padded frames" , lower , "to" , upper)
                        break
                    else:
                        if vid not in test:                
                            X_train_images.append(X_train_images_class[lower:upper])
                            Y_train_images.append(i)
                        else:
                            X_test_frames.append(X_train_images_class[lower:upper])
                        print("Added frames" , lower , "to" , upper)        
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
    
    VIDEOS_DIR = UNSEEN_VIDEO_DIR
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
        assert(max_len == 40 and stride == 10 and image_size=(172, 172))
        if train:
            X_test_full = np.load('../Numpy/End2End/X_test_full_training.npy')
            Y_test_full = np.load('../Numpy/End2End/Y_test_full_training.npy') 
            return X_test_full, Y_test_full
        else:
            X_test_unseen = np.load('../Numpy/End2End/X_test_unseen.npy')
            Y_test_unseen = np.load('../Numpy/End2End/Y_test_unseen.npy')
            return X_test_unseen, Y_test_unseen
    except FileNotFoundError, AssertionError:
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
                        print("Padded frames" , lower , "to" , upper)
                        break
                    else:
                        X_test_frames.append(X_train_images_class[lower:upper])
                        print("Added frames" , lower , "to" , upper)        
                print("Processed",videos[i][j],"of","class",classes[i])
        return np.array(X_test_images), np.array(Y_test_images)
