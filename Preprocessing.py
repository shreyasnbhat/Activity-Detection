
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import cv2
import numpy as np


# !find ./Videos/NNFL_Training_Set/ -not -name '*.avi' -delete
# 

# In[41]:


VIDEOS_DIR = './Videos/'
IMAGES_DIR = './Images/'


# In[42]:


import os
classes = list(os.listdir(VIDEOS_DIR))
print(classes)


# In[43]:


class_to_index = {}
for i in range(len(classes)):
    class_to_index[classes[i]] = i
class_to_index


# In[52]:


videos = []
for x in classes:
    videos.append(list(os.listdir(VIDEOS_DIR+x+'/')))
print(videos)


# In[53]:


# Using default 10 images per second
def convert_videos_to_images(video_path, image_path):
    print(video_path)
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    count = 0
    
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            count += 1
            name = "{0:0=3d}".format(count)
            cv2.imwrite(image_path+name+'.png', frame) 
        else: 
            break

    # When everything done, release the video capture object
    cap.release()

    print(count)


# In[56]:


for i in range(len(classes)):
    cls = classes[i]
    for j in range(len(videos[i])):
        vid = videos[i][j]
        video_r = VIDEOS_DIR+cls+'/'+ vid +'/'
        image_r = IMAGES_DIR+cls+'/'+ vid +'/'
        filelist= list(os.listdir(video_r))
        for fichier in filelist:
            if not(fichier.endswith(".avi")):
                filelist.remove(fichier)
        assert(len(filelist) == 1)
        convert_videos_to_images(video_r+filelist[0], image_r)


# In[57]:


for i in range(len(classes)):
    cls = classes[i]
    for j in range(len(videos[i])):
        vid = videos[i][j]
        video_r = VIDEOS_DIR+cls+'/'+ vid +'/'
        image_r = IMAGES_DIR+cls+'/'+ vid +'/'
        filelist = list(os.listdir(image_r))
        if len(filelist) <= 10:
            print(image_r)


# In[ ]:


def build_dataset():
    X_train = []
    for i in range(len(classes)):
    cls = classes[i]
    for j in range(len(videos[i])):
        vid = videos[i][j]
        video_r = VIDEOS_DIR+cls+'/'+ vid +'/'
        image_r = IMAGES_DIR+cls+'/'+ vid +'/'
        filelist = list(os.listdir(image_r))
        for fichier in filelist:
            if fichier.endswith(".png"):
                image = cv2.cvtColor(cv2.imread(image_r+fichier), cv2.

