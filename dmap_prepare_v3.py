# Shortcut key : Comment (CTRL-K, CTRL-C) and Uncomment (CTRL-K, CTRL-U) are the same in Python and C++.
# python3  code.py > log.txt 2 > err.txt

import cv2
from mtcnn import MTCNN
import pandas as pd
import numpy as np
import os
import json
import sys
import tensorflow as tf

from skimage.morphology import convex_hull_image

MAIN_PATH = '/media/umit/wd4tb/xDeepFake/deepfake-detection-challenge/'
TRAIN_FOLDER = MAIN_PATH + 'train_full_videos/'
TRAIN_REAL_FACE_FOLDER = MAIN_PATH + 'train_real_face'
TRAIN_FAKE_FACE_FOLDER = MAIN_PATH + 'train_fake_face'
TRAIN_FAKE_DMAP_FOLDER = MAIN_PATH + 'train_fake_dmap'


IMG_PERCENT_PIXEL_THRESHOLD = 0.2
IMG_NOISE_THRESHOLD = 20
IMG_SIZE = 64

IMG_FORMAT = '.png'

#width = 300
#height = 300
ex = 0

# load detector
detector = MTCNN()

# Train Main Path

# folder_count = np.arange(50)
# np.random.shuffle(folder_count)

while(1):

    k = int(np.random.randint(50, size=1))

    TRAIN_SUB_FOLDER = 'dfdc_train_part_' + str(k)
    print(f"Train samples: {len(os.listdir(os.path.join(TRAIN_FOLDER,TRAIN_SUB_FOLDER)))}")

    train_list = list(os.listdir(os.path.join(TRAIN_FOLDER,TRAIN_SUB_FOLDER)))
    ext_dict = []
    for file in train_list:
        file_ext = file.split('.')[1]
        if (file_ext not in ext_dict):
            ext_dict.append(file_ext)
    print(f"Extensions: {ext_dict}")

    for file_ext in ext_dict:
        print(f"Files with extension `{file_ext}`: {len([file for file in train_list if  file.endswith(file_ext)])}")

    json_file = [file for file in train_list if  file.endswith('json')][0]
    print(f"JSON file: {json_file}")

    xMeta = pd.read_json(os.path.join(TRAIN_FOLDER,TRAIN_SUB_FOLDER,json_file),orient='index')
    j = int(np.random.randint(len(xMeta), size=1))
        
    print('video =',xMeta.axes[0][j],'-',xMeta.label[j], '-',xMeta.original[j])
    
    i = int(np.random.randint(300, size=1))
    
    print(str(k) + ".folder " + str(j) + ".video " + str(i) + ".frame")

    if xMeta.label[j] == 'FAKE':
        
        # Read Orijinal video
        video_name_original= xMeta.original[j]
        video_path_original = os.path.join(TRAIN_FOLDER,TRAIN_SUB_FOLDER,video_name_original)
        video_name_split_original = video_name_original.split('.') 
        cap_original = cv2.VideoCapture(video_path_original)
        
        # Read FAKE video
        video_name_fake = xMeta.axes[0][j]
        video_path_fake = os.path.join(TRAIN_FOLDER,TRAIN_SUB_FOLDER,video_name_fake)
        video_name_split_fake = video_name_fake.split('.') 
        cap_fake = cv2.VideoCapture(video_path_fake)
        
        cap_original.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret_original, frame_original = cap_original.read()
        
        cap_fake.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret_fake, frame_fake = cap_fake.read()
        
        if (ret_original == True and ret_fake == True):
            
            # original video face detect
            frame_temp_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB) 
            face_original = detector.detect_faces(frame_temp_original)

            if face_original: # empty control

                for multi_face_original in range(1): #range(len(face_original)):

                    box_original = face_original[multi_face_original]['box']
                    box_original = list(map(abs, box_original)) # ABS control
                    
                    x = box_original[0]
                    y = box_original[1]
                    w = box_original[2]
                    h = box_original[3]
                    
                    #cv2.rectangle(frame_original,(x,y),(x+w,y+h),(0,255,0),2)
                    crop_original = frame_original[y:y+h,x:x+w]
                    crop_fake = frame_fake[y:y+h,x:x+w]
                    
                    dmap = abs(crop_fake/1 - crop_original/1)
                    
                    dmap = np.where(dmap < IMG_NOISE_THRESHOLD, 0, dmap)
                    dmap = np.where(dmap > 0, 1, dmap)
                    print(dmap.shape)
                    
                    total_pixel = dmap.shape[0]*dmap.shape[1]*dmap.shape[2]
                    percent_pixel = (np.sum(dmap)/total_pixel)*100
                    
                    print("SUM PERCENT = "+ str(percent_pixel))
                    
                    dmap_orj = np.copy(dmap)
                    dmap_orj = np.uint8(dmap_orj*255)
                    
                    if percent_pixel > IMG_PERCENT_PIXEL_THRESHOLD:

                        try:
                            chull_x = convex_hull_image(dmap[:,:,0])
                            chull_y = convex_hull_image(dmap[:,:,1])
                            chull_z = convex_hull_image(dmap[:,:,2])
                            cx = np.logical_or(chull_x,chull_y,chull_z)
                            
                            temp  = np.zeros((dmap.shape[0],dmap.shape[1]))
                            temp[cx] = 1
                            
                            exp = np.expand_dims(temp,axis=2)
                            dmap = np.concatenate([exp,exp,exp],axis=2)
                            dmap = np.uint8(dmap*255)
                            
                        except:
                            
                            print("Convex ERROR : ", str(k) + ".folder " + str(j) + ".video " + str(i) + ".frame")
                            
                        finally:
                            
                            dmap = np.uint8(dmap)
                        
                    else:
                        dmap  = np.zeros((dmap.shape[0],dmap.shape[1],dmap.shape[2]))
                        dmap = np.uint8(dmap)
                    
                    crop = np.concatenate([crop_original,crop_fake,dmap_orj,dmap],axis=1)
                    
                    crop_original_X = cv2.resize(crop_original, (IMG_SIZE, IMG_SIZE)) 
                    crop_fake_X = cv2.resize(crop_fake, (IMG_SIZE, IMG_SIZE))
                    dmap_X = cv2.resize(dmap, (IMG_SIZE, IMG_SIZE))  
                    crop_X = np.concatenate([crop_original_X,crop_fake_X,dmap_X],axis=1)
                    
                    cv2.imwrite('./img/fake.png', crop_X)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
        cap_original.release()
        cap_fake.release()

    if xMeta.label[j] == 'REAL':
        # Read REAL video
        video_name_real = xMeta.axes[0][j]
        video_path_real = os.path.join(TRAIN_FOLDER,TRAIN_SUB_FOLDER,video_name_real)
        video_name_split_real = video_name_real.split('.') 
        cap_real = cv2.VideoCapture(video_path_real)
        
        cap_real.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret_real, frame_real = cap_real.read()
        
        if (ret_real == True):
            # REAL video face detect
            frame_temp_real = cv2.cvtColor(frame_real, cv2.COLOR_BGR2RGB) 
            face_real = detector.detect_faces(frame_temp_real)

            if face_real: # empty control

                for multi_face_real in range(1): #range(len(face_real)):
                
                    box_real = face_real[multi_face_real]['box']
                    box_real = list(map(abs, box_real)) # ABS control
                    
                    x = box_real[0]
                    y = box_real[1]
                    w = box_real[2]
                    h = box_real[3]
                    
                    #cv2.rectangle(frame_real,(x,y),(x+w,y+h),(0,255,0),2)
                    crop_real = frame_real[y:y+h,x:x+w]
                    crop_real_X = cv2.resize(crop_real, (IMG_SIZE, IMG_SIZE)) 
                    
                    dmap  = np.zeros((IMG_SIZE,IMG_SIZE,3))
                    dmap = np.uint8(dmap)
                    
                    crop_X = np.concatenate([crop_real_X,dmap],axis=1)
                    cv2.imwrite('./img/real.png', crop_X)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
        cap_real.release()
    
cv2.destroyAllWindows()     
