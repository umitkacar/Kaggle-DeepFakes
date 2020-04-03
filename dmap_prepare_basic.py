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

MAIN_PATH = '/home/umit/xDataset/deepfake-detection-challenge/'
TRAIN_FOLDER = MAIN_PATH + 'train_full_videos/'
TRAIN_REAL_FACE_FOLDER = MAIN_PATH + 'train_real_face'
TRAIN_FAKE_FACE_FOLDER = MAIN_PATH + 'train_fake_face'
TRAIN_FAKE_DMAP_FOLDER = MAIN_PATH + 'train_fake_dmap'

IMG_FORMAT = '.png'

#width = 300
#height = 300
ex = 0
video_frame_jump_fake = 300
video_frame_jump_real = 300

# load detector
detector = MTCNN()

# Train Main Path

for k in range(1):

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

    for j in range(len(xMeta)):

        print('video =',xMeta.axes[0][j],'-',xMeta.label[j], '-',xMeta.original[j])

        if xMeta.label[j] == 'FAKE':
            # Read FAKE video
            video_name_fake = xMeta.axes[0][j]
            video_path_fake = os.path.join(TRAIN_FOLDER,TRAIN_SUB_FOLDER,video_name_fake)
            video_name_split_fake = video_name_fake.split('.') 
            cap_fake = cv2.VideoCapture(video_path_fake)

            i = 0
            while(cap_fake.isOpened()):

                print(str(k) + ".folder " + str(j) + ".video " + str(i) + ".frame")

                ret_fake, frame_fake = cap_fake.read()
                cap_fake.set(cv2.CAP_PROP_POS_FRAMES, i)
                i = i + video_frame_jump_fake
        
                if (ret_fake == True):
                    # FAKE video face detect
                    cv2.cvtColor(frame_fake, cv2.COLOR_BGR2RGB, frame_fake) 
                    face_fake = detector.detect_faces(frame_fake)

                    if face_fake: # empty control

                        for multi_face_fake in range(len(face_fake)):

                            box_fake = face_fake[multi_face_fake]['box']
                            box_fake = list(map(abs, box_fake)) # ABS control
                            #cv2.rectangle(frame_fake,(box_fake[0], box_fake[1]),(box_fake[0]+box_fake[2],box_fake[1]+box_fake[3]),(0,255,0),2)
                            crop_fake = frame_fake[box_fake[1]-2*ex:box_fake[1]+box_fake[3]+ex,box_fake[0]-ex:box_fake[0]+box_fake[2]+2*ex]
                            cv2.cvtColor(crop_fake, cv2.COLOR_RGB2BGR, crop_fake) 
                            #cv2.imshow("Crop FAKE Frame Window", crop_fake)

                            # write FAKE face detect
                            img_name_fake = video_name_split_fake[0] + '_' + str(i) +  '_' + str(multi_face_fake) + IMG_FORMAT
                            img_name_path_fake = os.path.join(TRAIN_FAKE_FACE_FOLDER,img_name_fake)
                            cv2.imwrite(img_name_path_fake, crop_fake)
                    
                else:
                    break
                
            cap_fake.release()
            cv2.destroyAllWindows()

        if xMeta.label[j] == 'REAL':
            # Read REAL video
            video_name_real = xMeta.axes[0][j]
            video_path_real = os.path.join(TRAIN_FOLDER,TRAIN_SUB_FOLDER,video_name_real)
            video_name_split_real = video_name_real.split('.') 
            cap_real = cv2.VideoCapture(video_path_real)
            
            i = 0
            while(cap_real.isOpened()):

                print(str(k) + ".folder " + str(j) + ".video " + str(i) + ".frame ")
                
                ret_real, frame_real = cap_real.read()
                cap_real.set(cv2.CAP_PROP_POS_FRAMES, i)
                i = i + video_frame_jump_real
                
                if (ret_real == True):
                    # REAL video face detect
                    cv2.cvtColor(frame_real, cv2.COLOR_BGR2RGB, frame_real) 
                    face_real = detector.detect_faces(frame_real)

                    if face_real: # empty control

                        for multi_face_real in range(len(face_real)):
                        
                            box_real = face_real[multi_face_real]['box']
                            box_real = list(map(abs, box_real)) # ABS control
                            #cv2.rectangle(frame_real,(box_real[0], box_real[1]),(box_real[0]+box_real[2],box_real[1]+box_real[3]),(0,255,0),2)
                            crop_real = frame_real[box_real[1]-2*ex:box_real[1]+box_real[3]+ex,box_real[0]-ex:box_real[0]+box_real[2]+2*ex]
                            cv2.cvtColor(crop_real, cv2.COLOR_RGB2BGR, crop_real) 
                            #cv2.imshow("Crop REAL Frame Window", crop_real)

                            # write REAL face detect
                            img_name_real = video_name_split_real[0] + '_' + str(i) + '_' + str(multi_face_real) + IMG_FORMAT
                            img_name_path_real = os.path.join(TRAIN_REAL_FACE_FOLDER,img_name_real)
                            cv2.imwrite(img_name_path_real, crop_real)

                else:
                    break
                    
            cap_real.release()
            cv2.destroyAllWindows()     
