
import sys
from datetime import datetime
import time
from mtcnn import MTCNN

import tensorflow as tf
import numpy as np
import cv2

from model.dtn import DTN

DEEPFAKE_MODEL_PATH = "/home/umit/xDeepFake/log/model/ckpt-204"

def leaf_l1_score(xlist, masklist, ch=None):
    loss_list = []
    xshape = xlist[0].shape
    scores = []
    for x, mask in zip(xlist, masklist):
        if ch is not None:
            score = tf.reduce_mean(tf.reshape(tf.abs(x[:, :, :, ch]), [xshape[0], -1]), axis=1)
        else:
            score = tf.reduce_mean(tf.reshape(tf.abs(x), [xshape[0], -1]), axis=1)
        spoof_score = score * mask[:, 0]
        scores.append(spoof_score)
    loss = np.sum(np.stack(scores, axis=1), axis=1)
    return loss

def _from_np_to_tf_func(image,label):
    return image.astype(np.float32), label.astype(np.float32)

if __name__ == "__main__":
    
    detector = MTCNN()
    dtn = DTN(32)
    dtn_op = tf.compat.v1.train.AdamOptimizer(0.0005, beta1=0.5)
    checkpoint = tf.train.Checkpoint(dtn=dtn,
                                     dtn_optimizer=dtn_op)
    checkpoint.restore(DEEPFAKE_MODEL_PATH)
    
    #cap = cv2.VideoCapture("/media/umit/wd4tb/xDeepFake/deepfake-detection-challenge/train_full_videos/dfdc_train_part_17/hugcokpuks.mp4")
    cap = cv2.VideoCapture("/home/umit/xDataset/deepfake-detection-challenge/test_videos/bwdmzwhdnw.mp4")
    # OpenCV image config                
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    upLeftCornerOfText     = (50,50)
    fontScale              = 1
    
    lineType               = 3
    fontColor              = (0,255,0)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        #frame = cv2.resize(frame, (1280,720))
        #frame = cv2.flip(cv2.transpose(frame), flipCode=1)
        if ret==True:
            xFace = detector.detect_faces(frame)
            if xFace:
                xBox = xFace[0]['box']
                xBox = list(map(abs, xBox))
                x = xBox[0];
                y = xBox[1];
                w = xBox[2];
                h = xBox[3];
                crop = frame[y:y+w,x:x+h]
                #crop = frame[xBox[1]:xBox[1]+xBox[3],xBox[0]:xBox[0]+xBox[2]]
                #cv2.imshow('crop',crop)
                cv2.rectangle(frame,(xBox[0], xBox[1]),(xBox[0]+xBox[2],xBox[1]+xBox[3]),(0,255,0),2);
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                
                crop_rgb = cv2.resize(crop_rgb, (256,256))
                crop_hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
                crop_rgb = crop_rgb / 255 
                crop_hsv = crop_hsv / 255
                
                image = np.concatenate([crop_rgb, crop_hsv], axis=2)
                
                extended_img = np.expand_dims(image, axis=0)
                extended_label = np.ones(shape=(1,1))
                
                image_ts, label_ts = tf.numpy_function(_from_np_to_tf_func, [extended_img, extended_label], [tf.float32, tf.float32])
                
                with tf.GradientTape() as tape:
                    dmap_pred, cls_pred, route_value, leaf_node_mask = dtn(image_ts, label_ts, False)
                 # Fusion score
                
                dmap_score = leaf_l1_score(dmap_pred, leaf_node_mask)
                cls_score = leaf_l1_score(cls_pred,leaf_node_mask)
                print("dmap_score = " + str("%.3f\n" % dmap_score))
                print("cls_score = " + str("%.3f\n" % cls_score))
                
                if dmap_score <= 0.1 and cls_score <= 0.2:
                    last_score = 0.2
                elif dmap_score > 0.1 and dmap_score <= 0.2 and cls_score <= 0.3:
                    last_score = 0.3
                elif dmap_score > 0.2 and dmap_score <= 0.3 and cls_score <= 0.4:
                    last_score = 0.4
                elif dmap_score > 0.3 and dmap_score <= 0.4 and cls_score >= 0.6:
                    last_score = 0.6
                elif dmap_score > 0.4 and dmap_score <= 0.45 and cls_score >= 0.8:
                    last_score = 0.75
                elif dmap_score > 0.45 and cls_score >= 0.9:
                    last_score = 0.85
                else:
                    last_score = 0.5
                
                if(last_score < 0.5):
                    result = "Real"
                    fontColor              = (0,255,0)
                elif(last_score == 0.5):
                    result = "Unknown"
                    fontColor              = (0,255,255)
                else:
                    result = "Fake"
                    fontColor              = (0,0,255)
                    
                print(result + " " + str("%.3f\n" % last_score))
                cv2.rectangle(frame,(x,y),(x+w,y+h),fontColor,lineType);
                cv2.putText(frame,result,(x,y-10),font,fontScale,fontColor,lineType)

            cv2.imshow('frame',frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        
    cap.release()
    cv2.destroyAllWindows()
   
        