# Copyright 2019
# 
# Yaojie Liu, Joel Stehouwer, Amin Jourabloo, Xiaoming Liu, Michigan State University
# 
# All Rights Reserved.
# 
# This research is based upon work supported by the Office of the Director of 
# National Intelligence (ODNI), Intelligence Advanced Research Projects Activity
# (IARPA), via IARPA R&D Contract No. 2017-17020200004. The views and 
# conclusions contained herein are those of the authors and should not be 
# interpreted as necessarily representing the official policies or endorsements,
# either expressed or implied, of the ODNI, IARPA, or the U.S. Government. The 
# U.S. Government is authorized to reproduce and distribute reprints for 
# Governmental purposes not withstanding any copyright annotation thereon. 
# ==============================================================================
"""
DTN for Zero-shot Face Anti-spoofing
Data Loading class.

"""
import tensorflow as tf
import numpy as np
import glob
import cv2

class Dataset():
    def __init__(self, config, mode):
        self.config = config
        if self.config.MODE == 'training':
            self.input_tensors = self.inputs_for_training(mode)
        else:
            self.input_tensors, self.name_list = self.inputs_for_testing()
        self.feed = iter(self.input_tensors)

    def inputs_for_training(self, mode):
        autotune = tf.data.experimental.AUTOTUNE
        if mode == 'train':
            data_dir = self.config.DATA_DIR
        else:
            data_dir = self.config.DATA_DIR_VAL
        data_samples = []
        for _dir in data_dir:
            _list = glob.glob(_dir+'/*.dat')
            data_samples += _list
        shuffle_buffer_size = len(data_samples)
        dataset = tf.data.Dataset.from_tensor_slices(data_samples)
        dataset = dataset.shuffle(shuffle_buffer_size).repeat(-1)
        dataset = dataset.map(map_func=self.parse_fn, num_parallel_calls=autotune)
        dataset = dataset.batch(batch_size=self.config.BATCH_SIZE).prefetch(buffer_size=autotune)
        return dataset

    def inputs_for_testing(self):
        autotune = tf.data.experimental.AUTOTUNE
        data_dir = self.config.DATA_DIR
        data_samples = []
        for _dir in data_dir:
            _list = sorted(glob.glob(_dir+'/*.dat'))
            data_samples += _list
        dataset = tf.data.Dataset.from_tensor_slices(data_samples)
        dataset = dataset.map(map_func=self.parse_fn, num_parallel_calls=autotune)
        dataset = dataset.batch(batch_size=self.config.BATCH_SIZE).prefetch(buffer_size=autotune)
        return dataset, data_samples

    def parse_fn(self, file):
        config = self.config
        image_size = config.IMAGE_SIZE
        dmap_size = config.MAP_SIZE
        label_size = 1

        def _parse_function(_file):
            
            _file = _file.decode('UTF-8')
            image_bytes = image_size * image_size * 3
            dmap_bytes = dmap_size * dmap_size
            step_bytes = image_bytes + dmap_bytes + label_size
            
            bin = np.fromfile(_file, dtype='uint8')
            maxval = int(len(bin) / step_bytes)
            
            temp = bin[0:image_bytes].shape
            if (temp[0] == 0 or maxval == 0):
                image = np.random.rand(256,256,6)
                dmap = np.ones((64,64,1))*0.5
                label = np.ones((1))*0.5
                #print("NO FACE")
            else:
                
                rand_face = tf.random.uniform(shape=[], minval=0, maxval=maxval, dtype=tf.int32)
                start_bytes = rand_face * step_bytes
                
                image_rgb = np.transpose(bin[start_bytes+0:start_bytes+image_bytes].reshape((3, image_size, image_size)), (1, 2, 0))
                image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
                
                #image_hsv = tf.image.rgb_to_hsv(image_rgb)
                image_rgb = image_rgb /255
                image_hsv = image_hsv /255
                
                image = np.concatenate([image_rgb, image_hsv], axis=2)
                
                dmap  = np.transpose(bin[start_bytes+image_bytes:start_bytes+image_bytes+dmap_bytes].reshape((1, dmap_size, dmap_size)), (1, 2, 0))
                label = bin[start_bytes+image_bytes+dmap_bytes:start_bytes+image_bytes+dmap_bytes+label_size] / 1
                dmap = np.where(dmap<2,0,dmap)
                dmap = (np.where(dmap>0,255,dmap) /255)*label
                
            return image.astype(np.float32), dmap.astype(np.float32), label.astype(np.float32)

        image_ts, dmap_ts, label_ts = tf.numpy_function(_parse_function, [file], [tf.float32, tf.float32, tf.float32])
        image_ts = tf.ensure_shape(image_ts, [config.IMAGE_SIZE, config.IMAGE_SIZE, 6])
        dmap_ts  = tf.ensure_shape(dmap_ts,  [config.MAP_SIZE, config.MAP_SIZE, 1])
        label_ts = tf.ensure_shape(label_ts, [1])
        return image_ts, dmap_ts, label_ts
