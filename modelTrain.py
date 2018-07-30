#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:57:02 2018

@author: ZSQ
"""
import numpy as np
import os
import tensorflow as tf
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
#from generatenpy import dataProcess
import matplotlib.pyplot as plt
from keras.callbacks import *
#from keras.utils.vis_utils import plot_model
from skimage.io import imsave, imread
import shutil
import time
import h5py
import pickle
import glob
import re
import fnmatch
import gc


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

starttime = time.clock()

class myUnet(object):

    def __init__(self, img_rows = 192, img_cols = 144):

        self.img_rows = img_rows
        self.img_cols = img_cols
        

    def read_path(self, train_data_path, classify_mode):
        file_names = glob.glob(train_data_path)  #读取文件路径
        fname_path = []
        for i in range(len(file_names)):
            fname_all = os.path.basename(file_names[i])
            cls_name = re.split('cls', fname_all)[0]
            if fnmatch.fnmatch(cls_name, 'traindata_' + str(classify_mode)):
                fname_path.append(file_names[i])
                
        '''
        for j in range(num_file):
            data = np.load(fname_path[j])
            imgs_train = data["train_data"]
            imgs_mask_train = data["train_label"]
        '''
        return fname_path
    
    def data_generator(self, data, targets, batch_size):
        batches = (len(data) + batch_size - 1)//batch_size
        while(True):
            for i in range(batches):
                X = data[i*batch_size : (i+1)*batch_size]
                Y = targets[i*batch_size : (i+1)*batch_size]
                yield (X, Y)
    

    def get_model(self):
        inputs = Input((self.img_rows, self.img_cols,4))		
        
        #unet with crop(because padding = valid) 
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
        print("conv1 shape:",conv1.shape)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv1)
        print("conv1 shape:",conv1.shape)
        crop1 = Cropping2D(cropping=((0,0),(0,0)))(conv1)
        print("crop1 shape:",crop1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:",pool1.shape)
        
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(pool1)
        print("conv2 shape:",conv2.shape)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv2)
        print("conv2 shape:",conv2.shape)
        crop2 = Cropping2D(cropping=((0,0),(0,0)))(conv2)
        print("crop2 shape:",crop2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:",pool2.shape)
        
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(pool2)
        print("conv3 shape:",conv3.shape)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
        print("conv3 shape:",conv3.shape)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv3)
        print("conv3 shape:",conv3.shape)
        crop3 = Cropping2D(cropping=((0,0),(0,0)))(conv3)
        print("crop3 shape:",crop3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:",pool3.shape)
        
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool3)
        print("conv4 shape:",conv4.shape)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv4)
        #drop4 = Dropout(0.5)(conv4)
        crop4 = Cropping2D(cropping=((0,0),(0,0)))(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)          #above is not trainable
        
        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same')(pool4)
        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv5)
        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv5)
        drop5 = Dropout(0.5)(conv5)
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)      #above is VGG16 model
         
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
        merge6 = merge([crop4,up6], mode = 'concat', concat_axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same')(conv6)
        
        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
        merge7 = merge([crop3,up7], mode = 'concat', concat_axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same')(conv7)
        
        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
        merge8 = merge([crop2,up8], mode = 'concat', concat_axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same')(conv8)
        
        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
        merge9 = merge([crop1,up9], mode = 'concat', concat_axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv9)
        conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same')(conv9)
        #conv10 = Conv2D(4, 1, 1, activation = 'softmax')(conv9)
        conv10 = Conv2D(2, 1, 1, activation = 'sigmoid')(conv9)
        
        vgg16_model = Model(input = inputs, output = pool5)
        my_model = Model(input = inputs, output = conv10)
        print("============")
        print("the input is :" )
        print(inputs)
        print("the output is :" )
        print(vgg16_model.output)
        
        return my_model, vgg16_model
    
    def train_fitune(self, classify_mode, weights_path, fname_path):
        print("loading data")
               
        
        data = np.load(fname_path)
        imgs_train = data["train_data"]
        imgs_mask_train = data["train_label"]
        imgs_mask_train = imgs_mask_train[:,:,:,np.newaxis]
        
        print("loading data done")

        my_model, vgg16_model = self.get_model()        
        print("got unet")
                   
        my_model.load_weights(weights_path, by_name = True)   #load weights for VGG16
        print('Model loaded.')
        
        
        # set the first 25 layers (up to the last conv block)
        # to non-trainable (weights will not be updated)
        '''
        for layer in my_model.layers[:14]:
            layer.trainable = False
        '''
        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        if classify_mode == 2:
            sgd = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
            my_model.compile(loss='binary_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy']) 
            
        if classify_mode == 3 or classify_mode == 4 or classify_mode == 5:
            sgd = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
            my_model.compile(loss='categorical_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])  
                     
        model_path = "./save_model/my_model_" + str(classify_mode) + '_0' + ".hdf5"
        model_checkpoint = ModelCheckpoint(model_path, monitor='loss',verbose=1, save_best_only=True)
        print('Fitting model...')
        my_model.fit(imgs_train, imgs_mask_train, batch_size=5, epochs=1, verbose=1, shuffle=True,validation_split=0.1, callbacks=[model_checkpoint])
        del imgs_train, imgs_mask_train
        gc.collect()
        print('train step of fintune is done')
        return model_path


    def train_next(self, classify_mode, data_path, weights_path, weights_top_path, step):
        #log_filepath = "./train_log/train_log_" + str(step)
        print("loading data")
        
        data = np.load(data_path)
        imgs_train = data["train_data"]
        imgs_mask_train = data["train_label"]
        
        print("loading data done")

        my_model, vgg16_model = self.get_model()        
        print("got unet")
           
		#Visualize model
        plot_model(my_model, './model-architecture/my-model-architecture.png', show_shapes=True)
        
        my_model.load_weights(weights_path, by_name = True)   #load weights for VGG16
        my_model.load_weights(weights_top_path, by_name = True)  #load weights for top layers
        print('Model loaded.')
        
        
        # set the first 25 layers (up to the last conv block)
        # to non-trainable (weights will not be updated)
        '''
        for layer in my_model.layers[:14]:
            layer.trainable = False
        '''
        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        if classify_mode == 2:
            sgd = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
            my_model.compile(loss='binary_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy']) 
            
        if classify_mode == 3 or classify_mode == 4 or classify_mode == 5:
            sgd = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
            my_model.compile(loss='categorical_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])  
                     
        model_path = "./save_model/my_model_" + str(classify_mode) + '_' + str(step) + ".hdf5"
        model_checkpoint = ModelCheckpoint(model_path, monitor='loss',verbose=1, save_best_only=True)
        print('Fitting model...')
        my_model.fit(imgs_train, imgs_mask_train, batch_size=5, epochs=1, verbose=1, shuffle=True,validation_split=0.1, callbacks=[model_checkpoint])
        print('train==' + str(step) + '==of fintune is done')
        del imgs_train, imgs_mask_train
        gc.collect()
        return model_path

if __name__ == '__main__':
    
    # Setting GPU id
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    # path to the vgg16 model weights files.
    weights_path = './vgg16_weight/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    #top_model_weights_path = 'fc_model.h5'
    #top_model_weights_path = './bottleneck_fc_model/bottleneck_fc_model.h5'
    
    myunet = myUnet()
    img_height = 199
    img_width = 144
    nb_epoch = 5
    classify_mode = 2   #set the mode of classify
    train_data_path = "./save_data/*.npz"
    
    fname_path = myunet.read_path(train_data_path, classify_mode)
    num_file = len(fname_path)
    number = 0
    last_iter_weights = []   #one epoch is over ,save the last iteration's weights for the next epoch
    for step_epoch in range(nb_epoch):
        if step_epoch == 0:
            myunet.train_fitune(classify_mode, weights_path, fname_path[0])
        else:
            myunet.train_next(classify_mode, fname_path[0], weights_path, last_iter_weights, number)
    
        fname_path = fname_path[1:]
        for step in range(num_file - 1):
            number  = step_epoch * (num_file - 1) + step
            model_path_ = "./save_model/my_model_" + str(classify_mode) + '_' + str(number) + ".hdf5"
            number = number + 1
            model_path = myunet.train_next(classify_mode, fname_path[step], weights_path, model_path_, number)
        last_iter_weights = model_path
        
endtime = time.clock()
print("The training running time is %g s" %(endtime-starttime))