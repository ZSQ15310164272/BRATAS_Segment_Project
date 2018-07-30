#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:01:00 2018

@author: ZSQ
"""
import numpy as np
from keras.models import load_model
import os
import shutil
from skimage.io import imsave, imread
import cv2
from PIL import Image
import matplotlib.pyplot as plt



def painter(pos_label, pos_pred, path):
    fig = plt.figure()  
    #plot the label image
    ax1 = fig.add_subplot(121)  
    #set the title  
    ax1.set_title('label image')  
    #set the x axis  
    plt.xlabel('X')  
    #set the y axis  
    plt.ylabel('Y')  
    #plot scatter  
    ax1.scatter(pos_label[0], pos_label[1], c = 'w', marker = ',') 
    ax1.scatter(pos_label[2], pos_label[3], c = 'r', marker = ',') 
    ax1.scatter(pos_label[4], pos_label[5], c = 'g', marker = ',') 
    ax1.scatter(pos_label[6], pos_label[7], c = 'y', marker = ',') 
    ax1.scatter(pos_label[8], pos_label[9], c = 'b', marker = ',') 
    
    #plot the pred image
    ax1 = fig.add_subplot(122)  
    #set the title  
    ax1.set_title('predict image')  
    #set the x axis  
    plt.xlabel('X')  
    #set the y axis  
    plt.ylabel('Y')  
    #plot scatter  
    ax1.scatter(pos_pred[0], pos_pred[1], c = 'w', marker = ',') 
    ax1.scatter(pos_pred[2], pos_pred[3], c = 'r', marker = ',') 
    ax1.scatter(pos_pred[4], pos_pred[5], c = 'g', marker = ',') 
    ax1.scatter(pos_pred[6], pos_pred[7], c = 'y', marker = ',') 
    ax1.scatter(pos_pred[8], pos_pred[9], c = 'b', marker = ',') 
    #show the image  
    plt.savefig(path, format='png', dpi=300)
    #plt.show()  
    #fig.clf()

def get_position(lab):     
    #get each label's position for a label_slice 

    x0 = np.where(lab == 0)[0]
    y0 = np.where(lab == 0)[1]
    
    x1 = np.where(lab == 1)[0]
    y1 = np.where(lab == 1)[1]
    
    x2 = np.where(lab == 2)[0]
    y2 = np.where(lab == 2)[1]
    
    x3 = np.where(lab == 3)[0]
    y3 = np.where(lab == 3)[1]
    
    x4 = np.where(lab == 4)[0]
    y4 = np.where(lab == 4)[1]
    
    position = [x0, y0, x1, y1, x2, y2, x3, y3, x4, y4]
    
    return position

test_mode = 2
num_testset = '_0'
data_path = "./save_data/traindata_" + str(test_mode) + "cls" + num_testset + ".npz"
data = np.load(data_path)   #use #2901-2956 as the test samples
imgs_test = data["train_data"]
imgs_label = data["train_label"][:,:,:,np.newaxis]

# load the model
num_model = '_7'
model_path = "./save_model/my_model_" + str(test_mode) + num_model + ".hdf5"
model = load_model(model_path)
    
# predict the test set
score = model.evaluate(imgs_test, imgs_label, batch_size=1, verbose=1, sample_weight=None)
print("score: " + str(score))
imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
imgs_pred = np.argmax(imgs_mask_test, axis=-1)

np.save("./imgs_mask_test/imgs_mask_test_" + str(test_mode) + ".npy", imgs_mask_test)
print('-' * 30)
print('Saving predicted masks to files...')
print('-' * 30)

pred_dir = "preds_" + str(test_mode) + "cls"
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
else:
    shutil.rmtree(pred_dir)
    os.mkdir(pred_dir)
						
for image, image_id in zip(imgs_pred, range(len(imgs_pred))):
    position_label = get_position(imgs_label[image_id][:, :, 0])    
    position_pred = get_position(image)
    path = os.path.join(pred_dir, str(image_id) + '_result.png')
    painter(position_label, position_pred, path)
print('finished save all image')
    





