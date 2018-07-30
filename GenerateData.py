#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 19:53:05 2018

@author: ZSQ
"""

import glob
import os
import fnmatch
from PIL import Image
import SimpleITK as sitk
import numpy as np
#import pylab
import matplotlib.pyplot as plt
import cv2
import pickle
import tensorflow as tf
import gc

class BRATS(object):
    """
    flow of class working  --BRATS2015(total 252 samples)--
    1. read files
    2. crop train image and label image to 144*192(width*height)
    3. stack four slice from each modality image, then form a train sample (validation set from train set)  
    =============================================================================================
        error1: we find that the size of slice in some image from each modality at a patient is smaller
        than other, so we pick some patients from leaderboard set to substitute the test set.
        error2: the slice number of some image from each modality at a patient is different in train 
        set.
    =============================================================================================
    5. stack four slice from each modality image, then form a test sample
    """
    
    def __init__(self):
        self.train_data_path= ''   # to be used with glob package

    def set_training_path(self,path):
        self.train_data_path = path

    def get_training_path(self):
        return self.train_data_path

    def read_train_files(self):
        labels = glob.glob(self.train_data_path)  #读取文件路径
        file_names_t1 = []
        file_names_t2 = []
        file_names_t1c = []
        file_names_flair = []
        file_names_gt = []
        

        # we suppose that images are read in sequence. Afte 4 files, 5th is ground truth
        for i in range(len(labels)):
            fname = os.path.basename(labels[i])
            if fnmatch.fnmatch(fname, '*T1c.*'):
                file_names_t1c.append(labels[i])
            elif fnmatch.fnmatch(fname, '*T1.*'):
                file_names_t1.append(labels[i])
            elif fnmatch.fnmatch(fname, '*T2.*'):
                file_names_t2.append(labels[i])
            elif fnmatch.fnmatch(fname, '*Flair.*'):
                 file_names_flair.append(labels[i])
            elif fnmatch.fnmatch(fname, '*_3more*'):
                 file_names_gt.append(labels[i])

        return [file_names_t1, file_names_t1c, file_names_t2, file_names_flair, file_names_gt]
        
    def norm_image(self, img):
        img_max = img.max()
        img2 = img.astype(np.float)/img_max
        img2 = img2.astype(np.float32) 
        return img2

    
    def crop_image(self, tfiles, roi_size, class_num):  
        """

        steps:      1- read a picture
                    2- make its slices
                    3- for each slice extract ROI   144*192
                    
        """
        
        flist_4_m = [tfiles[0], tfiles[1], tfiles[2], tfiles[3]]   #for 4 modality
        
        
        mlist = tfiles[4]    #for label
                         
        num = len(flist_4_m[0])
        all_image_4_m = []
        all_label_4_m = []
        for i in range(num):    #num is the number of patients
            image_4_m = []
            
            for n in range(4):      #read 4 modality
                image = sitk.ReadImage(flist_4_m[n][i])
                image_label = sitk.ReadImage(mlist[i])
                
                img_arr = sitk.GetArrayFromImage(image)  # 176*216*176   z*y*x
                label_arr = sitk.GetArrayFromImage(image_label)
                
                slices = img_arr.shape[0]   
                #print(label_arr.shape)
                image_crop = []
                label_crop = []
                for slice_idx in range(slices):  # dimensions are changed and now first dimention gives depth
    
                    img_slice = np.array(img_arr[slice_idx, :, :])    # height*width                    
                    label_slice = np.array(label_arr[slice_idx, :, :])
                    
                    if len(np.where(label_slice > 0)[0]) == 0:
                        continue
                    
                    img_norm = self.norm_image(img_slice)                    
                                                                              
                    '''
                    img_max = img_slice.max()
                    if img_max == 0:
                        continue                    
                    img_norm = img_slice.astype(np.float)/img_max
                    img_norm = (img_norm * 255).astype(np.float32)
                    img_norm = np.uint8(img_norm)
                    '''
                    width = img_norm.shape[1]
                    height = img_norm.shape[0]
                    origin_x = np.int(np.floor(width / 2) - roi_size[0] / 2)
                    origin_y = np.int(np.floor(height / 2) - roi_size[1] / 2)                        
                    each_image_crop = img_norm[origin_y:(origin_y+roi_size[1]),origin_x:(origin_x+roi_size[0])]
                    #print(each_image_crop)
                    image_crop.append(each_image_crop)
                    each_label_crop = label_slice[origin_y:(origin_y+roi_size[1]),origin_x:(origin_x+roi_size[0])]
                    #print(each_label_crop)
                    # 2-class segmentation, we set the label to be 2 class(including 0 and 1)
                    # 2-cls : 0 and 1+2+3+4(as 1)
                    # 3-cls : 0 and 2 and 1+3+4(as 1)
                    # 4-cls : 0 and 2 and 3 and 1+4(as 1)
                    # 5-cls : 0 and 1 and 2 and 3 and 4
                    if class_num == 2:
                        each_label_crop[np.where(each_label_crop > 0)] = 1  
                    elif class_num == 3:
                        each_label_crop[np.where(each_label_crop == 1)] = 1
                        each_label_crop[np.where(each_label_crop == 3)] = 1
                        each_label_crop[np.where(each_label_crop == 4)] = 1
                    elif class_num == 4:
                        each_label_crop[np.where(each_label_crop == 1)] = 1
                        each_label_crop[np.where(each_label_crop == 4)] = 1                                         
                    
                    label_crop.append(each_label_crop)  
                    del each_image_crop, each_label_crop
                    print('cropping image    '+ str(slice_idx))                                       
                    
                image_4_m.append(image_crop)                
                            
            all_image_4_m.append(image_4_m)
            all_label_4_m.append(label_crop)     #each modality has same label
        
        return all_image_4_m,  all_label_4_m    #NO PROBLEM
            
    def one_hot_coding(self, img, class_num):
        img2 = np.reshape(img, [np.size(img), 1])
        img2 = tf.one_hot(indices = img2, depth = class_num, axis = 1)
    
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            code = sess.run(img2)
            #print(code)
 
        one_hot = np.reshape(code, [np.shape(img)[0], np.shape(img)[1], class_num])
        
        return one_hot


          
    def stack_4_modality(self, all_img_, class_num):    #stack four modality slice to a sample        
        gc.disable()
        all_img = all_img_[0]
        all_label = all_img_[1]
        train_data = []
        train_label = []
        num = len(all_img)
        for i in range(num):   #the number of patients
            image_4_m = all_img[i]
            label_4_m = all_label[i]
            each_data = []
            each_label = [] 
            each_data_slice = np.zeros([192, 144, 4]) 
            #each_label_slice = np.zeros([192, 144, 4])
            #each_label_slice = np.ones([192, 144, 4], np.int)
            each_label_slice = np.ones([192, 144], np.int)
            image_4_m_0 = image_4_m[0]
            image_4_m_1 = image_4_m[1]
            image_4_m_2 = image_4_m[2]
            image_4_m_3 = image_4_m[3]
            slice_num_0 = len(image_4_m_0)
            slice_num_1 = len(image_4_m_1)
            slice_num_2 = len(image_4_m_2)
            slice_num_3 = len(image_4_m_3)
            slice_num_min = np.min([slice_num_0,slice_num_1,slice_num_2,slice_num_3])
            for m in range(slice_num_min):
                each_data_slice[:, :, 0] = image_4_m_0[m]
                each_data_slice[:, :, 1] = image_4_m_1[m]
                each_data_slice[:, :, 2] = image_4_m_2[m]
                each_data_slice[:, :, 3] = image_4_m_3[m]
                each_data.append(each_data_slice)   #data type is float64,but the image is uint8
                '''
                each_label_slice[:, :, 0] = label_4_m[m]
                each_label_slice[:, :, 1] = label_4_m[m]
                each_label_slice[:, :, 2] = label_4_m[m]
                each_label_slice[:, :, 3] = label_4_m[m]
                each_label.append(each_label_slice.astype(np.int8))   
                '''          
                #each_label_slice = self.one_hot_coding(label_4_m[m], class_num)
                each_label_slice = label_4_m[m]
                each_label.append(each_label_slice)
                #del each_data_slice, each_label_slice
                print('stacking image   ' + str(m))
            train_data.append(each_data)
            train_label.append(each_label)
        
        #convert to ndarray
        train_all_data = []
        train_all_label = []
        for n in range(len(train_data)):
            for l in range(len(train_data[n])):
                tmp_data = train_data[n][l]
                tmp_label = train_label[n][l]
                train_all_data.append(tmp_data)
                train_all_label.append(tmp_label)
        del train_data, train_label
        gc.enable()
        return np.asarray(train_all_data), np.asarray(train_all_label)    #bug
    
    def shuffle_split_data(self,test_num, data, lab):
        #shuffle the data
        num = np.shape(lab)[0]
        index = np.arange(num)
        np.random.shuffle(index)
        data = data[index, :, :, :]
        #lab = lab[index, :, :, :]
        lab = lab[index, :, :]
        
        #split the data to trainimg and testimg
        trainimg = data[:(num-test_num), :, :, :]
        trainlab = lab[:(num-test_num), :, :]
        #trainlab = lab[:int(num * 0.9), :, :]
        testimg = data[-test_num:, :, :, :]
        testlab = lab[-test_num:, :, :]
        #testlab = lab[-int(num * 0.1):, :, :]
        return trainimg, trainlab, testimg, testlab
    
    def save_data(self, trainimg, trainlab, testimg, testlab, batchnum):
        print(">>>>>>>>saving data of batch  " + str(batchnum) + "  >>>>>>>>>>>>>>")
        
        path_traindata = "./save_data/traindata_" + str(class_num) + "cls_" + str(batchnum) + ".npz"
        path_testdata = "./save_data/testdata_" + str(class_num) + "cls_" + str(batchnum) + ".npz"
        np.savez(path_traindata, train_data = trainimg, train_label = trainlab)
        np.savez(path_testdata, test_data = testimg, test_label = testlab)
            
        print(">>>>>>>>finished save data of batch  " + str(batchnum) + "  >>>>>>>>>>>>>>")
        
    def get_train_data(self, train_path, test_num, roi_size, class_num):
        
        print("->>>>>>>>>starting to generate train set and labels>>>>>>")
        
        self.set_training_path(train_path)
        tfiles = self.read_train_files()
        num_tfiles = len(tfiles[0])
        batch_num = 10
        ratio = int(np.floor(num_tfiles/batch_num))
        s = 0
        for i in range(ratio):
            
            batch_tfiles = [tfiles[0][s:(batch_num*(i+1))],tfiles[1][s:(batch_num*(i+1))],
                            tfiles[2][s:(batch_num*(i+1))],tfiles[3][s:(batch_num*(i+1))],
                            tfiles[4][s:(batch_num*(i+1))]]
            s = batch_num*(i+1) + 1
        
            img_cropped, label_cropped = self.crop_image(batch_tfiles, roi_size, class_num)
        
            all_img_ = [img_cropped, label_cropped]
            train_data, train_label = self.stack_4_modality(all_img_, class_num)
            
            print("->>>>>>>> generating train set and labels>>>>>>>>>")
            
            trainimg, trainlab, testimg, testlab = self.shuffle_split_data(test_num, train_data, train_label)
            self.save_data(trainimg, trainlab, testimg, testlab, i)
            


if __name__ == '__main__':
    
    br = BRATS()
    global roi_size
    class_num = 2   #classify mode
    test_num = 20  #resign 20 for test
    roi_size = [144, 192]   #set image width and height for each sample
    #all data 252 samples
    br.get_train_data('./BRATS2015_Training/**/**/**/*.mha', test_num, roi_size, class_num)
   
    
    
    
   
    
    
    







