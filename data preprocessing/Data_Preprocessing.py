# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:41:16 2020

@author: Zihan Yang
"""


import os
import glob



import sys 
sys.path.append("..") 
# import cfg
import random
#F:/yzh/data process results/DL/PreprocessDataset/SCC/train/1.tiff
# BASE = "D:/NeckTissue/dataset/"
BASE = "F:/yzh/data_process_results/DL/ImageCrop/SCC/step32-16Crop_SCC_1/"

if __name__ == '__main__':
    traindata_path = BASE + 'train'
    labels = os.listdir(traindata_path)
    
    valdata_path = BASE + 'val'
    vallabels = os.listdir(valdata_path)
    
    testdata_path = BASE + 'test'
    testlabels = os.listdir(testdata_path)
    
    #
    txtpath = BASE
    
    # print(labels)
    for index, label in enumerate(labels):
        print(label)
        print(index)
        trainsubfilename = os.listdir(os.path.join(traindata_path,label))
        for index1, element in enumerate(trainsubfilename):
            imglist = glob.glob(os.path.join(traindata_path,label,element, '*.tiff'))
            # print(imglist)
            random.shuffle(imglist)
            # print(len(imglist))
            # trainlist = imglist[:int(0.8*len(imglist))]
            trainlist = imglist
            with open(txtpath + 'train.txt', 'a')as f:
                for img in trainlist:
                    # print(img + ' ' + str(index))
                    f.write(img + ' ' + str(index))
                    f.write('\n')
                
    for index, label in enumerate(vallabels):
        valsubfilename = os.listdir(os.path.join(valdata_path,label))
        for index1, element in enumerate(valsubfilename):
            imglist = glob.glob(os.path.join(valdata_path,label,element, '*.tiff'))
            #imglist = glob.glob(os.path.join(valdata_path,label, '*.tiff'))
            # print(imglist)
            random.shuffle(imglist)
            # print(len(imglist))
            # trainlist = imglist[:int(0.8*len(imglist))]
            trainlist = imglist
            with open(txtpath + 'val.txt', 'a')as f:
                for img in trainlist:
                    # print(img + ' ' + str(index))
                    f.write(img + ' ' + str(index))
                    f.write('\n')
                
    for index, label in enumerate(testlabels):
        testsubfilename = os.listdir(os.path.join(testdata_path,label))
        for index1, element in enumerate(testsubfilename):
            imglist = glob.glob(os.path.join(testdata_path,label,element, '*.tiff'))
            # imglist = glob.glob(os.path.join(testdata_path,label, '*.tiff'))
            # print(imglist)
            random.shuffle(imglist)
            # print(len(imglist))
            # trainlist = imglist[:int(0.8*len(imglist))]
            trainlist = imglist
            with open(txtpath + 'test.txt', 'a')as f:
                for img in trainlist:
                    # print(img + ' ' + str(index))
                    f.write(img + ' ' + str(index))
                    f.write('\n')



