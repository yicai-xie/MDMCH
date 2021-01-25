# -*- coding: utf-8 -*-
"""
Created on Mon May 21 11:11:26 2018

@author: Administrator
"""
import numpy as np
import scipy
import scipy.io
import os

model_dir = 'G:\XLM\源码\DCMH\data\imagenet-vgg-f.mat'
data = scipy.io.loadmat(model_dir)
layers = ('conv1', 'relu1', 'norm1', 'pool1','conv2', 'relu2', 'norm2', 'pool2','conv3', 'relu3', 'conv4', 'relu4', 'conv5',
		'relu5', 'pool5','fc6', 'relu6', 'fc7', 'relu7','fc8')
weights = data['layers'][0]
mean = data['normalization'][0][0][0]