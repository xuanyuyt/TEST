# coding=utf-8

import os
import sys
import caffe
import cv2
import numpy as np
import matplotlib.pyplot as plt

root = 'D:/Other_Dataets/Car/'  # 根目录
deploy = root + 'deploy.prototxt'  # deploy文件
caffe_model = root + 'snapshot_iter_10000fc.caffemodel'  # 训练好的 caffemodel
dir = root + 'Ts2/'

def predictImage(filename):
    net = caffe.Classifier(deploy, caffe_model,image_dims=(96, 96),raw_scale=255, channel_swap=(2, 1, 0))
    caffe.set_mode_cpu()
    # caffe.set_mode_gpu()
    # caffe.set_device(0)
    fd = open(dir + filename)
    line = fd.readline()
    index = 0

    while line:
        print index
        line = line.strip()
        info = line.split(' ')
        imgPath = info[0]
        print imgPath

        input_image = caffe.io.load_image(dir + imgPath)
        plt.imshow(input_image)
        prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
        #print 'prediction shape:', prediction[0].shape
        plt.plot(prediction[0])
        print 'predicted class:', prediction[0].argmax()

        prediction = net.predict([input_image], oversample=False)
        #print 'prediction shape:', prediction[0].shape
        plt.plot(prediction[0])
        print 'argument predicted class:', prediction[0].argmax()
        plt.show()
        line = fd.readline()
        index = index + 1

predictImage( 'ImageName.txt')


# classification.py predictImage Test.txt