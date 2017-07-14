# coding=utf-8

import os
import sys
import caffe
import cv2
import numpy as np

root = 'D:/Other_Dataets/Car/'  # 根目录
deploy = 'deploy.prototxt'  # deploy文件
caffe_model = 'snapshot_iter_10000.caffemodel'  # 训练好的 caffemodel
dir = root + 'Ts2/'
labels_kinds = dir + 'LabelKinds.txt'  # 类别名称文件，将数字标签转换回类别名称

# img=root+'data/DRIVE/test/60337.jpg'   #随机找的一张待测图片

def predictImage(filename):
    net = caffe.Net(deploy, caffe_model, caffe.TEST)  # 加载model和network
    #caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device(0)
    fd = open(dir + filename)
    line = fd.readline()
    index = 0
    rightPredict = 0

    while line:
        print index
        line = line.strip()
        info = line.split(' ')
        imgPath = info[0]
        groundtruth = info[1]
        print imgPath

        # 图片预处理设置
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  # 设定图片的shape格式(1,3,28,28)
        transformer.set_transpose('data', (2, 0, 1))  # 改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
        # transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用
        transformer.set_raw_scale('data', 255)  # 缩放到【0，255】之间
        transformer.set_channel_swap('data', (2, 1, 0))  # 交换通道，将图片由RGB变为BGR

        im = caffe.io.load_image(dir + imgPath)  # 加载图片
        net.blobs['data'].data[...] = transformer.preprocess('data', im)  # 执行上面设置的图片预处理操作，并将图片载入到blob中

        # 执行测试
        out = net.forward()

        labels = np.loadtxt(labels_kinds, str, delimiter='\t')  # 读取类别名称文件
        prob = net.blobs['prob'].data[0].flatten()  # 取出最后一层（prob）属于某个类别的概率值，并打印,'prob'为最后一层的名称
        print prob
        order = prob.argsort()[-1]  # 将概率值排序，取出最大值所在的序号 ,9指的是分为0-9十类
        # argsort()函数是从小到大排列
        print 'the class is:', labels[order]  # 将该序号转换成对应的类别名称，并打印
        if labels[order] == groundtruth:
            rightPredict = rightPredict + 1
        #f = file(dir + "Predict.txt", "a+")
        #f.writelines(imgPath + ' ' + labels[order] + '\n')
        line = fd.readline()
        index = index + 1
    print 'Accuracy',(float(rightPredict) / index), "%"

def predictSimpleImage(filename):
    net = caffe.Net(deploy, caffe_model, caffe.TEST)  # 加载model和network
    caffe.set_mode_cpu()
    # caffe.set_mode_gpu()
    # caffe.set_device(0)

    # 图片预处理设置
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  # 设定图片的shape格式(1,3,28,28)
    transformer.set_transpose('data', (2, 0, 1))  # 改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
    # transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用
    transformer.set_raw_scale('data', 255)  # 缩放到【0，255】之间
    transformer.set_channel_swap('data', (2, 1, 0))  # 交换通道，将图片由RGB变为BGR

    net.blobs['data'].reshape(1, 3, 96, 96)
    im = caffe.io.load_image(filename)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)  # 执行上面设置的图片预处理操作，并将图片载入到blob中

    # 执行测试
    out = net.forward()

    labels = np.loadtxt(labels_kinds, str, delimiter='\t')  # 读取类别名称文件
    prob = net.blobs['prob'].data[0].flatten()  # 取出最后一层（prob）属于某个类别的概率值，并打印,'prob'为最后一层的名称
    print prob
    order = prob.argsort()[-1]  # 将概率值排序，取出最大值所在的序号 ,9指的是分为0-9十类
    # argsort()函数是从小到大排列
    print 'the class is:', labels[order]  # 将该序号转换成对应的类别名称，并打印


predictSimpleImage('10001neg.jpg')

#  predictImage('ImageName.txt')
#  predictSimpleImage('10001.jpg')