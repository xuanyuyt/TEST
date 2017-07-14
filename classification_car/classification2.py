# coding=utf-8

import os
import sys
import caffe
import cv2
import numpy as np

root = 'D:/Other_Dataets/Car/'  # 根目录
deploy = root + 'deploy2.prototxt'  # deploy文件
caffe_model = root + '_iter_10000.caffemodel'  # 训练好的 caffemodel
dir = root + 'Ts2/'


# img=root+'data/DRIVE/test/60337.jpg'   #随机找的一张待测图片

def predictImage(filename):
    net = caffe.Net(deploy, caffe_model, caffe.TEST)  # 加载model和network
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

        colorImage = cv2.imread(dir + imgPath)

        # 执行测试
        data4DL = np.zeros([[1,1,1,1], 1, 1, 1])
        net.set_input_arrays(colorImage.astype(np.float32), data4DL.astype(np.float32))
        out = net.forward()

        labels = np.loadtxt(labels_kinds, str, delimiter='\t')  # 读取类别名称文件
        prob = net.blobs['ip1'].data[0].flatten()  # 取出最后一层（prob）属于某个类别的概率值，并打印,'prob'为最后一层的名称
        print prob
        order = prob.argsort()[-1]  # 将概率值排序，取出最大值所在的序号 ,9指的是分为0-9十类
        # argsort()函数是从小到大排列
        print 'the class is:', labels[order]  # 将该序号转换成对应的类别名称，并打印
        f = file(dir + "Predict.txt", "a+")
        f.writelines(imgPath + ' ' + labels[order] + '\n')
        line = fd.readline()
        index = index + 1

labels_kinds = dir + 'LabelKinds.txt'  # 类别名称文件，将数字标签转换回类别名称


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        func = globals()[sys.argv[1]]
        func(*sys.argv[2:])


# classification.py predictImage ImageName.txt