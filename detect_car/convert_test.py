# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from  math import pow
from skimage import transform as tf
import cv2
import caffe
import pylab
#from nms import nms_average, nms_max
#import render_result

caffe.set_mode_gpu()
caffe.set_device(0)
# caffe.set_mode_cpu()


model_path = './'
model_define = model_path + 'deploy.prototxt'
model_weight = model_path + 'snapshot_iter_10000.caffemodel'
model_define_fc = model_path + 'deploy_fc.prototxt'
model_weight_fc = model_path + 'snapshot_iter_10000_fc.caffemodel'


params =  ['ip1']
params_fc = ['fc6-conv']




def convert_full_conv(model_define, model_weight, model_define_fc, model_weight_fc):
    '''
    @breif: 将原始网络转换为全卷积模型
    @param: model_define,二分类网络定义文件
    @param: model_weight，二分类网络训练好的参数
    @param: model_define_fc,生成的全卷积网络定义文件
    @param: model_weight_fc，转化好的全卷积网络的参数
    '''
    net = caffe.Net(model_define, model_weight, caffe.TEST)
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
    for fc in params:
        print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape,
                                                                                   fc_params[fc][1].shape)
    net_fc = caffe.Net(model_define_fc, model_weight, caffe.TEST)
    conv_params = {pr: (net_fc.params[pr][0].data, net_fc.params[pr][1].data) for pr in params_fc}
    for conv in params_fc:
        print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape,
                                                                                   conv_params[conv][1].shape)
    for pr, pr_conv in zip(params, params_fc):
        conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
        conv_params[pr_conv][1][...] = fc_params[pr][1]
    net_fc.save(model_weight_fc)
    print 'convert done!'
    return net_fc


def face_detection_image(net, image_name):
    img_count = 0
    im = caffe.io.load_image(image_name.strip())
    # mean_blob = caffe.proto.caffe_pb2.BlobProto()
    # mean_blob.ParseFromString(open('C:/caffe-master2/data/ilsvrc12/imagenet_mean.binaryproto', 'rb').read())
    # mean_npy = caffe.io.blobproto_to_array(mean_blob)
    # a = mean_npy[0, :, 0, 0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # transformer.set_mean('data', a)
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)
    # make classification map by forward and print prediction indices at each location
    out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
    print out['prob'][0, 1]
    # print out['prob'][0].argmax(axis=0)
    # show net input and confidence map (probability of the top prediction at each location)
    plt.subplot(1, 2, 1)
    plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
    plt.subplot(1, 2, 2)
    plt.imshow(out['prob'][0, 1])
    img_count += 1
    pylab.show()


if __name__ == "__main__":
    if not os.path.isfile(model_weight_fc):
        net_fc = convert_full_conv(model_define, model_weight, model_define_fc, model_weight_fc)
    else:
        net_fc = caffe.Net(model_define_fc, model_weight_fc, caffe.TEST)

    imgList = "D:/Other_Dataets/Car/Ts2/ImageName.txt"  # 10001neg
    for imgFile in open(imgList).readlines():  # 对于每个测试图片
        face_detection_image(net_fc, 'D:/Other_Dataets/Car/Ts2/'+ imgFile)

    # filename = "10001.jpg" # 10001neg
    # face_detection_image(net_fc, filename)

    plt.close('all')
