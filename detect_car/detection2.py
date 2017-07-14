# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import PIL
import operator
from  math import pow
from time import clock
from PIL import Image, ImageDraw, ImageFont

import caffe

caffe.set_mode_gpu()
caffe.set_device(0)
#caffe.set_mode_cpu()

model_path = 'model/'
model_define = model_path + 'deploy.prototxt'
model_weight = model_path + 'snapshot_21_iter_10000.caffemodel'
model_define_fc = model_path + 'deploy_fc.prototxt'
model_weight_fc = model_path + 'snapshot_21_iter_10000_fc.caffemodel'

# helper show filter outputs
def show_filters(net):
    net.forward()
    plt.figure()
    filt_min, filt_max = net.blobs['conv'].data.min(), net.blobs['conv'].data.max()
    for i in range(3):  # three feature map.
        plt.subplot(1, 4, i + 2)
        plt.title("filter #{} output".format(i))
        plt.imshow(net.blobs['conv'].data[0, i], vmin=filt_min, vmax=filt_max)
        plt.tight_layout()
        plt.axis('off')
        plt.show()


def generateBoundingBox(featureMap, scale):
    boundingBox = []
    stride = 16
    cellSize = 96
    # 227 x 227 cell, stride=32
    for (x, y), prob in np.ndenumerate(featureMap):
        if (prob >= 0.99):
            boundingBox.append(
                [int((stride * y) / scale), int((x * stride) / scale), int((stride * y + cellSize - 1) / scale),
                 int((stride * x + cellSize - 1) / scale), prob])
    # sort by prob, from max to min.
    # boxes = np.array(boundingBox)
    return boundingBox


def nms_average(boxes, overlapThresh=0.2):
    result_boxes = []
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(boxes[:, 4])

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # area of i.
        area_i = np.maximum(0, x2[i] - x1[i] + 1) * np.maximum(0, y2[i] - y1[i] + 1)
        area_array = np.zeros(len(idxs) - 1)
        area_array.fill(area_i)
        # compute the ratio of overlap
        # overlap = (w * h) / (area[idxs[:last]]  - w * h + area_array)

        overlap = (w * h) / (area[idxs[:last]])
        delete_idxs = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        xmin = 10000
        ymin = 10000
        xmax = 0
        ymax = 0
        ave_prob = 0
        width = x2[i] - x1[i] + 1
        height = y2[i] - y1[i] + 1
        for idx in delete_idxs:
            ave_prob += boxes[idxs[idx]][4]
            if (boxes[idxs[idx]][0] < xmin):
                xmin = boxes[idxs[idx]][0]
            if (boxes[idxs[idx]][1] < ymin):
                ymin = boxes[idxs[idx]][1]
            if (boxes[idxs[idx]][2] > xmax):
                xmax = boxes[idxs[idx]][2]
            if (boxes[idxs[idx]][3] > ymax):
                ymax = boxes[idxs[idx]][3]
        if (x1[i] - xmin > 0.1 * width):
            xmin = x1[i] - 0.1 * width
        if (y1[i] - ymin > 0.1 * height):
            ymin = y1[i] - 0.1 * height
        if (xmax - x2[i] > 0.1 * width):
            xmax = x2[i] + 0.1 * width
        if (ymax - y2[i] > 0.1 * height):
            ymax = y2[i] + 0.1 * height
        result_boxes.append([xmin, ymin, xmax, ymax, ave_prob / len(delete_idxs)])
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, delete_idxs)

    # return only the bounding boxes that were picked using the
    # integer data type
    # result = np.delete(boxes[pick],np.where(boxes[pick][:, 4] < 0.9)[0],  axis=0)
    # print boxes[pick]
    return result_boxes


def nms_max(boxes, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in xrange(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

        # return only the bounding boxes that were picked
    return boxes[pick]


def convert_full_conv():  # 转换成全卷积模型
    # Load the original network and extract the fully connected layers' parameters.
    net = caffe.Net(model_define, model_weight, caffe.TEST)
    params = ['ip1']
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
    # Load the fully convolutional network to transplant the parameters.
    net_full_conv = caffe.Net(model_define_fc, model_weight, caffe.TEST)
    params_full_conv = ['fc6-conv']
    conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}
    for pr, pr_conv in zip(params, params_full_conv):
        conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
        conv_params[pr_conv][1][...] = fc_params[pr][1]
    net_full_conv.save(model_weight_fc)


def face_detection(imgList):
    img_count = 1
    for imgFile in open(imgList).readlines():  # 对于每个测试图片
        scales = []
        factor = 0.793700526
        img = Image.open(imgFile.strip())  # 读取图片
        min_range = 0
        max_range = 0
        if (img.size[0] > img.size[1]):
            min_range = img.size[1]
            max_range = img.size[0]
        else:
            min_range = img.size[0]
            max_range = img.size[1]

        # delim = 1600 / max_range
        # if (delim == 1):
        #     scales.append(1)
        # elif (delim > 1):
        #     scales.append(delim)

        scales.append(pow(factor, 8))
        # min_range = min_range * factor * 0.16
        # factor_count = 1
        # while (min_range >= 144):
        #     scales.append(pow(factor, factor_count)* 0.16)
        #     min_range = min_range * factor
        #     factor_count += 1
        total_boxes = []
        print 'size:', img.size[0], img.size[1]
        print scales
        for scale in scales:
            # resize image
            scale_img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
            scale_img.save("tmp.jpg")
            # print 'size:', scale_img.size[0], scale_img.size[1]
            # modify the full_conv prototxt.
            prototxt = open(model_define_fc, 'r')
            new_line = ""
            for i, line in enumerate(prototxt):
                if i == 5:
                    new_line += "input_dim: " + str(scale_img.size[1]) + "\n"
                elif i == 6:
                    new_line += "input_dim: " + str(scale_img.size[0]) + "\n"
                else:
                    new_line += line
            output = open('deploy_fc2.prototxt', 'w')
            output.write(new_line)
            output.close()
            prototxt.close()
            net_full_conv = caffe.Net('deploy_fc2.prototxt', model_weight_fc, caffe.TEST)
            # load input and configure preprocessing
            im = caffe.io.load_image("tmp.jpg")
            transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
            #transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
            transformer.set_transpose('data', (2, 0, 1))
            transformer.set_channel_swap('data', (2, 1, 0))
            transformer.set_raw_scale('data', 255.0)

            # make classification map by forward and print prediction indices at each location
            out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
            # print out['prob'][0, 1]
            # print out['prob'][0].argmax(axis=0)
            boxes = generateBoundingBox(out['prob'][0, 1], scale)
            # plt.subplot(1, 2, 1)
            # plt.imshow(transformer.deprocess('data', net_full_conv.blobs['data'].data[0]))
            # plt.subplot(1, 2, 2)
            # plt.imshow(out['prob'][0,1])
            # plt.show()
            if (boxes):
                total_boxes.extend(boxes)

        # nms
        boxes_nms = np.array(total_boxes)
        true_boxes1 = nms_max(boxes_nms, overlapThresh=0.3)
        true_boxes = nms_average(np.array(true_boxes1), overlapThresh=0.07)
        # # display the nmx bounding box in  image.
        # draw = ImageDraw.Draw(img)
        # print "width:", img.size[0], "height:", img.size[1]
        # for box in true_boxes:
        #     draw.rectangle((box[0], box[1], box[2], box[3]), outline=(255, 0, 0))
        #     # font_path=os.environ.get("FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
        #     # ttFont = ImageFont.truetype(font_path, 20)
        #     draw.text((box[0], box[1]), "{0:.2f}".format(box[4]))
        # img.show()
        # img.save("result/" + str(img_count) + ".jpg")
        # plt.subplots(1)
        colorImage = cv2.imread(imgFile.strip())
        num = 0
        for box in true_boxes: # true_boxes
            # cropImg = colorImage[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            # cropImg = cv2.resize(cropImg, (96, 96), interpolation=cv2.INTER_LINEAR)
            #
            # cv2.imwrite('E:/tyang/Car/Tr21/Neg22'+ str(img_count) +'_' + str(num) + '.jpg', cropImg)

            cv2.rectangle(colorImage, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            num += 1

        cv2.namedWindow('detect')

        cv2.imshow('detect',colorImage)
        cv2.waitKey(0)
        img_count += 1


if __name__ == "__main__":
    start = clock()
    # convert_full_conv()
    face_detection("D:/Other_Dataets/Car/Ts1/part5/Path_Images.txt")
    finish = clock()
    print "耗时： "
    print (finish - start)