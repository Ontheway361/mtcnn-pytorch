#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import os
import cv2
import sys
import time
import numpy as np
sys.path.append(os.getcwd())
from six.moves import cPickle
import mtcnn.config as config
import mtcnn.core.vision as vision
from mtcnn.core.imagedb import ImageDB
from mtcnn.core.utils import convert_to_square,IoU
from mtcnn.core.image_reader import TestImageLoader
from mtcnn.core.detect import MtcnnDetector, create_mtcnn_net

from IPython import embed

data_dir  = '/home/faceu' # temporary
# anno_dir  = '/home/jovyan/gpu3-data2/lujie/data/benchmark_images/faceu/anno_store'
anno_dir  = '../anno_store'
pnet_file = './aku_model/pnet/pnet_epoch_20.pt'    # TODO
prefix    = ''
use_cuda  = False


def gen_rnet_data(data_dir, anno_dir, pnet_model_file, use_cuda=True):
    ''' Generate the train data of RNet with trained-PNet '''

    # load trained pnet model
    pnet, _, _     = create_mtcnn_net(p_model_path=pnet_model_file, use_cuda=use_cuda)
    mtcnn_detector = MtcnnDetector(pnet=pnet, min_face_size=12)

    # load original_anno_file, length = 12880
    anno_file = os.path.join(anno_dir, 'local_wide_anno.txt')  # TODO :: [local_wide_anno, wide_anno_train]
    imagedb = ImageDB(anno_file, mode='test', prefix_path='')
    imdb = imagedb.load_imdb()

    image_reader = TestImageLoader(imdb, 1, False)
    print('size:%d' %image_reader.size)

    batch_idx, all_boxes = 0, list()


    for databatch in image_reader:

        if (batch_idx + 1) % 100 == 0:
            print ("%d images done" % (batch_idx + 1))
        im = databatch

        # obtain boxes and aligned boxes
        boxes, boxes_align = mtcnn_detector.detect_pnet(im=im)  # Time costly

        if boxes_align is None:
            all_boxes.append(np.array([]))
            batch_idx += 1
            continue

        # if vis:
        #     rgb_im = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
        #     vision.vis_two(rgb_im, boxes[:100, :], boxes_align[:100, :])

        all_boxes.append(boxes_align)
        batch_idx += 1

    save_path = os.path.join(anno_dir, 'rnet')

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file = os.path.join(save_path, "detections_%d.pkl" % int(time.time()))
    with open(save_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    # gen_rnet_sample_data(data_dir, anno_dir, save_file)


def gen_rnet_sample_data(data_dir, anno_dir, det_boxs_file):
    ''' Generate the train data for RNet '''


    part_save_dir = os.path.join(data_dir, "train_rnet/part")
    pos_save_dir  = os.path.join(data_dir, "train_rnet/positive")
    neg_save_dir  = os.path.join(data_dir, "train_rnet/negative")

    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image
    anno_file = os.path.join(anno_dir, 'local_wide_anno.txt')  # TODO :: [local_wide_anno, wide_anno_train]
    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    net, image_size = "rnet", 24
    num_of_images = len(annotations)
    im_idx_list, gt_boxes_list = [], []
    print ("processing %d images in total" % num_of_images)

    for annotation in annotations:

        annotation = annotation.strip().split(' ')
        im_idx = os.path.join('', annotation[0])
        # im_idx = annotation[0]

        boxes = list(map(float, annotation[1:]))
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)

    save_path = os.path.join(anno_dir, 'rnet')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    f1 = open(os.path.join(save_path, 'pos_%d.txt' % image_size), 'w')
    f2 = open(os.path.join(save_path, 'neg_%d.txt' % image_size), 'w')
    f3 = open(os.path.join(save_path, 'part_%d.txt'% image_size), 'w')

    # print(det_boxs_file)
    det_handle = open(det_boxs_file, 'rb')

    det_boxes = cPickle.load(det_handle)

    print(len(det_boxes), num_of_images)

    n_idx, p_idx, d_idx = 0, 0, 0
    image_done = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):

        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        # change to square
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0

        for box in dets:

            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width  = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or \
               (x_right > img.shape[1] - 1) or (y_bottom > img.shape[0] - 1):
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            # Iou with all gts must below 0.3
            if np.max(Iou) < 0.3 and neg_num < 60:
                # save the examples
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                # print(save_file)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    f1.close()
    f2.close()
    f3.close()


if __name__ == '__main__':

    gen_rnet_data(data_dir, anno_dir, pnet_file, use_cuda, )
