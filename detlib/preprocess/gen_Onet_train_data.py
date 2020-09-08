#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import time
import argparse
import numpy as np
sys.path.append(os.getcwd())
from six.moves import cPickle
from detlib.detector.imagedb import ImageDB
from detlib.detector.utils import convert_to_square,IoU
from detlib.detector.image_reader import TestImageLoader
from detlib.detector.detect import MtcnnDetector,create_mtcnn_net


data_dir  = '/home/faceu'
anno_dir  = os.path.join(data_dir, '5keypoints')
pnet_file = './model/checkout/pnet/pnet_epoch_20.pt'
rnet_file = './model/checkout/rnet/rnet_epoch_14.pt'
use_cuda  = True


def gen_onet_data(data_dir, anno_dir, pnet_file, rnet_file, use_cuda=True, vis=False):


    pnet, rnet, _  = create_mtcnn_net(p_model_path=pnet_file, r_model_path=rnet_file, use_cuda=use_cuda)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, min_face_size=12)
    
    anno_file = os.path.join(anno_dir, 'anno_store/wide_anno_train.txt')
    imagedb = ImageDB(anno_file, mode='test', prefix_path='')
    imdb    = imagedb.load_imdb()
    image_reader = TestImageLoader(imdb, 1, False)
    print('size:%d' % image_reader.size) # still use wideface dataset

    all_boxes, batch_idx = list(), 0
    for databatch in image_reader:

        if (batch_idx + 1) % 100 == 0:
            print("%d images done" % (batch_idx + 1))

        im = databatch

        # pnet detection = [x1, y1, x2, y2, score, reg]
        p_boxes_align = mtcnn_detector.detect_pnet(im=im)

        # rnet detection
        r_boxes_align = mtcnn_detector.detect_rnet(im=im, dets=p_boxes_align)

        if r_boxes_align is None:
            all_boxes.append(np.array([]))
            batch_idx += 1
            continue

        all_boxes.append(r_boxes_align)
        batch_idx += 1

    save_path = os.path.join(anno_dir, 'onet')

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file = os.path.join(save_path, "detections_%d.pkl" % int(time.time()))
    with open(save_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
    
    # save_file = '/home/faceu/5keypoints/onet/detections_1571576511.pkl'
    gen_onet_sample_data(data_dir, anno_dir, save_file)


def gen_onet_sample_data(data_dir, anno_dir, det_boxs_file):

    neg_save_dir  = os.path.join(anno_dir, "onet/negative")
    pos_save_dir  = os.path.join(anno_dir, "onet/positive")
    part_save_dir = os.path.join(anno_dir, "onet/part")

    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image
    
    anno_file = os.path.join(anno_dir, 'anno_store/wide_anno_train.txt')
    
    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    image_size = 48
    net = "onet"

    im_idx_list = list()
    gt_boxes_list = list()
    num_of_images = len(annotations)
    print("processing %d images in total" % num_of_images)

    for annotation in annotations:

        annotation = annotation.strip().split(' ')
        im_idx = os.path.join('',annotation[0])

        boxes = list(map(float, annotation[1:]))
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)

    save_path = os.path.join(anno_dir, 'anno_store/onet')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    f1 = open(os.path.join(save_path, 'pos_%d.txt' % image_size), 'w')
    f2 = open(os.path.join(save_path, 'neg_%d.txt' % image_size), 'w')
    f3 = open(os.path.join(save_path, 'part_%d.txt' % image_size), 'w')

    det_handle = open(det_boxs_file, 'rb')

    det_boxes = cPickle.load(det_handle)
    print(len(det_boxes), num_of_images)
    # assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        for box in dets:
            x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
            width = x_right - x_left + 1
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
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
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


def model_store_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/model_store"


if __name__ == '__main__':

    gen_onet_data(data_dir, anno_dir, pnet_file, rnet_file, use_cuda)
