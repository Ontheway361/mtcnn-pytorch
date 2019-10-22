#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import sys
import time
import random
import argparse
import numpy as np
sys.path.append(os.getcwd())
import mtcnn.core.utils as utils

root_dir  = '/home/faceu/'

from IPython import embed

def gen_data(args):

    image_id = 0
    if args.img_size == 12:
        folder_name = 'pnet'
    elif args.img_size == 24:
        folder_name = 'rnet'
    elif args.img_size == 48:
        folder_name = 'onet'
    else:
        raise TypeError('img_size must be 12, 24 or 48')

    txt_save_dir = os.path.join(args.save_dir, 'anno_store/%s' % folder_name)
    img_save_dir = os.path.join(args.save_dir, '%s/landmark' % folder_name)
    for folder in [txt_save_dir, img_save_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    save_txt_name = 'landmark_%d.txt' % args.img_size
    save_txt_to   = os.path.join(txt_save_dir, save_txt_name)

    with open(args.anno_file, 'r') as f2:
        annotations = f2.readlines()
    f2.close()

    num = len(annotations)
    print("%d total images" % num)

    l_idx =0
    f = open(save_txt_to, 'w')
    for idx, annotation in enumerate(annotations):

        annotation = '/'.join(annotation.strip().split('\\')).split(' ')

        assert len(annotation)==15, "each line should have 15 element"

        im_path  = os.path.join(args.data_dir, annotation[0])
        gt_box   = np.array(list(map(float, annotation[1:5])), dtype=np.int32)
        landmark = np.array(list(map(float, annotation[5:])), dtype=np.float)

        img = cv2.imread(im_path)

        assert (img is not None)

        height, width, channel = img.shape

        if (idx + 1) % 100 == 0:
            print("%d images done, landmark images: %d" % (idx+1, l_idx))

        x1, x2, y1, y2 = gt_box # NOTE :: box [x1, x2, y1, y2]
        gt_box[1], gt_box[2] = y1, x2

        box_w, box_h = x2 - x1 + 1, y2 - y1 + 1
        if max(box_w, box_h) < 40 or x1 < 0 or y1 < 0:
            continue

        for i in range(args.num_rands):

            bbox_size = np.random.randint(int(min(box_w, box_h) * 0.8), np.ceil(1.25 * max(box_w, box_h)))
            delta_x   = np.random.randint(-box_w * 0.2, box_w * 0.2)
            delta_y   = np.random.randint(-box_h * 0.2, box_h * 0.2)
            nx1 = max(x1 + box_w / 2 - bbox_size / 2 + delta_x, 0)
            ny1 = max(y1 + box_h / 2 - bbox_size / 2 + delta_y, 0)

            nx2 = nx1 + bbox_size
            ny2 = ny1 + bbox_size

            if nx2 > width or ny2 > height:
                continue

            crop_box   = np.array([nx1, ny1, nx2, ny2])
            cropped_im = img[int(ny1):int(ny2) + 1, int(nx1):int(nx2) + 1, :]
            resized_im = cv2.resize(cropped_im, (args.img_size, args.img_size),interpolation=cv2.INTER_LINEAR)

            offset_x1 = (x1 - nx1) / float(bbox_size)
            offset_y1 = (y1 - ny1) / float(bbox_size)
            offset_x2 = (x2 - nx2) / float(bbox_size)
            offset_y2 = (y2 - ny2) / float(bbox_size)

            offset_left_eye_x = (landmark[0] - nx1) / float(bbox_size)
            offset_left_eye_y = (landmark[1] - ny1) / float(bbox_size)

            offset_right_eye_x = (landmark[2] - nx1) / float(bbox_size)
            offset_right_eye_y = (landmark[3] - ny1) / float(bbox_size)

            offset_nose_x = (landmark[4] - nx1) / float(bbox_size)
            offset_nose_y = (landmark[5] - ny1) / float(bbox_size)

            offset_left_mouth_x = (landmark[6] - nx1) / float(bbox_size)
            offset_left_mouth_y = (landmark[7] - ny1) / float(bbox_size)

            offset_right_mouth_x = (landmark[8] - nx1) / float(bbox_size)
            offset_right_mouth_y = (landmark[9] - ny1) / float(bbox_size)

            iou = utils.IoU(crop_box.astype(np.float), np.expand_dims(gt_box.astype(np.float), 0))

            if iou > args.threshold:

                save_file = os.path.join(img_save_dir, "%s.jpg" % l_idx)
                cv2.imwrite(save_file, resized_im)

                f.write(save_file + ' -2 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n' % \
                (offset_x1, offset_y1, offset_x2, offset_y2, offset_left_eye_x, offset_left_eye_y, \
                 offset_right_eye_x, offset_right_eye_y, offset_nose_x, offset_nose_y, \
                 offset_left_mouth_x, offset_left_mouth_y, offset_right_mouth_x, offset_right_mouth_y))
                l_idx += 1

    f.close()


def gen_config():

    parser = argparse.ArgumentParser(description=' Generate lmk file')

    parser.add_argument('--anno_file', type=str,  default=os.path.join(root_dir, 'cuhk_mm/anno_file.txt'))
    parser.add_argument('--data_dir',  type=str,  default=os.path.join(root_dir, 'cuhk_mm'))
    parser.add_argument('--save_dir',  type=str,  default=os.path.join(root_dir, '5keypoints'))
    parser.add_argument('--img_size',  type=int,  default=48)    # TODO
    parser.add_argument('--num_rands', type=int,  default=10)    # TODO
    parser.add_argument('--threshold', type=float,default=0.65)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    gen_data(gen_config())
