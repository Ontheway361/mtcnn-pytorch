#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import sys
import numpy as np
import pandas as pd
sys.path.append(os.getcwd())
from mtcnn.data_preprocess.utils import IoU

from IPython import embed

#root_dir  = '/Users/relu/data/benchmark_images/faceu'   # local-pc
#root_dir  = '/home/jovyan/gpu3-data2/relu/benchmark_images/faceu' # server
root_dir  = '/home/faceu' # temporary
im_dir    = os.path.join(root_dir, 'face_detection/WIDER_train/images')
part_save_dir = os.path.join(root_dir, 'train_data/train_pnet/part')
pos_save_dir  = os.path.join(root_dir, 'train_data/train_pnet/positive')
neg_save_dir  = os.path.join(root_dir, 'train_data/train_pnet/negative')

for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# store labels of positive, negative, part images
anno_dir = '/home/jovyan/gpu3-data2/lujie/data/benchmark_images/faceu/anno_store'
anno_file = os.path.join(anno_dir, 'wide_anno_train.csv')
f1 = open(os.path.join(anno_dir, 'pnet/pos_12.txt'), 'w')
f2 = open(os.path.join(anno_dir, 'pnet/neg_12.txt'), 'w')
f3 = open(os.path.join(anno_dir, 'pnet/part_12.txt'), 'w')

# anno_file: store labels of the wider face training data
# with open(anno_file, 'r') as f:
#     annotations = f.readlines()
df_anno = pd.read_csv(anno_file)
num = len(df_anno)
print("%d pics in total" % num)   # 12880

default_num_neg = 50
default_num_pop = 20  # pos and part
print_freq      = 100

p_idx, n_idx, d_idx = 0, 0, 0 # positive, negative, dont care

for idx, im_anno in df_anno.iterrows():

    # annotation = annotation.strip().split(' ')
    # img_info   = '/'.join(annotation[0].split('/')[-2:])
    im_path = os.path.join(im_dir, im_anno['img_path'])
    boxes = np.array(eval(im_anno['gt_box']), dtype=np.int32)
    img = cv2.imread(im_path)

    if (idx + 1) % print_freq == 0:
        print(idx + 1, "images done")

    height, width, channel = img.shape

    neg_num = 0
    while neg_num < default_num_neg:

        size = np.random.randint(12, min(width, height) / 2)
        nx = np.random.randint(0, width - size)
        ny = np.random.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny: ny + size, nx: nx + size, :]
        resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
            save_flag = cv2.imwrite(save_file, resized_im)
            if save_flag:
                f2.write(save_file + ' 0\n')
                n_idx += 1
                neg_num += 1

    for box in boxes:

        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if min(w, h)<=0 or max(w, h) < 20 or x1 < 0 or y1 < 0:
            continue

        # generate negative examples that have overlap with gt
        for i in range(5):

            size = np.random.randint(12, min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)

            delta_x = np.random.randint(max(-size, -x1), w)
            delta_y = np.random.randint(max(-size, -y1), h)
            nx1 = max(0, x1 + delta_x)
            ny1 = max(0, y1 + delta_y)

            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = IoU(crop_box, boxes)

            cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                save_flag = cv2.imwrite(save_file, resized_im)
                if save_flag:
                    f2.write(save_file + ' 0\n')
                    n_idx += 1

        # generate positive examples and part faces
        for i in range(default_num_pop):

            size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            try:
                delta_x = np.random.randint(-w * 0.2, w * 0.2)
                delta_y = np.random.randint(-h * 0.2, h * 0.2)
            except Exception as e:
                print(e)
                break

            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)

            cropped_im = img[int(ny1):int(ny2), int(nx1):int(nx2), :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                save_flag = cv2.imwrite(save_file, resized_im)
                if save_flag:
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                save_flag = cv2.imwrite(save_file, resized_im)
                if save_flag:
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    d_idx += 1
    print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))

f1.close()
f2.close()
f3.close()
