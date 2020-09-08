#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.getcwd())
import detlib.preprocess.assemble as assemble

# anno_dir = '/home/jovyan/gpu3-data2/lujie/data/benchmark_images/faceu/anno_store'
anno_dir = '/home/faceu/5keypoints/anno_store'

rnet_postive_file  = os.path.join(anno_dir, 'rnet/pos_24.txt')
rnet_part_file     = os.path.join(anno_dir, 'rnet/part_24.txt')
rnet_neg_file      = os.path.join(anno_dir, 'rnet/neg_24.txt')
rnet_landmark_file = os.path.join(anno_dir, 'rnet/landmark_24.txt')  # TODO

train_file  = os.path.join(anno_dir, 'rnet/train_anno_24.txt')
eval_file   = os.path.join(anno_dir, 'rnet/eval_anno_24.txt')


if __name__ == '__main__':

    anno_list = []

    anno_list.append(rnet_postive_file)
    anno_list.append(rnet_part_file)
    anno_list.append(rnet_neg_file)
    anno_list.append(rnet_landmark_file)

    chose_count = assemble.assemble_data(train_file, eval_file, anno_list)
