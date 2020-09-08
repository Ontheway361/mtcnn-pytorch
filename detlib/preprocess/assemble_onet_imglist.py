#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.getcwd())
import detlib.preprocess.assemble as assemble


# anno_dir = '/home/jovyan/gpu3-data2/lujie/data/benchmark_images/faceu/anno_store'
anno_dir = '/home/faceu/5keypoints/anno_store'

onet_postive_file  = os.path.join(anno_dir, 'onet/pos_48.txt')
onet_part_file     = os.path.join(anno_dir, 'onet/part_48.txt')
onet_neg_file      = os.path.join(anno_dir, 'onet/neg_48.txt')
onet_landmark_file = os.path.join(anno_dir, 'onet/landmark_48.txt')  # TODO

train_file  = os.path.join(anno_dir, 'onet/train_anno_48.txt')
eval_file   = os.path.join(anno_dir, 'onet/eval_anno_48.txt')



if __name__ == '__main__':

    anno_list = []

    anno_list.append(onet_postive_file)
    anno_list.append(onet_part_file)
    anno_list.append(onet_neg_file)
    anno_list.append(onet_landmark_file)

    assemble.assemble_data(train_file, eval_file, anno_list)
