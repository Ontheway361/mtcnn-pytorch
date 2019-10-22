#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.getcwd())
import mtcnn.data_preprocess.assemble as assemble

anno_dir = '/home/jovyan/gpu3-data2/lujie/data/benchmark_images/faceu/anno_store'

pnet_postive_file  = os.path.join(anno_dir, 'pnet/pos_12.txt')
pnet_part_file     = os.path.join(anno_dir, 'pnet/part_12.txt')
pnet_neg_file      = os.path.join(anno_dir, 'pnet/neg_12.txt')
pnet_landmark_file = os.path.join(anno_dir, 'pnet/landmark_12.txt')

train_file  = os.path.join(anno_dir, 'pnet/train_anno_12.txt')
eval_file   = os.path.join(anno_dir, 'pnet/eval_anno_12.txt')

if __name__ == '__main__':

    anno_list = []

    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)
    anno_list.append(pnet_landmark_file)

    chose_count = assemble.assemble_data(train_file, eval_file, anno_list)
