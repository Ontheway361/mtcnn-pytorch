#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.getcwd())
import mtcnn.data_preprocess.assemble as assemble


anno_dir = '/home/jovyan/gpu3-data2/lujie/data/benchmark_images/faceu/anno_store'

pnet_postive_file  = os.path.join(anno_dir, 'onet/pos_48.txt')
pnet_part_file     = os.path.join(anno_dir, 'onet/part_48.txt')
pnet_neg_file      = os.path.join(anno_dir, 'onet/neg_48.txt')
pnet_landmark_file = os.path.join(anno_dir, 'onet/landmark_48.txt')  # TODO

train_file  = os.path.join(anno_dir, 'onet/train_anno_12.txt')
eval_file   = os.path.join(anno_dir, 'onet/eval_anno_12.txt')



if __name__ == '__main__':

    anno_list = []

    anno_list.append(onet_postive_file)
    anno_list.append(onet_part_file)
    anno_list.append(onet_neg_file)
    anno_list.append(onet_lmktrain_file)

    assemble.assemble_data(train_file, eval_file, anno_list)
