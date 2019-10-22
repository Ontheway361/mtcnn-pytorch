#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import numpy.random as npr

#assemble the pos, neg, part annotations to one file
def assemble_data(train_file, eval_file, anno_file_list = [], split_ratio = 0.7):

    if len(anno_file_list)==0:
        return 0

    if os.path.exists(train_file):
        os.remove(train_file)
        
    if os.path.exists(eval_file):
        os.remove(eval_file)

    base_num = 200000  # default-pair : [25w,  10w]
    train_count, eval_count = 0, 0
    for anno_file in anno_file_list:

        with open(anno_file, 'r') as f:
            print(anno_file)
            anno_lines = f.readlines()
        
        if len(anno_lines) > base_num * 3:
            idx_keep = npr.choice(len(anno_lines), size=base_num * 3, replace=True)
        elif len(anno_lines) > base_num * 2:
            idx_keep = npr.choice(len(anno_lines), size=len(anno_lines), replace=True)
        else:
            idx_keep = np.arange(len(anno_lines))
            np.random.shuffle(idx_keep)
        
        num_train = int(len(idx_keep) * split_ratio)
        
        with open(train_file, 'a+') as f1:
            for idx in idx_keep[:num_train]:
                f1.write(anno_lines[idx])
                train_count += 1
    
        with open(eval_file, 'a+') as f2:
            for idx in idx_keep[num_train:]:
                f2.write(anno_lines[idx])
                eval_count += 1
    
    print('There are %d train-instances; %d eval-instances' % (train_count, eval_count))
    return train_count, eval_count
