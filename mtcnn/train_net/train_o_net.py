#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
sys.path.append(os.getcwd())
import mtcnn.train_net.train as train
from mtcnn.core.imagedb import ImageDB

from IPython import embed

def train_net(args):

    imagedb    = ImageDB(args.anno_file)
    train_imdb = imagedb.load_imdb()
    train_imdb = imagedb.append_flipped_images(train_imdb)
    
    imagedb    = ImageDB(args.eval_file)
    eval_imdb  = imagedb.load_imdb()
    
    print('train : %d\teval : %d' % (len(train_imdb), len(eval_imdb)))
    
    train.train_onet(args, train_imdb, eval_imdb)


def parse_args():

    parser = argparse.ArgumentParser(description='Train  ONet')
    
    # {landmark_48_train:311617, landmark_48_cls_train:15809}
    parser.add_argument('--anno_file',  type=str,   default='./anno_store/onet/train_anno_48.txt')  
    parser.add_argument('--eval_file',  type=str,   default='./anno_store/onet/eval_anno_48.txt')
    parser.add_argument('--model_path', type=str,   default='./68_keypoints_model/onet')
    parser.add_argument('--factors',    type=list,  default=[1, 0.5, 2.4])  # [0.8, 0.6, 1.5]
    parser.add_argument('--use_lmkinfo',type=bool,  default=True)
    parser.add_argument('--imgsize',    type=int,   default=48)
    parser.add_argument('--end_epoch',  type=int,   default=20)
    parser.add_argument('--frequent',   type=int,   default=1000)
    parser.add_argument('--lr',         type=float, default=1e-3)    # TODO
    parser.add_argument('--batch_size', type=int,   default=64)      # TODO
    parser.add_argument('--use_cuda',   type=bool,  default=True)    # TODO
    parser.add_argument('--prefix_path',type=str,   default='')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    train_net(parse_args())
