#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
sys.path.append(os.getcwd())
from mtcnn.core.imagedb import ImageDB
from mtcnn.train_net.train import train_pnet

from IPython import embed

anno_dir = '/home/faceu/5keypoints/anno_store'

def train_net(args):

    imagedb    = ImageDB(args.anno_file)
    train_imdb = imagedb.load_imdb()
    train_imdb = imagedb.append_flipped_images(train_imdb)
    
    imagedb    = ImageDB(args.eval_file)
    eval_imdb  = imagedb.load_imdb()
    
    print('train : %d\teval : %d' % (len(train_imdb), len(eval_imdb)))
    
    train_pnet(args, train_imdb, eval_imdb)


def parse_args():

    parser = argparse.ArgumentParser(description='Train PNet')

    parser.add_argument('--anno_file',  type=str,   default=os.path.join(anno_dir, 'pnet/train_anno_12.txt'))
    parser.add_argument('--eval_file',  type=str,   default=os.path.join(anno_dir, 'pnet/eval_anno_12.txt'))
    parser.add_argument('--model_path', type=str,   default='./model/checkout/pnet')
    parser.add_argument('--factors',    type=list,  default=[1.0, 0.5, 0.5])
    parser.add_argument('--end_epoch',  type=int,   default=20)      # TODO
    parser.add_argument('--frequent',   type=int,   default=1000)
    parser.add_argument('--lr',         type=float, default=1e-3)    # TODO  :: 1e-2
    parser.add_argument('--batch_size', type=int,   default=512)     # TODO
    parser.add_argument('--use_cuda',   type=bool,  default=True)    # TODO

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    train_net(parse_args())
