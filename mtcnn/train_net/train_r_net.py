#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
sys.path.append(os.getcwd())
import mtcnn.train_net.train as train
from mtcnn.core.imagedb import ImageDB

anno_dir = '/home/faceu/5keypoints/anno_store/rnet'

def train_net(args):

    imagedb = ImageDB(args.anno_file)
    gt_imdb = imagedb.load_imdb()
    gt_imdb = imagedb.append_flipped_images(gt_imdb) # data argument

    eval_db = ImageDB(args.eval_file)
    ev_imdb = eval_db.load_imdb()
    print('train : %d\teval : %d' % (len(gt_imdb), len(ev_imdb)))
    
    train.train_rnet(args, gt_imdb, ev_imdb)


def parse_args():

    parser = argparse.ArgumentParser(description='Train  RNet')

    parser.add_argument('--anno_file',  type=str,   default=os.path.join(anno_dir, 'train_anno_24.txt'))
    parser.add_argument('--eval_file',  type=str,   default=os.path.join(anno_dir, 'eval_anno_24.txt'))
    parser.add_argument('--model_path', type=str,   default='model/checkout/rnet')
    parser.add_argument('--factors',    type=list,  default=[1.0, 0.5, 0.5])
    parser.add_argument('--use_lmkinfo',type=bool,  default=False)       # TODO
    parser.add_argument('--imgsize',    type=int,   default=24)
    parser.add_argument('--end_epoch',  type=int,   default=20)
    parser.add_argument('--frequent',   type=int,   default=1000)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int,   default=256)     # default = 32
    parser.add_argument('--use_cuda',   type=bool,  default=True)    # TODO
    parser.add_argument('--prefix_path',type=str,   default='')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    train_net(parse_args())
