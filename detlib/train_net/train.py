#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import torch
import numpy as np
from torch.autograd import Variable
import detlib.detector.image_tools as image_tools
from detlib.detector.image_reader import TrainImageReader
from detlib.detector.models import PNet,RNet,ONet,LossFn

from IPython import embed

def compute_accuracy(prob_cls, gt_cls, thresh_prob = 0.6):
    ''' Just focus on negative and positive instance '''

    prob_cls = torch.squeeze(prob_cls)
    gt_cls   = torch.squeeze(gt_cls)
    mask     = torch.ge(gt_cls, 0) # gt_cls >= 0
    valid_gt_cls   = torch.masked_select(gt_cls, mask)
    valid_prob_cls = torch.masked_select(prob_cls, mask)
    size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
    prob_ones  = torch.ge(valid_prob_cls, thresh_prob).float()
    right_ones = torch.eq(prob_ones, valid_gt_cls).float()

    return torch.div(torch.mul(torch.sum(right_ones), float(1.0)), float(size))


def eval_net(args, net, eval_data):
    ''' Monitor the training process '''

    net.eval()
    st_acc, st_cls, st_det, st_lmk, st_all = 0, 0, 0, 0, 0
    lossfn, batch_idx = LossFn(), 1   # TODO
    for image, (gt_label, gt_bbox, gt_landmark) in eval_data:

        im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]

        im_tensor = torch.stack(im_tensor)

        im_tensor   = Variable(im_tensor)
        gt_label    = Variable(torch.from_numpy(gt_label).float())
        gt_bbox     = Variable(torch.from_numpy(gt_bbox).float())
        gt_landmark = Variable(torch.from_numpy(gt_landmark).float())

        if args.use_cuda:
            im_tensor   = im_tensor.cuda()
            gt_label    = gt_label.cuda()
            gt_bbox     = gt_bbox.cuda()
            gt_landmark = gt_landmark.cuda()

        with torch.no_grad():
            cls_pred, box_offset_pred, landmark_offset_pred = net(im_tensor)

        cls_loss        = lossfn.cls_loss(gt_label, cls_pred)
        box_offset_loss = lossfn.box_loss(gt_label, gt_bbox, box_offset_pred)
        landmark_loss   = lossfn.landmark_loss(gt_label, gt_landmark, landmark_offset_pred)
        all_loss        = cls_loss * args.factors[0] + box_offset_loss * args.factors[1] \
                               + landmark_loss * args.factors[2]

        accuracy = compute_accuracy(cls_pred, gt_label)
        st_acc += accuracy.data.cpu().numpy()
        st_cls += cls_loss.data.cpu().numpy()
        st_det += box_offset_loss.data.cpu().numpy()
        st_all += all_loss.data.cpu().numpy()

        try:
            lmk_loss = landmark_loss.data.cpu().numpy()   # TODO
        except:
            lmk_loss = 0.0
        st_lmk += lmk_loss

        batch_idx += 1

    st_acc /= batch_idx
    st_cls /= batch_idx
    st_det /= batch_idx
    st_lmk /= batch_idx
    st_all /= batch_idx
    st_cache = (st_acc, st_cls, st_det, st_lmk, st_all)
    print("Eval result acc: %.4f, cls_loss: %.4f, bbox_loss: %.4f, lmk_loss: %.4f, all_loss: %.4f" % \
          (st_acc, st_cls, st_det, st_lmk, st_all))

    return st_cache


def train_pnet(args, train_imdb, eval_imdb):
    ''' paper ratio : 1.0 : 0.5 : 0.5 '''

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    lossfn = LossFn()
    net = PNet(is_train=True, use_cuda=args.use_cuda)

    if args.use_cuda:
        net.cuda()
    optimizer  = torch.optim.Adam(net.parameters(), lr=args.lr)   # TODO
    train_data = TrainImageReader(train_imdb, 12, args.batch_size, shuffle=True)
    eval_data  = TrainImageReader(eval_imdb, 12 , args.batch_size, shuffle=True)
    
    for cur_epoch in range(1, args.end_epoch+1):

        # training-process
        train_data.reset() # shuffle
        net.train()
        start_time = time.time()
        
        record_acc, record_cls, record_box, record_lmk, record_all = 0, 0, 0, 0, 0
        for batch_idx, (image, (gt_label, gt_bbox, gt_landmark)) in enumerate(train_data):

            im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]

            im_tensor = torch.stack(im_tensor)

            im_tensor   = Variable(im_tensor)
            gt_label    = Variable(torch.from_numpy(gt_label).float())
            gt_bbox     = Variable(torch.from_numpy(gt_bbox).float())
            gt_landmark = Variable(torch.from_numpy(gt_landmark).float())

            if args.use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label  = gt_label.cuda()
                gt_bbox   = gt_bbox.cuda()
                gt_landmark = gt_landmark.cuda()

            cls_pred, box_offset_pred, landmark_offset_pred = net(im_tensor)   # NOTE
            
            cls_loss = lossfn.cls_loss(gt_label, cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label, gt_bbox, box_offset_pred)
            landmark_loss = lossfn.landmark_loss(gt_label, gt_landmark, landmark_offset_pred)  # BUG
            all_loss = cls_loss * args.factors[0] + box_offset_loss * args.factors[1] + landmark_loss * args.factors[2]
            
            accuracy = compute_accuracy(cls_pred, gt_label)
            record_acc += accuracy.data.cpu().numpy()
            record_cls += cls_loss.data.cpu().numpy()
            record_box += box_offset_loss.data.cpu().numpy()
            try:
                lmk_loss = landmark_loss.data.cpu().numpy()
            except:
                lmk_loss = 0 # means no landmark
                    
            record_lmk += lmk_loss
            record_all += all_loss.data.cpu().numpy()
            
            if (batch_idx + 1) % args.frequent==0:
                print("Epoch: %d|%d, bid: %d, acc: %.4f, cls_loss: %.4f, bbox_loss: %.4f, lmk_loss: %.4f, all_loss: %.4f"\
                      % (cur_epoch, args.end_epoch, (batch_idx + 1), record_acc/batch_idx, record_cls/batch_idx, \
                         record_box/batch_idx, record_lmk/batch_idx, record_all/batch_idx))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
        end_time = time.time()
        print('single epoch cost time : %.2f mins' % ((end_time-start_time) / 60))
        
        eval_data.reset()
        res_cache = eval_net(args, net, eval_data)
        torch.save(net.state_dict(), os.path.join(args.model_path, "pnet_epoch_%d.pt" % cur_epoch))


def train_rnet(args, imdb, eval_imdb):

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    lossfn = LossFn()
    net = RNet(is_train=True, use_cuda=args.use_cuda)

    if args.use_cuda:
        net.cuda()

    optimizer  = torch.optim.Adam(net.parameters(), lr=args.lr)
    train_data = TrainImageReader(imdb, args.imgsize, args.batch_size, shuffle=True)
    eval_data  = TrainImageReader(eval_imdb, args.imgsize, args.batch_size, shuffle=True)

    for cur_epoch in range(1, args.end_epoch+1):

        train_data.reset()
        net.train()
        record_acc, record_cls, record_box, record_lmk, record_all = 0, 0, 0, 0, 0
        for batch_idx, (image, (gt_label, gt_bbox, gt_landmark))in enumerate(train_data):

            im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) \
                          for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor)
            gt_label  = Variable(torch.from_numpy(gt_label).float())

            gt_bbox   = Variable(torch.from_numpy(gt_bbox).float())
            gt_landmark = Variable(torch.from_numpy(gt_landmark).float())

            if args.use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label  = gt_label.cuda()
                gt_bbox   = gt_bbox.cuda()
                gt_landmark = gt_landmark.cuda()

            cls_pred, box_offset_pred, landmark_offset_pred = net(im_tensor)

            cls_loss        = lossfn.cls_loss(gt_label,cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)
            landmark_loss   = lossfn.landmark_loss(gt_label, gt_landmark, landmark_offset_pred)
            all_loss        = cls_loss*args.factors[0] + box_offset_loss*args.factors[1] + landmark_loss*args.factors[2]
            
            accuracy = compute_accuracy(cls_pred, gt_label)
            record_acc += accuracy.data.cpu().numpy()
            record_cls += cls_loss.data.cpu().numpy()
            record_box += box_offset_loss.data.cpu().numpy()
            try:
                lmk_loss = landmark_loss.data.cpu().numpy()
            except:
                lmk_loss = 0 # means no landmark
                    
            record_lmk += lmk_loss
            record_all += all_loss.data.cpu().numpy()
            
            if (batch_idx + 1) % args.frequent==0:
                print("Epoch: %d|%d, bid: %d, acc: %.4f, cls_loss: %.4f, bbox_loss: %.4f, lmk_loss: %.4f, all_loss: %.4f"\
                      % (cur_epoch, args.end_epoch, (batch_idx + 1), record_acc/batch_idx, record_cls/batch_idx, \
                         record_box/batch_idx, record_lmk/batch_idx, record_all/batch_idx))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
        
        eval_data.reset()
        res_cache = eval_net(args, net, eval_data)

        torch.save(net.state_dict(), os.path.join(args.model_path,"rnet_epoch_%d.pt" % cur_epoch))


def train_onet(args, train_imdb, eval_imdb):

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    lossfn = LossFn()   # TODO
    net = ONet(is_train=True)
    if args.use_cuda:
        net.cuda()

    optimizer  = torch.optim.Adam(net.parameters(), lr=args.lr)
    train_data = TrainImageReader(train_imdb, args.imgsize, args.batch_size, shuffle=True)
    eval_data  = TrainImageReader(eval_imdb, args.imgsize, args.batch_size, shuffle=True)

    for cur_epoch in range(1, args.end_epoch+1):

        train_data.reset()
        net.train()
        record_acc, record_cls, record_box, record_lmk, record_all = 0, 0, 0, 0, 0
        for batch_idx, (image, (gt_label, gt_bbox, gt_landmark)) in enumerate(train_data):

            im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) \
                          for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor   = Variable(im_tensor)
            gt_label    = Variable(torch.from_numpy(gt_label).float())
            gt_bbox     = Variable(torch.from_numpy(gt_bbox).float())
            gt_landmark = Variable(torch.from_numpy(gt_landmark).float())

            if args.use_cuda:
                im_tensor   = im_tensor.cuda()
                gt_label    = gt_label.cuda()
                gt_bbox     = gt_bbox.cuda()
                gt_landmark = gt_landmark.cuda()

            cls_pred, box_offset_pred, landmark_offset_pred = net(im_tensor)

            cls_loss        = lossfn.cls_loss(gt_label, cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label, gt_bbox, box_offset_pred)

            landmark_loss   = lossfn.landmark_loss(gt_label, gt_landmark, landmark_offset_pred)
            all_loss        = cls_loss*args.factors[0] + box_offset_loss*args.factors[1] + landmark_loss*args.factors[2]
            
            accuracy = compute_accuracy(cls_pred, gt_label)
            record_acc += accuracy.data.cpu().numpy()
            record_cls += cls_loss.data.cpu().numpy()
            record_box += box_offset_loss.data.cpu().numpy()
            try:
                lmk_loss = landmark_loss.data.cpu().numpy()
            except:
                lmk_loss = 0 # means no landmark
                    
            record_lmk += lmk_loss
            record_all += all_loss.data.cpu().numpy()
            
            if (batch_idx + 1) % args.frequent==0:
                print("Epoch: %d|%d, bid: %d, acc: %.4f, cls_loss: %.4f, bbox_loss: %.4f, lmk_loss: %.4f, all_loss: %.4f"\
                      % (cur_epoch, args.end_epoch, (batch_idx + 1), record_acc/batch_idx, record_cls/batch_idx, \
                         record_box/batch_idx, record_lmk/batch_idx, record_all/batch_idx))
            
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        eval_data.reset()
        res_cache = eval_net(args, net, eval_data)
        torch.save(net.state_dict(), os.path.join(args.model_path, 'onet_epoch_%d.pt' % cur_epoch))
        # torch.save(net, os.path.join(args.model_path,"onet_epoch_model_%d.pkl" % cur_epoch))
