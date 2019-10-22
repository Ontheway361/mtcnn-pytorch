#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
from mtcnn.core.vision import vis_face
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector

from IPython import embed

if __name__ == '__main__':
    
    model_info = 'self'
    imglists   = [s.split('.')[0] for s in os.listdir('imgs/')]
    # org
#     pnet, rnet, onet = create_mtcnn_net(p_model_path='model/original/pnet_epoch.pt', \
#                                         r_model_path='model/original/rnet_epoch.pt', \
#                                         o_model_path="model/original/onet_epoch.pt", \
#                                         use_cuda=False)
    # self
    pnet, rnet, onet = create_mtcnn_net(p_model_path='model/checkout/pnet.pt', \
                                        r_model_path='model/checkout/rnet.pt', \
                                        o_model_path='model/checkout/onet.pt', \
                                        use_cuda=True)

    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)
    
    for img_name in imglists:
        
        img    = cv2.imread('imgs/%s.jpg' % img_name)
        img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxs, landmarks = mtcnn_detector.detect_face(img)

        save_name = 'result/r_%s_%s.jpg' % (img_name, model_info)
        print('save img name : %s' % save_name)
        vis_face(img_bg, bboxs, landmarks, save_name)
