#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import time
from mtcnn.core.vision import vis_face
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector

from IPython import embed

if __name__ == '__main__':

    pnet, rnet, onet = create_mtcnn_net(p_model_path='./model/pnet_epoch.pt', \
                                        r_model_path='./model/rnet_epoch.pt', \
                                        o_model_path="./model/onet_epoch.pt", \
                                        use_cuda=False)

    mtcnn_detector   = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    img    = cv2.imread('./imgs/test1.jpg')
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bboxs, landmarks = mtcnn_detector.detect_face(img)

    save_name = 'r_test1.jpg'
    vis_face(img_bg, bboxs, landmarks, save_name)
