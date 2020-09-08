#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import detlib as dlib

from IPython import embed

if __name__ == '__main__':
    
    faceu = dlib.DetectFace(min_face=24, use_cuda=False)
    for img_file in os.listdir('imgs/'):
        
        if '.jpg' in img_file:
            img = cv2.imread(os.path.join('imgs', img_file))
            bboxs, landmarks = faceu.detect_face(img)
            save_name = 'result/r_%s' % img_file
            dlib.easy_vis(img, bboxs, landmarks, save_name)
