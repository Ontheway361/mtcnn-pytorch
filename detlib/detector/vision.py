#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from IPython import embed

def easy_vis(img, bboxes, lmks, save_path):
    bboxes = bboxes.astype('int')
    for box in bboxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
    for lmk in lmks:
        lmk = lmk.reshape((5, 2))
        for i in range(5):
            cv2.circle(img, (lmk[i, 0], lmk[i, 1]), radius=2, color=(0, 255, 0))
    cv2.imwrite(save_path, img)
