#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

from IPython import embed


def xxyy2xyxy(anno_file, trans_file):
	''' CUHKMM '''

	with open(anno_file, 'r') as f:
		anno_info = f.readlines()
	f.close()

	img_list, bbox_list, lmk_list = [], [], []
	box_idx = [0, 2, 1, 3]
	lmk_idx = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
	with open(trans_file, 'w') as f:
		for row in anno_info:

			row  = '/'.join(row.strip().split('\\')).split(' ')
			bbox = list(np.array(row[1:5], dtype=np.float32)[box_idx])
			lmks = list(np.array(row[5:], dtype=np.float32)[lmk_idx])
			bbox_str = ' '.join([str(c) for c in bbox])
			lmks_str = ' '.join([str(c) for c in lmks])
			info = row[0] + ' ' + bbox_str + ' ' + lmks_str + '\n'
			f.write(info)
	f.close()
	print('format trans was finished ...')



# def xywh2xyxy(anno_file, trans_file):
#     ''' Wide Face '''
#
# 	with open(anno_file, 'r') as f:
# 		anno_info = f.readlines()
# 	f.close()
#
# 	img_list, bbox_list = [], []
# 	for row in anno_info:
#
# 		row  = row.split(' ')
# 		bbox = np.array(row[1:-1], dtype=np.int32).reshape(-1, 4)
#         img_list.append(row[0])
# 		bbox_list.append(bbox)
#
# 	for coordinate in bbox_list:
# 		coordinate[:, 2] = coordinate[:, 0] + coordinate[:, 2]
# 		coordinate[:, 3] = coordinate[:, 1] + coordinate[:, 3]
#
# 	with open(trans_file, 'w') as f:
# 		for n, c in zip(img_list, bbox_list):
# 			a = str(list(c.reshape(1, -1)[0, :]))[1:-1].split(',')
# 			s = ''
# 			for i in a:
# 				s = s + i
# 			content = n + ' ' + s + '\n'
#
# 			f.write(content)
# 	f.close()
#     print('format trans was finished ...')



if __name__ == '__main__':

	anno_file  = '/Volumes/ubuntu/relu/benchmark_images/faceu/landmark/cuhk_mm/anno_file.txt'
	trans_file = './trans_file.txt'

	xxyy2xyxy(anno_file, trans_file)
