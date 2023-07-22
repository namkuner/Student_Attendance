#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:09:25 2019
Take cropped face from image

@author: AIRocker
"""

import sys
import os
sys.path.append(os.path.join(sys.path[0], 'MTCNN'))
from MTCNN import create_mtcnn_net
from align_trans import *
import cv2
import argparse
from datetime import datetime
import torch
from pathlib import Path

parser = argparse.ArgumentParser(description='take ID from Picture')
parser.add_argument('--image','-i', default='images/Sheldon.jpg', type=str,help='input the image of the person')
parser.add_argument('--name','-n', default='Sheldon', type=str,help='input the name of the person')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image = cv2.imread("WIN_20230608_16_46_40_Pro.jpg")
bboxes, landmarks = create_mtcnn_net(image, 20, device,
                                     p_model_path='Weights/pnet_Weights',
                                     r_model_path='Weights/rnet_Weights',
                                     o_model_path='Weights/onet_Weights')
print(landmarks)
warped_face = Face_alignment(image, default_square=True, landmarks=landmarks)
cv2.imshow("abc",warped_face[0])

data_path = Path('facebank')
print(data_path)
save_path = data_path  / "Sheldon"
print(save_path)
if not save_path.exists():
    save_path.mkdir()

cv2.imwrite(str(save_path/'{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-"))), warped_face[0])
