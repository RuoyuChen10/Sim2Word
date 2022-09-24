import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import random
import tensorflow as tf
from tqdm import tqdm
import math

from MTCNN_Portable.mtcnn import MTCNN

VGGFace2_test_image_path="/home/cry/data2/VGGFace2/train"

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

# load input images and corresponding 5 landmarks
def load_img_and_box(img_path, detector):
    #Reading image
    image = Image.open(img_path)
    if img_path.split('.')[-1]=='png':
        image = image.convert("RGB")
    # BGR
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    #Detect 5 key point
    face = detector.detect_faces(img)[0]
    box = face["box"]
    image = cv2.imread(img_path)
    return image, box

def box_crop(image, box):
    shape=(224,224)
    # print(image.shape)
    # print((int(np.floor(box[1]-box[3]*0.15)),int(np.ceil(box[1]+box[3]*1.15))))
    # print((int(np.floor(box[0]-box[2]*0.15)),int(np.ceil(box[0]+box[2]*1.15))))

    if int(np.floor(box[1]-box[3]*0.15)) < 0:
        top = 0
        top_ = -int(np.floor(box[1]-box[3]*0.15))
    else:
        top = int(np.floor(box[1]-box[3]*0.15))
        top_ = 0
    if int(np.ceil(box[1]+box[3]*1.15)) > image.shape[0]:
        bottom = image.shape[0]
    else:
        bottom = int(np.ceil(box[1]+box[3]*1.15))
    if int(np.floor(box[0]-box[2]*0.15)) < 0:
        left = 0
        left_ = -int(np.floor(box[0]-box[2]*0.15))
    else:
        left = int(np.floor(box[0]-box[2]*0.15))
        left_ = 0
    if int(np.ceil(box[0]+box[2]*1.15)) > image.shape[1]:
        right = image.shape[1]
    else:
        right = int(np.ceil(box[0]+box[2]*1.15))
    img_zero = np.zeros((int(np.ceil(box[1]+box[3]*1.15))-int(np.floor(box[1]-box[3]*0.15)),
                         int(np.ceil(box[0]+box[2]*1.15))-int(np.floor(box[0]-box[2]*0.15)),
                         3))
    img = image[
        top:bottom,
        left:right]
    img_zero[top_:top_+bottom-top,left_:left_+right-left] = img
    img = img_zero
    im_shape = img.shape[:2]
    ratio = float(shape[0]) / np.min(im_shape)
    img = cv2.resize(
        img,
        dsize=(math.ceil(im_shape[1] * ratio),   # width
            math.ceil(im_shape[0] * ratio))  # height
        )
    new_shape = img.shape[:2]
    h_start = (new_shape[0] - shape[0])//2
    w_start = (new_shape[1] - shape[1])//2
    img = img[h_start:h_start+shape[0], w_start:w_start+shape[1]]
    return img


with open("./List.txt", "r") as f:  # 打开文件
    data = f.read()  # 读取文件
data = data.split('\n')

with tf.device('gpu:0'):
    detector = MTCNN()
    for image_paths in tqdm(data):
        try:
            path1 = os.path.join(VGGFace2_test_image_path,image_paths.split(' ')[0])
            path2 = os.path.join(VGGFace2_test_image_path,image_paths.split(' ')[1])
            
            for image_path in [path2]:
                img,box = load_img_and_box(image_path, detector)
                img_ = box_crop(img, box)
                idy = image_path.split('/')[-2]
                mkdir(os.path.join("./dataset/VGGFace2-train",idy))
                cv2.imwrite(os.path.join("./dataset/VGGFace2-train",idy,image_path.split('/')[-1]),img_)
        except:
            pass
    for image_path in [path1]:
        img,box = load_img_and_box(image_path, detector)
        img_ = box_crop(img, box)
        idy = image_path.split('/')[-2]
        mkdir(os.path.join("./dataset/VGGFace2-train",idy))
        cv2.imwrite(os.path.join("./dataset/VGGFace2-train",idy,image_path.split('/')[-1]),img_)