# -*- coding: utf-8 -*-  

"""
Created on 2021/4/21

@author: Ruoyu Chen
"""

import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np

import Datasets.dataload as dl
from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from utils import *

from Semantically_interpretable import Segmantically_Attributes
from utils import *

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

def Path_Image_Preprocessing(net_type,path):
    '''
    Precessing the input images
        image_dir: single image input path, such as "/home/xxx/10.jpg"
    '''
    if net_type in ['ArcFace-r50','ArcFace-r100',"CosFace-r50","CosFace-r100"]:
        return transforms(Image.open(path).resize((112, 112), Image.BILINEAR))
    elif net_type in ["VGGFace2","VGGFace2-verification"]:
        mean_bgr = np.array([91.4953, 103.8827, 131.0912])
        image = cv2.imread(path)
        assert image is not None
        image = cv2.resize(image,(224,224))
        image = image.astype(np.float32)
        image -= mean_bgr
        # H * W * C   -->   C * H * W
        image = image.transpose(2,0,1)
        return torch.tensor(image)

def Image_Preprocessing(net_type,image):
    '''
    Precessing the input images
        image_dir: single image input path, such as "/home/xxx/10.jpg"
    '''
    if net_type in ['ArcFace-r50','ArcFace-r100',"CosFace-r50","CosFace-r100"]:
        image = Image.fromarray(cv2.cvtColor(np.uint8(image),cv2.COLOR_BGR2RGB))
        return transforms(image.resize((112, 112), Image.BILINEAR))
    elif net_type in ["VGGFace2","VGGFace2-verification"]:
        mean_bgr = np.array([91.4953, 103.8827, 131.0912])
        assert image is not None
        image = cv2.resize(image,(224,224))
        image = image.astype(np.float32)
        image -= mean_bgr
        # H * W * C   -->   C * H * W
        image = image.transpose(2,0,1)
        return torch.tensor(image)

def Read_Datasets(Datasets_type):
    '''
    Read the path in different datasets
    '''
    if Datasets_type == "VGGFace2-train":
        dataset_list = "./Verification/text/VGGFace2-train.txt"
        if os.path.exists(dataset_list):
            with open(dataset_list, "r") as f:
                datas = f.read().split('\n')
        else:
            raise ValueError("File {} not in path".format(dataset_list))
    elif Datasets_type == "VGGFace2-test":
        dataset_list = "./Verification/text/VGGFace2-test.txt"
        if os.path.exists(dataset_list):
            with open(dataset_list, "r") as f:
                datas = f.read().split('\n')
        else:
            raise ValueError("File {} not in path".format(dataset_list))
    elif Datasets_type == "Celeb-A":
        dataset_list = "./Verification/text/CelebA.txt"
        if os.path.exists(dataset_list):
            with open(dataset_list, "r") as f:
                datas = f.read().split('\n')
        else:
            raise ValueError("File {} not in path".format(dataset_list))
    return datas

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Interpretability Test')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        choices=['VGGFace2-train','VGGFace2-test','Celeb-A'],
                        default='VGGFace2-train',
                        help='Datasets.')
    parser.add_argument('--recognition-net',
                        type=str,
                        default='VGGFace2',
                        choices=["VGGFace2"],
                        help='Face identity recognition network.')
    parser.add_argument('--attribute-net',
                        type=str,
                        default='./pre-trained/Face-Attributes2.pth',
                        help='Attribute network, name or path.')
    parser.add_argument('--verification-net',
                        type=str,
                        default='CosFace-r50',
                        choices=['ArcFace-r50','ArcFace-r100',"CosFace-r50","CosFace-r100","VGGFace2-verification"],
                        help='Which network using for face verification.')
    parser.add_argument('--heatmap-method', type=str, default='GradCAM',
                        choices=['GradCAM','GradCAM++'],
                        help='Attribute network.')
    parser.add_argument('--thresh', type=float, default=0.6,
                        help='Thresh.')
    parser.add_argument('--output-dir', type=str, default='./results/',
                        help='output directory to save results.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)