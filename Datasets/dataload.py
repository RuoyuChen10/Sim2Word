# -*- coding: utf-8 -*-  

"""
Created on 2021/1/31

@author: Ruoyu Chen
"""

import os
import cv2
import numpy as np
import torch

import torchvision.transforms as transforms

from PIL import Image

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

def data_dir_read(path):
    '''
    Read the TXT file and get the dictionaries and labels
        path: the path to datasets dir
    '''
    data = []
    for line in open(path):
        data.append(line)
    return data

def Image_precessing(image_dir):
    '''
    Precessing the input images
        image_dir: single image input path, such as "/home/xxx/10.jpg"
    '''
    try:
        mean_bgr = np.array([91.4953, 103.8827, 131.0912])
        image = cv2.imread(image_dir)
        assert image is not None
        image = cv2.resize(image,(224,224))
        image = image.astype(np.float32)
        image -= mean_bgr
        # H * W * C   -->   C * H * W
        image = image.transpose(2,0,1)
        return image
    except:
        return None

def analysis_data(data, datasets_dir):
    '''
    Transpose the charactor to image and label
    '''
    inputs = []
    labels1 = []
    labels2 = []
    labels3 = []
    for charactor in data:
        image_dir = charactor.split(' ')[0]
        Image = Image_precessing(os.path.join(datasets_dir,image_dir))
        if Image is not None:
            inputs.append(Image)
            labels1.append(int(charactor.split(' ')[1]))
            labels2.append(int(charactor.split(' ')[2]))
            labels3.append(int(charactor.split(' ')[3][:-1]))
        else:
            continue
    return np.array(inputs), labels1, labels2, labels3

"""
Update on 2021/4/21 add function:
    Path_Image_Preprocessing(net_type,path)
    Image_Preprocessing(net_type,image)
    Read_Datasets(Datasets_type)

@author: Ruoyu Chen
"""

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