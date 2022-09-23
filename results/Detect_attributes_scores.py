# -*- coding: utf-8 -*-  

"""
Created on 2021/05/05

@author: Ruoyu Chen
"""

import numpy as np
import cv2
import heapq
import json
import argparse
import os
import math
import matplotlib.pyplot as plt
from matplotlib.image import imread

from tqdm import tqdm

attributes_name = [
    "Gender","Age","Race","Bald","Wavy Hair",
    "Receding Hairline","Bangs","Sideburns","Hair color","no beard",
    "Mustache","5 o Clock Shadow","Goatee","Oval Face","Square Face",
    "Round Face","Double Chin","High Cheekbones","Chubby","Obstructed Forehead",
    "Fully Visible Forehead","Brown Eyes","Bags Under Eyes","Bushy Eyebrows","Arched Eyebrows",
    "Mouth Closed","Smiling","Big Lips","Big Nose","Pointy Nose"
]

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return None

def Detect_attributes(path,save_path):
    # path = "./scores-group-Celeb-A-CosFace-r100/1.json"

    plt.figure()
    plt.subplot(1,2,1)

    if path.split('.')[-1]=="json":
        with open(path,'r') as load_f:
            load_dict = json.load(load_f)
        image1 = imread(load_dict["image-path1"])
        image2 = imread(load_dict["image-path2"])

        save_location = np.array([
                0,0,0,1,1,
                1,1,1,0,1,
                1,1,1,1,1,
                1,1,1,1,1,
                1,1,1,1,1,
                1,1,1,1,1
            ])

        i = 1
        image = [image1,image2]
        for attribute_class,attribute_score in zip(["Attribute1-class","Attribute2-class"],["Attribute1-score","Attribute2-score"]):
            attr_class = load_dict[attribute_class]
            attr_score = load_dict[attribute_score]

            jud = attr_class * save_location
            jud[jud != 0] = 2
            jud[jud == 0] = 1
            jud[jud == 2] = 0

            title = ""

            for attribute_id in range(30):
                if jud[attribute_id] == 1:
                    title = title + attributes_name[attribute_id] + '    ' + str(attr_class[attribute_id]) + '    ' + '%.4f' % attr_score[attribute_id] + '\n'
            
            plt.subplot(1,2,i)
            plt.axis('off')
            plt.title(title, size=6,y=-0.6)
            plt.imshow(image[i-1])
            i+=1

    plt.savefig(save_path)
    # plt.show()
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Plot AUC curve')
    # general
    parser.add_argument('--Json-fold',
                        type=str,
                        default='./Multi-ID-topk/scores-group-VGGFace2-test-CosFace-r50-topk-1/json',
                        help='Datasets.')

    args = parser.parse_args()
    return args

def main(args):
    save_fold = args.Json_fold.replace("json",'images-attributes')
    mkdir(save_fold)

    lists = os.listdir(args.Json_fold)

    for path in tqdm(lists):
        if path.split('.')[-1]=="json":
            Detect_attributes(os.path.join(args.Json_fold,path),os.path.join(save_fold,path.replace(".json",".jpg")))
        break

if __name__ == "__main__":
    args = parse_args()
    main(args)