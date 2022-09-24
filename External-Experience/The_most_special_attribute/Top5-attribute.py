# -*- coding: utf-8 -*-  

"""
Created on 2021/4/11

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

def similarity2distance(similarity):
    '''
    Convert similarity range [-1,1] to cosine distance [0,1]
        return a new similarity range [0,1], the value more close 
        to 1 that the face more similar.
    '''
    dist = np.arccos(similarity) / math.pi
    return 1-dist

def Top_5_attributes(path,save_path):
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
        for image_key, attribute_class in zip(["attributes-image1","attributes-image2"],["Attribute1-class","Attribute2-class"]):
            attr_dict = load_dict[image_key]
            attr_class = load_dict[attribute_class]

            jud = attr_class * save_location
            jud[jud != 0] = 2
            jud[jud == 0] = 1
            jud[jud == 2] = 0

            value = np.array(list(attr_dict.values()))[:,-1]

            value = value * jud
            value[value==0]=-1

            index = heapq.nlargest(7,range(len(value)),value.__getitem__)
            value_convert = similarity2distance(np.array(value)[index])
            
            title = ""

            for id_,v in zip(index,value_convert.tolist()):
                
                title = title + attributes_name[id_] + ' ' + '%.4f' % v + '\n'

            plt.subplot(1,2,i)
            plt.axis('off')
            plt.title(title, size=10, y=-0.5)
            plt.imshow(image[i-1])
            i+=1

        attr_dict = load_dict["attributes-both-masked"]

        attr_class1 = load_dict["Attribute1-class"]
        attr_class2 = load_dict["Attribute2-class"]

        jud1 = attr_class1 * save_location
        jud1[jud1 != 0] = 2; jud1[jud1 == 0] = 1; jud1[jud1 == 2] = 0

        jud2 = attr_class2 * save_location
        jud2[jud2 != 0] = 2; jud2[jud2 == 0] = 1; jud2[jud2 == 2] = 0

        jud = jud1+jud2; jud[jud>0]=1

        value = list(attr_dict.values())
        
        value = value * jud
        value[value==0]=-1

        index = heapq.nlargest(6,range(len(value)),value.__getitem__)
        value_convert = similarity2distance(np.array(value)[index])

        title = "Similarity: "+"%.4f"%(1-math.acos(load_dict["similarity"])/math.pi)+"\n"
        
        for id_,v in zip(index,value_convert.tolist()):
            title = title + attributes_name[id_] + ' ' + '%.4f' % v + '\n'

        plt.suptitle(title,size=10)
    plt.savefig(save_path)
    # plt.show()
    plt.close()

def main(args):
    save_fold = args.Json_fold.replace("json",'images')
    mkdir(save_fold)

    lists = os.listdir(args.Json_fold)

    for path in tqdm(lists):
        if path.split('.')[-1]=="json":
            # try:
            Top_5_attributes(os.path.join(args.Json_fold,path),os.path.join(save_fold,path.replace(".json",".jpg")))
            # except:
            #     pass
            # break

def parse_args():
    parser = argparse.ArgumentParser(description='Plot AUC curve')
    # general
    parser.add_argument('--Json-fold',
                        type=str,
                        default='./Multi-ID-topk-vis/scores-group-Test1-VGGFace2-verification-topk-1-erase-black/json',
                        help='Datasets.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)