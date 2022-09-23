# -*- coding: utf-8 -*-  

"""
Created on 2021/05/23

@author: Ruoyu Chen
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import math

from tqdm import tqdm

def main(args):
    lists = os.listdir(args.Json_fold)
    attributes_found = np.zeros(30)

    save_location = np.array([
            0,0,0,1,1,
            1,1,1,0,1,
            1,1,1,1,1,
            1,1,1,1,1,
            1,1,1,1,1,
            1,1,1,1,1
        ])

    for path in tqdm(lists):
        
        if path.split('.')[-1]=="json":
            with open(os.path.join(args.Json_fold,path),'r') as load_f:
                load_dict = json.load(load_f)
            for image_key, attribute_class in zip(["attributes-image1","attributes-image2"],["Attribute1-class","Attribute2-class"]):
                if load_dict["match"] == 0:
                    
                    attr_dict = load_dict[image_key]
                    attr_class = load_dict[attribute_class]

                    jud = attr_class * save_location
                    jud[jud != 0] = 2
                    jud[jud == 0] = 1
                    jud[jud == 2] = 0

                    value = np.array(list(attr_dict.values()))[:,-1]

                    value = value * jud
                    value[value==0]=-1
                    
                    max_value = np.max(value)
                    if max_value==max_value: # No nan
                        number = np.sum(np.array(value>load_dict["similarity"],np.int32))

                        attributes_found[number] += 1
                        
    print(attributes_found)
    print(np.sum(attributes_found))

def parse_args():
    parser = argparse.ArgumentParser(description='Stastic the found attributes number')
    # general
    parser.add_argument('--Json-fold',
                        type=str,
                        # default='./Multi-ID-topk/scores-group-Celeb-A-VGGFace2-verification-topk-1/json',
                        default='./ablation-no-counterfactual/scores-group-VGGFace2-train-VGGFace2-verification-topk-1-erase-black/json',
                        help='Datasets.')
    parser.add_argument('--Attribute-number',
                        type=int,
                        default=30,
                        help='Number of attribute.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)