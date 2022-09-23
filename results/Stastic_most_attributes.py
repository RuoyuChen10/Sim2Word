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

from tqdm import tqdm

def main(args):
    lists = os.listdir(args.Json_fold)
    attributes = np.zeros(30)
    gender = np.zeros(2)
    age = np.zeros(3)
    race = np.zeros(3)
    hair_color = np.zeros(5)

    save_location = np.array([
            0,0,0,1,1,
            1,1,1,0,1,
            1,1,1,1,1,
            1,1,1,1,1,
            1,1,1,1,1,
            1,1,1,1,1
        ])
    i = 0
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
                    if max_value==max_value:
                        index = np.where(value == max_value)[0][0]

                        if max_value < load_dict["similarity"]:
                            max_value = load_dict["similarity"]
                            pass
                            
                        if index == 0:
                            gender[attr_class[0]] += 1
                        
                        elif index == 1:
                            age[attr_class[1]] += 1
                        
                        elif index == 2:
                            race[attr_class[2]] += 1

                        elif index == 8:
                            hair_color[attr_class[8]] += 1

                        attributes[index] += 1
                        i += 1
                        print(i)
    print(attributes)
    print(np.sum(attributes))

    print(gender)
    print(age)
    print(race)
    print(hair_color)

def parse_args():
    parser = argparse.ArgumentParser(description='Stastic the most attributes')
    # general
    parser.add_argument('--Json-fold',
                        type=str,
                        default='./Multi-ID-topk/scores-group-VGGFace2-test-CosFace-r100-topk-1/json',
                        # default='./ablation-no-counterfactual/scores-group-VGGFace2-train-VGGFace2-verification-topk-1-erase-black/json',
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