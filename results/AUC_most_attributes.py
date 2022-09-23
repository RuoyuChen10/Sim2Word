# -*- coding: utf-8 -*-  

"""
Created on 2021/4/11

@author: Ruoyu Chen
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import math
from sklearn import metrics
from sklearn.metrics import auc
from prettytable import PrettyTable
from tqdm import tqdm

def AUC(score,label):
    score = np.array(score)
    label = np.array(label)

    x_labels = [10**-6, 10**-5, 10**-4,10**-3, 10**-2, 10**-1,0.2,0.4,0.6,0.8,1]
    tpr_fpr_table = PrettyTable(map(str, x_labels))
    
    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # print(fpr,tpr)

    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr) # select largest tpr at same fpr

    tpr_fpr_row = []
    
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.4f' % tpr[min_index])
    tpr_fpr_table.add_row(tpr_fpr_row)

    print(tpr_fpr_table)
    print("ROC AUC: {}".format(roc_auc))

def Metric(args):
    lists = os.listdir(args.Json_fold)
    scores = []
    label = []

    for path in tqdm(lists):
        
        if path.split('.')[-1]=="json":
            with open(os.path.join(args.Json_fold,path),'r') as load_f:
                load_dict = json.load(load_f)
                    
                max_value = load_dict["similarity"]
                     
                # max_value = 0

                # for key in attr_dict.keys():
                #     if attr_dict[key][1]>max_value:
                #         max_value = attr_dict[key][1]
                dist = 1-np.arccos(max_value) / math.pi

                if dist == dist:
                    scores.append(dist)
                    label.append(load_dict["match"])
    
    print("Original:")
    AUC(scores,label)

def main(args):
    lists = os.listdir(args.Json_fold)
    scores = []
    label = []

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

                if max_value < load_dict["similarity"] or load_dict["match"]==1:
                    max_value = load_dict["similarity"]
                     
                # max_value = 0

                # for key in attr_dict.keys():
                #     if attr_dict[key][1]>max_value:
                #         max_value = attr_dict[key][1]
                dist = 1-np.arccos(max_value) / math.pi

                if dist == dist:
                    scores.append(dist)
                    label.append(load_dict["match"])
    print("After ours:")
    AUC(scores,label)

def parse_args():
    parser = argparse.ArgumentParser(description='Plot AUC curve')
    # general
    parser.add_argument('--Json-fold',
                        type=str,
                        default='./Multi-ID-thresh/scores-group-VGGFace2-test-CosFace-r50-id_thresh-0.5/json',
                        # default='./Multi-ID-topk/scores-group-VGGFace2-test-CosFace-r50-topk-9/json',
                        # default='./ablation-no-counterfactual/scores-group-Celeb-A-VGGFace2-verification-topk-1-erase-black/json',
                        help='Datasets.')
    parser.add_argument('--Attribute-number',
                        type=int,
                        default=30,
                        help='Number of attribute.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    Metric(args)
    main(args)