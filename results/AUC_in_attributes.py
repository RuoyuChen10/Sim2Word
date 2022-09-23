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
from sklearn import metrics
from sklearn.metrics import auc
from prettytable import PrettyTable
from tqdm import tqdm

def AUC(score,label,x_labels,tpr_fpr_table,name):
    score = np.array(score)
    label = np.array(label)
    
    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr) # select largest tpr at same fpr

    tpr_fpr_row = []
    
    for fpr_iter in np.arange(2,len(x_labels)):
        _, min_index = min(list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.4f' % tpr[min_index])
    tpr_fpr_table.add_row([name,roc_auc]+tpr_fpr_row)

    # plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=1)

    # plt.plot([0,1], [0,1], 'r', lw=1)

    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc="lower right")

    # print(tpr_fpr_table)
    # plt.savefig('AUC.jpg',dpi=400,bbox_inches='tight')

def main(args):
    lists = os.listdir(args.Json_fold)
    distribution_data = [[[],[]] for i in range(args.Attribute_number)]

    for path in tqdm(lists):
        
        if path.split('.')[-1]=="json":
            with open(os.path.join(args.Json_fold,path),'r') as load_f:
                load_dict = json.load(load_f)
            for image_key in ["attributes-image1","attributes-image2"]:
                attr_dict = load_dict[image_key]

                for key in attr_dict.keys():
                    if load_dict["match"] ==0 and attr_dict[key][1]==attr_dict[key][1]:
                        distribution_data[int(key.split('_')[-1])][0].append(attr_dict[key][1])
                        distribution_data[int(key.split('_')[-1])][1].append(load_dict["match"])
                    elif load_dict["match"] ==1:
                        distribution_data[int(key.split('_')[-1])][0].append(load_dict["similarity"])
                        distribution_data[int(key.split('_')[-1])][1].append(load_dict["match"])
    
    x_labels = ["Attributes","roc_auc",10**-6, 10**-5, 10**-4,10**-3, 10**-2, 10**-1,0.2,0.4,0.6,0.8,1]
    tpr_fpr_table = PrettyTable(map(str, x_labels))
    
    i = 0
    for inf in distribution_data:
        AUC(inf[0],inf[1],x_labels,tpr_fpr_table,name="attribute_"+str(i))
        i += 1
    print(tpr_fpr_table)

def parse_args():
    parser = argparse.ArgumentParser(description='Plot AUC curve')
    # general
    parser.add_argument('--Json-fold',
                        type=str,
                        default='./Multi-ID-topk/scores-group-Celeb-A-VGGFace2-verification-topk-1/json',
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