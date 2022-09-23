# -*- coding: utf-8 -*-  

"""
Created on 2021/4/9

@author: Ruoyu Chen
"""

import json
import os
import numpy as np
import argparse

def Read_Json():
    score_group_path = "./scores-group-CelebA"
    lists = os.listdir(score_group_path)

    distribution_data = np.zeros(30)

    for path in lists:
        
        if path.split('.')[-1]=="json":
            with open(os.path.join(score_group_path,path),'r') as load_f:
                load_dict = json.load(load_f)
            for image_key in ["attributes-image1","attributes-image2"]:
                similarity = load_dict["similarity"]
                attr_dict = load_dict[image_key]

                max_value = 0
                max_index = None

                for key in attr_dict.keys():
                    if attr_dict[key][1]>max_value:
                        max_value = attr_dict[key][1]
                        max_index = int(key.split('_')[-1])
                if max_value>similarity:
                    distribution_data[max_index]+=1
    np.savetxt('task.txt', distribution_data, fmt="%d")
                

def main(args):
    if args.Task == 'json':
        Read_Json()

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Verification List')
    # general

    parser.add_argument('--Task',
                        type=str,
                        default='json',
                        choices=["json"],
                        help='Which taskes to choose.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)