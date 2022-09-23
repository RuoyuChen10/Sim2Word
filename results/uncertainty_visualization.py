# -*- coding: utf-8 -*-  

"""
Created on 2022/1/18

@author: Ruoyu Chen
"""

import json
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tqdm import tqdm

Face_attributes_name = np.array([
    "Gender","Age","Race","Bald","Wavy Hair",
    "Receding Hairline","Bangs","Sideburns","Hair color","no beard",
    "Mustache","5 o Clock Shadow","Goatee","Oval Face","Square Face",
    "Round Face","Double Chin","High Cheekbones","Chubby","Obstructed Forehead",
    "Fully Visible Forehead","Brown Eyes","Bags Under Eyes","Bushy Eyebrows","Arched Eyebrows",
    "Mouth Closed","Smiling","Big Lips","Big Nose","Pointy Nose"
])

save_location = np.array([  # 大部分二分类0代表是，1代表不是
    0,0,0,1,1,
    1,1,1,0,1,
    1,1,1,1,1,
    1,1,1,1,1,
    1,1,1,1,1,
    1,1,1,1,1
])

Gender = ["Male","Female"]
Age = ["Young","Middle Aged","Senior"]
Race = ["Asian","White","Black"]
Hair_color = ["Black Hair","Blond Hair","Brown Hair","Gray Hair","Unknown Hair"]

def main(args):
    """
    The main code 
    """
    # Read the main fold where save the json files (Contain the experiment results)
    json_path = args.Json_fold
    json_file_names = tqdm(os.listdir(json_path))

    # Loop to visit the file
    for json_file in json_file_names:
        # Read the content of the json file
        json_file_path = os.path.join(json_path, json_file)
        with open(json_file_path,'r') as load_f:
            load_dict = json.load(load_f)
        
        # Select the detected attributes
        idx = (save_location * load_dict["predicted_attribute"] + 1) * (np.array(load_dict["predicted_attribute_score"])>0.8).astype(np.int)
        idx = np.where(idx==1)[0][:,np.newaxis]
        idx = np.where(((np.array(load_dict["uncertain_indices"])==idx).astype(int)).sum(axis=0)==1)[0]

        # Choose the corresponding attribute and uncertainty value
        attribute = np.array(load_dict["attributes_sortting"])[idx]
        uncertain_value = np.array(load_dict["uncertain_value"])[idx]

        # Modify the precise attribute name
        attribute[attribute=="Gender"] = Gender[load_dict["predicted_attribute"][0]]
        attribute[attribute=="Age"] = Age[load_dict["predicted_attribute"][1]]
        attribute[attribute=="Race"] = Race[load_dict["predicted_attribute"][2]]
        attribute[attribute=="Hair color"] = Hair_color[load_dict["predicted_attribute"][8]]

        # Open the image file
        image = Image.open(load_dict["image_path"])

        # Append the original image uncertainty
        attribute = np.insert(attribute,0,"Original")
        uncertain_value = np.insert(uncertain_value,0,load_dict["uncertain_original"])

        # Plt the image
        fig, [ax1, ax2] = plt.subplots(1,2, gridspec_kw = {'width_ratios':[1, 4]}, figsize=(30,8))
        ax1.spines["left"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)
        ax1.imshow(image)

        ax2.bar(attribute[0], uncertain_value[0],color="green",alpha = 0.3)
        ax2.bar(attribute[1:], uncertain_value[1:],color="royalblue",alpha = 0.7)
        # ax2.bar(attribute[0], uncertain_value[0], color="green",alpha = 0.2)
        ax2.set_xticklabels(attribute,fontsize=24,rotation=60)
        # plt.ylim((round(uncertain_value.min(), 4)-1e-4, round(uncertain_value.max(), 4)+1e-4))
        plt.ylim((0,0.5))
        plt.yticks(fontsize=26)
        plt.ylabel("Uncertainty",fontsize=32)

        plt.savefig(os.path.join(args.Save_fold, json_file.replace(".json",".jpg")),bbox_inches='tight')

def parse_args():
    parser = argparse.ArgumentParser(description='Plot AUC curve')
    # general
    parser.add_argument('--Json-fold',
                        type=str,
                        # default='./single-person-uncertain/Json',
                        default = "single-person-uncertain-scale-200-new/Json",
                        help='Datasets.')
    parser.add_argument('--Save-fold',
                        type=str,
                        # default='./single-person-uncertain/Image',
                        default = "single-person-uncertain-scale-200-new/Image",
                        help='save dir.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)