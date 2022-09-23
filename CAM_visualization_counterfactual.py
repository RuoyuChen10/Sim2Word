# -*- coding: utf-8 -*-  

"""
Created on 2021/04/27
Update  on 2021/05/09   Add Erasing method

@author: Ruoyu Chen
"""

import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np
import json

import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from Datasets.dataload import Path_Image_Preprocessing,Image_Preprocessing
from interpretability.Semantically_interpretable import Segmantically_Attributes
from utils import *

def norm_image(image):
    """
    Normalization image
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def save_heatmap(image_input,mask,save_path):
    '''
    Save the heatmap of ones
    '''
    # Read image
    image = cv2.imread(image_input)
    # print(np.max(mask))
    
    masks = norm_image(np.array(mask)).astype(np.uint8)

    # mask->heatmap
    heatmap = cv2.applyColorMap(masks, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    # merge heatmap to original image
    cam = 0.5*heatmap + 0.5*np.float32(image)
    cv2.imwrite(save_path,cam)

def main(args):
    # Path save
    mkdir(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method))
    mkdir(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam"))
    
    # Read image path
    datas = Read_Datasets(args.Datasets)

    # Load Recognition Network
    recognition_net = get_network(args.recognition_net)
    if torch.cuda.is_available():
        recognition_net.cuda()
    recognition_net.eval()

    # Load Attributes Network
    attribute_net = get_network(None,args.attribute_net)
    if torch.cuda.is_available():
        attribute_net.cuda()
    attribute_net.eval()
    seg_attr = Segmantically_Attributes(recognition_net,attribute_net,args.heatmap_method)

    # Iterate through each piece of data
    if args.Datasets == 'VGGFace2-train':
        image_dir_path = "Verification/dataset/VGGFace2-train"
    elif args.Datasets == "VGGFace2-test":
        image_dir_path = "Verification/dataset/VGGFace2-test"
    elif args.Datasets == "Celeb-A":
        image_dir_path = "Verification/dataset/CelebA-test"
    
    num = 1
    for data in tqdm(datas):
        try:
            # Get path
            path1 = os.path.join(os.getcwd(),image_dir_path,data.split(' ')[0])
            path2 = os.path.join(os.getcwd(),image_dir_path,data.split(' ')[1])

            # Read Image
            image1 = Path_Image_Preprocessing("VGGFace2",path1).numpy()
            image2 = Path_Image_Preprocessing("VGGFace2",path2).numpy()

            ##### Mask Game #####
            # Image1 attributes
            seg_attr_interpretable1, mask1_1, mask1_2 = seg_attr.ablation_no_attribute_v2(image1,image2)

            # Image2 attributes
            seg_attr_interpretable2, mask2_1, mask2_2 = seg_attr.ablation_no_attribute_v2(image2,image1)

            if args.visualization == True:
                mkdir(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",str(num)+"-n1-n2","heatmap-id"))
                mkdir(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",str(num)+"-n1-n2","heatmap-counter"))

                # Save the Visualization Results
                save_heatmap(path1,mask1_1,os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",str(num)+"-n1-n2","heatmap-id","real.jpg"))
                save_heatmap(path1,mask1_2,os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",str(num)+"-n1-n2","heatmap-id","counter.jpg"))
                save_heatmap(path1,1-mask1_2,os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",str(num)+"-n1-n2","heatmap-id","counter-f.jpg"))
                save_heatmap(path1,seg_attr_interpretable1,os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",str(num)+"-n1-n2","heatmap-counter","heatmap.jpg"))

            if args.visualization == True:
                mkdir(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",str(num)+"-n2-n1","heatmap-id"))
                mkdir(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",str(num)+"-n2-n1","heatmap-counter"))
                
                # Save the Visualization Results
                save_heatmap(path2,mask2_1,os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",str(num)+"-n2-n1","heatmap-id","real.jpg"))
                save_heatmap(path2,mask2_2,os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",str(num)+"-n2-n1","heatmap-id","counter.jpg"))
                save_heatmap(path2,1-mask2_2,os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",str(num)+"-n2-n1","heatmap-id","counter-f.jpg"))
                save_heatmap(path2,seg_attr_interpretable2,os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",str(num)+"-n2-n1","heatmap-counter","heatmap.jpg"))    
        except:
            pass
        num += 1

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Interpretability Test')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        choices=['VGGFace2-train','VGGFace2-test','Celeb-A','Test'],
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
                        default='ArcFace-r100',
                        choices=['ArcFace-r50','ArcFace-r100',"CosFace-r50","CosFace-r100","VGGFace2-verification"],
                        help='Which network using for face verification.')
    parser.add_argument('--heatmap-method', type=str, default='GradCAM',
                        choices=['GradCAM','GradCAM++'],
                        help='Attribute network.')
    parser.add_argument('--calculate-similarity',type=bool,default=True,
                        help="If compute the similarity of the images with masked images")
    parser.add_argument('--visualization',type=bool,default=True,
                        help="If compute the similarity of the images with masked images")
    parser.add_argument('--thresh', type=float, default=0.6,
                        help='Thresh.')
    parser.add_argument('--topk', type=int, default=1,
                        help='Top k classes.')
    parser.add_argument('--Erasing-method', type=str, default="black",
                        choices=["black","white","mean","random"],
                        help='Which method to erasing.')
    parser.add_argument('--output-dir', type=str, default='./results/ablation-no-attribute',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)