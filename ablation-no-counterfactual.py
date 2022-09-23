# -*- coding: utf-8 -*-  

"""
Created on 2021/04/27
Update  on 2021/05/09   Add Erasing method

@author: Ruoyu Chen
不加counterfacutal map，只有属性
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

def Erasing(image,mask,method):
    '''
    image: (H,W,3)
    mask: (H,W) saved pixels set 1
    method: black, white, mean, random
    '''
    if method == "black":
        image_mask = (image.transpose(2,0,1)*mask).transpose(1,2,0)
    elif method == "white":
        mask_white = np.ones(mask.shape)*(1-mask)*255
        image_mask = (image.transpose(2,0,1)*mask+mask_white).transpose(1,2,0)
    elif method == "mean":
        mask_mean = np.ones(mask.shape)*(1-mask)*125
        image_mask = (image.transpose(2,0,1)*mask+mask_mean).transpose(1,2,0)
    elif method == "random":
        mask_random = np.ones(mask.shape)*(1-mask) * np.random.randint(0,256,(3,mask.shape[0],mask.shape[1]))
        mask_random = mask_random.transpose(1,2,0)
        image_mask = (image.transpose(2,0,1)*mask).transpose(1,2,0) + mask_random
    return image_mask

def main(args):
    # Path save
    mkdir(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method))
    mkdir(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"json"))
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
    
    # Load Verification Network
    verification_net = get_network(args.verification_net)   # Already cuda() and eval() operation

    # Iterate through each piece of data
    if args.Datasets == 'VGGFace2-train':
        image_dir_path = "Verification/dataset/VGGFace2-train"
    elif args.Datasets == "VGGFace2-test":
        image_dir_path = "Verification/dataset/VGGFace2-test"
    elif args.Datasets == "Celeb-A":
        image_dir_path = "Verification/dataset/CelebA-test"
    elif args.Datasets == "Test":
        image_dir_path = "External-Experience/The_most_special_attribute/dataset/VGGFace2-train"
    
    num = 1
    for data in tqdm(datas):
        try:
            scores = {}
            scores["Datasets"] = args.Datasets
            scores["Heatmap-method"] = args.heatmap_method
            scores["thresh"] = args.thresh
            scores["topk"] = args.topk
            scores["verification-net"] = args.verification_net

            # Get path
            path1 = os.path.join(os.getcwd(),image_dir_path,data.split(' ')[0])
            path2 = os.path.join(os.getcwd(),image_dir_path,data.split(' ')[1])
            if_same = data.split(' ')[2]

            # Read Image
            image1 = Path_Image_Preprocessing("VGGFace2",path1).numpy()
            image2 = Path_Image_Preprocessing("VGGFace2",path2).numpy()

            scores["image-path1"] = path1
            scores["image-path2"] = path2
            scores["match"] = int(if_same)
            
            # Verification image
            image1_ = Path_Image_Preprocessing(args.verification_net,path1)
            image2_ = Path_Image_Preprocessing(args.verification_net,path2)
            
            feature1 = F.normalize(verification_net(torch.unsqueeze(image1_, dim=0).cuda()),p=2,dim=1)
            feature2 = F.normalize(verification_net(torch.unsqueeze(image2_, dim=0).cuda()),p=2,dim=1)

            scores["similarity"] = torch.cosine_similarity(feature1[0], feature2[0], dim=0).item()

            ##### Mask Game #####
            # Image1 attributes
            seg_attr_interpretable1, attribute_id1, scores_attr1 = seg_attr.ablation_no_counterfactual(image1)
            seg_attr_interpretable1 = 1 - seg_attr_interpretable1

            # Image2 attributes
            seg_attr_interpretable2, attribute_id2, scores_attr2 = seg_attr.ablation_no_counterfactual(image2)
            seg_attr_interpretable2 = 1 - seg_attr_interpretable2

            scores["Attribute1-class"] = attribute_id1
            scores["Attribute1-score"] = scores_attr1
            scores["Attribute2-class"] = attribute_id2
            scores["Attribute2-score"] = scores_attr2

            # Dict for the game
            attr1 = {}
            attr2 = {}
            attr3 = {}

            image_1 = cv2.imread(path1)
            image_2 = cv2.imread(path2)

            seg_attr_interpretable1[seg_attr_interpretable1<1-args.thresh] = 0
            seg_attr_interpretable1[seg_attr_interpretable1>1-args.thresh] = 1

            seg_attr_interpretable2[seg_attr_interpretable2<1-args.thresh] = 0
            seg_attr_interpretable2[seg_attr_interpretable2>1-args.thresh] = 1

            for i in range(0,seg_attr_interpretable1.shape[0]):
                feature_attr_1 = None
                feature_attr_2 = None
                # No nan of image 1:

                Erasing_image = Erasing(image_1,seg_attr_interpretable1[i],args.Erasing_method)

                if args.visualization == True:
                    mkdir(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",str(num)+"-n1-n2","mask"))
                    mkdir(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",str(num)+"-n1-n2","images"))
                    # Save the Visualization Results
                    cv2.imwrite(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",
                                                str(num)+"-n1-n2","mask","attribute"+str(i)+'.jpg'),
                                seg_attr_interpretable1[i]*255)
                    cv2.imwrite(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",
                                                str(num)+"-n1-n2","images","attribute-o"+str(i)+'.jpg'),
                                Erasing_image)

                if args.calculate_similarity == True:
                    inputs = Image_Preprocessing(args.verification_net, Erasing_image)
                    feature_attr_1 = F.normalize(verification_net(torch.unsqueeze(inputs, dim=0).cuda()),p=2,dim=1)

                    attr1["image1-f_attr_"+str(i)] = (torch.cosine_similarity(feature_attr_1[0], feature1[0], dim=0).item(),
                                                    torch.cosine_similarity(feature_attr_1[0], feature2[0], dim=0).item())
                # No nan of image 2:
                
                Erasing_image = Erasing(image_2,seg_attr_interpretable2[i],args.Erasing_method)

                if args.visualization == True:
                    mkdir(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",str(num)+"-n2-n1","mask"))
                    mkdir(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",str(num)+"-n2-n1","images"))
                    # Save the Visualization Results
                    cv2.imwrite(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",
                                                str(num)+"-n2-n1","mask","attribute"+str(i)+'.jpg'),
                                seg_attr_interpretable2[i]*255)
                    cv2.imwrite(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"cam",
                                                str(num)+"-n2-n1","images","attribute-o"+str(i)+'.jpg'),
                                Erasing_image)

                if args.calculate_similarity == True:
                    inputs = Image_Preprocessing(args.verification_net,Erasing_image)
                    feature_attr_2 = F.normalize(verification_net(torch.unsqueeze(inputs, dim=0).cuda()),p=2,dim=1)

                    attr2["image2-f_attr_"+str(i)] = (torch.cosine_similarity(feature_attr_2[0], feature2[0], dim=0).item(),
                                                    torch.cosine_similarity(feature_attr_2[0], feature1[0], dim=0).item())
                
                # For both masked
                if feature_attr_1 is not None and feature_attr_2 is not None:
                    if args.calculate_similarity == True:
                        attr3["Both_masked-f_attr_"+str(i)] = torch.cosine_similarity(feature_attr_1[0], feature_attr_2[0], dim=0).item()
            
            if args.calculate_similarity == True:
                scores["attributes-image1"] = attr1
                scores["attributes-image2"] = attr2
                scores["attributes-both-masked"] = attr3

            with open(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net+"-topk-"+str(args.topk)+"-erase-"+args.Erasing_method,"json",str(num)+".json"), "w") as f:
                f.write(json.dumps(scores, ensure_ascii=False, indent=4, separators=(',', ':')))
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
                        default='CosFace-r100',
                        choices=['ArcFace-r50','ArcFace-r100',"CosFace-r50","CosFace-r100","VGGFace2-verification"],
                        help='Which network using for face verification.')
    parser.add_argument('--heatmap-method', type=str, default='GradCAM',
                        choices=['GradCAM','GradCAM++'],
                        help='Attribute network.')
    parser.add_argument('--calculate-similarity',type=bool,default=True,
                        help="If compute the similarity of the images with masked images")
    parser.add_argument('--visualization',type=bool,default=True,
                        help="If compute the similarity of the images with masked images")
    parser.add_argument('--thresh', type=float, default=0.8,
                        help='Thresh.')
    parser.add_argument('--topk', type=int, default=1,
                        help='Top k classes.')
    parser.add_argument('--Erasing-method', type=str, default="black",
                        choices=["black","white","mean","random"],
                        help='Which method to erasing.')
    parser.add_argument('--output-dir', type=str, default='./results/ablation-no-counterfactual',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)