# -*- coding: utf-8 -*-  

"""
Created on 2021/4/8

@author: Ruoyu Chen

Note: Don't use
"""

import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np
import json

import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

import Datasets.dataload as dl
from Semantically_interpretable import Segmantically_Attributes
from utils import *

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

def Path_Image_Preprocessing(net_type,path):
    '''
    Precessing the input images
        image_dir: single image input path, such as "/home/xxx/10.jpg"
    '''
    if net_type in ['ArcFace-r50','ArcFace-r100',"CosFace-r50","CosFace-r100"]:
        return transforms(Image.open(path).resize((112, 112), Image.BILINEAR))
    elif net_type in ["VGGFace2","VGGFace2-verification"]:
        mean_bgr = np.array([91.4953, 103.8827, 131.0912])
        image = cv2.imread(path)
        assert image is not None
        image = cv2.resize(image,(224,224))
        image = image.astype(np.float32)
        image -= mean_bgr
        # H * W * C   -->   C * H * W
        image = image.transpose(2,0,1)
        return torch.tensor(image)

def Image_Preprocessing(net_type,image):
    '''
    Precessing the input images
        image_dir: single image input path, such as "/home/xxx/10.jpg"
    '''
    if net_type in ['ArcFace-r50','ArcFace-r100',"CosFace-r50","CosFace-r100"]:
        image = Image.fromarray(cv2.cvtColor(np.uint8(image),cv2.COLOR_BGR2RGB))
        return transforms(image.resize((112, 112), Image.BILINEAR))
    elif net_type in ["VGGFace2","VGGFace2-verification"]:
        mean_bgr = np.array([91.4953, 103.8827, 131.0912])
        assert image is not None
        image = cv2.resize(image,(224,224))
        image = image.astype(np.float32)
        image -= mean_bgr
        # H * W * C   -->   C * H * W
        image = image.transpose(2,0,1)
        return torch.tensor(image)

def Read_Datasets(Datasets_type):
    '''
    Read the path in different datasets
    '''
    if Datasets_type == "VGGFace2":
        if os.path.exists("./Verification/VGGFace2.txt"):
            with open("./Verification/VGGFace2.txt", "r") as f:
                datas = f.read().split('\n')
        else:
            raise ValueError("File {} not in path".format("./Verification/VGGFace2.txt"))
    elif Datasets_type == "Celeb-A":
        if os.path.exists("./Verification/CelebA.txt"):
            with open("./Verification/CelebA.txt", "r") as f:
                datas = f.read().split('\n')
        else:
            raise ValueError("File {} not in path".format("./Verification/CelebA.txt"))
    return datas

def main(args):
    # Path save
    mkdir(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net))
    
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
    seg_attr = Segmantically_Attributes(recognition_net,attribute_net)
    
    # Load Verification Network
    verification_net = get_network(args.verification_net)   # Already cuda() and eval() operation
    # if torch.cuda.is_available():
    #     verification_net.cuda()
    # verification_net.eval()

    # Iterate through each piece of data
    if args.Datasets == "VGGFace2":
        image_dir_path = "Verification/VGGFace2-test"
    elif args.Datasets == "Celeb-A":
        image_dir_path = "Verification/CelebA-test"
    num = 1
    for data in tqdm(datas):
        try:
            scores = {}
            scores["Datasets"] = args.Datasets
            scores["Heatmap-method"] = args.heatmap_method
            scores["thresh"] = args.thresh
            scores["verification-net"] = args.verification_net

            # Get path
            path1 = os.path.join(os.getcwd(),image_dir_path,data.split(' ')[0])
            path2 = os.path.join(os.getcwd(),image_dir_path,data.split(' ')[1])
            if_same = data.split(' ')[2]

            # Read Image
            image1 = dl.Image_precessing(path1)
            image2 = dl.Image_precessing(path2)

            scores["image-path1"] = path1
            scores["image-path2"] = path2
            scores["match"] = int(if_same)

            # Get class
            out = recognition_net(torch.tensor([image1,image2]).cuda())
            class1 = torch.argmax(out[0])
            class2 = torch.argmax(out[1])

            scores["class1"] = class1.item()
            scores["class2"] = class2.item()
            
            # Verification image
            image1_ = Path_Image_Preprocessing(args.verification_net,path1)
            image2_ = Path_Image_Preprocessing(args.verification_net,path2)
            
            feature1 = F.normalize(verification_net(torch.unsqueeze(image1_, dim=0).cuda()),p=2,dim=1)
            feature2 = F.normalize(verification_net(torch.unsqueeze(image2_, dim=0).cuda()),p=2,dim=1)

            scores["similarity"] = torch.cosine_similarity(feature1[0], feature2[0], dim=0).item()

            # Image1 as first
            seg_attr_interpretable = seg_attr.Segmantically_Attributes_Interpretable(image1,class2,args.heatmap_method)
            seg_attr_interpretable = 1 - seg_attr_interpretable

            attr = {}
            image = cv2.imread(path1)
            for i in range(0,seg_attr_interpretable.shape[0]):
                # No nan
                if np.max(seg_attr_interpretable[i]) == np.max(seg_attr_interpretable[i]):
                    # thresh
                    seg_attr_interpretable[i][seg_attr_interpretable[i]<1-args.thresh] = 0
                    seg_attr_interpretable[i][seg_attr_interpretable[i]>1-args.thresh] = 1

                    inputs = Image_Preprocessing(args.verification_net,(image.transpose(2,0,1)*seg_attr_interpretable[i]).transpose(1,2,0))
                    feature = F.normalize(verification_net(torch.unsqueeze(inputs, dim=0).cuda()),p=2,dim=1)

                    attr["image1-f_attr_"+str(i)] = (torch.cosine_similarity(feature[0], feature1[0], dim=0).item(),
                                                    torch.cosine_similarity(feature[0], feature2[0], dim=0).item())
            scores["attributes-image1"] = attr

            # Image2 as first
            seg_attr_interpretable = seg_attr.Segmantically_Attributes_Interpretable(image2,class1,args.heatmap_method)
            seg_attr_interpretable = 1 - seg_attr_interpretable
            
            attr = {}
            image = cv2.imread(path2)
            for i in range(0,seg_attr_interpretable.shape[0]):
                # No nan
                if np.max(seg_attr_interpretable[i]) == np.max(seg_attr_interpretable[i]):
                    # thresh
                    seg_attr_interpretable[i][seg_attr_interpretable[i]<1-args.thresh] = 0
                    seg_attr_interpretable[i][seg_attr_interpretable[i]>1-args.thresh] = 1

                    inputs = Image_Preprocessing(args.verification_net,(image.transpose(2,0,1)*seg_attr_interpretable[i]).transpose(1,2,0))
                    
                    feature = F.normalize(verification_net(torch.unsqueeze(inputs, dim=0).cuda()),p=2,dim=1)

                    attr["image2-f_attr_"+str(i)] = (torch.cosine_similarity(feature[0], feature2[0], dim=0).item(),
                                                    torch.cosine_similarity(feature[0], feature1[0], dim=0).item())
            scores["attributes-image2"] = attr
            with open(os.path.join(args.output_dir,"scores-group-"+args.Datasets+"-"+args.verification_net,str(num)+".json"), "w") as f:
                f.write(json.dumps(scores, ensure_ascii=False, indent=4, separators=(',', ':')))
            num += 1
        except:
            pass

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Interpretability Test')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        choices=['VGGFace2','Celeb-A'],
                        default='Celeb-A',
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
                        default='CosFace-r50',
                        choices=['ArcFace-r50','ArcFace-r100',"CosFace-r50","CosFace-r100","VGGFace2-verification"],
                        help='Which network using for face verification.')
    parser.add_argument('--heatmap-method', type=str, default='GradCAM',
                        choices=['GradCAM','GradCAM++'],
                        help='Attribute network.')
    parser.add_argument('--thresh', type=float, default=0.6,
                        help='Thresh.')
    parser.add_argument('--output-dir', type=str, default='./results/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)