# -*- coding: utf-8 -*-  

"""
Created on 2021/05/06
Update  on 2021/05/27

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

from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from Datasets.dataload import Path_Image_Preprocessing, Image_Preprocessing
from interpretability.Semantically_interpretable import Segmantically_Attributes
from utils import *

def Read_Datasets_v(dataset_list):
    '''
    Read the path in different datasets
    '''
    # dataset_list = "./External-Experience/The_most_special_attribute/List.txt"
    if os.path.exists(dataset_list):
        with open(dataset_list, "r") as f:
            datas = f.read().split('\n')
    else:
        raise ValueError("File {} not in path".format(dataset_list))
    return datas

def get_last_conv_name(net):
    """
    Get the name of last convolutional layer
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name

def get_heatmap_method(net, method="GradCAM"):
    '''
    Get the method to generate heatmap
    '''
    layer_name = get_last_conv_name(net)
    if method == "GradCAM": 
        cam = GradCAM(net, layer_name)
    elif method == "GradCAM++":
        cam = GradCamPlusPlus(net, layer_name)
    return cam

def gen_cam(image_dir, mask):
    """
    Generate heatmap
        :param image: [H,W,C]
        :param mask: [H,W],range 0-1
        :return: tuple(cam,heatmap)
    """
    # Read image
    image = cv2.imread(image_dir)
    # mask->heatmap
    num = mask.shape[0]
    for i in range(0,num):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask[i]), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap)

        # merge heatmap to original image
        cam = 0.5*heatmap + 0.5*np.float32(image)
    return cam, (heatmap).astype(np.uint8)

def make_save_dir(args):
    '''
    Make dir the save dir
    '''
    if args.type_choose == 'topk':
        mkdir(os.path.join(args.output_dir,"CAM",
            "scores-group-"+args.Datasets+"-"+args.heatmap_method+"-topk-"+str(args.topk)))
        mkdir(os.path.join(args.output_dir,"CAM",
            "scores-group-"+args.Datasets+"-"+args.heatmap_method+"-topk-"+str(args.topk),
            "Identity-CAM"))
        mkdir(os.path.join(args.output_dir,"CAM",
            "scores-group-"+args.Datasets+"-"+args.heatmap_method+"-topk-"+str(args.topk),
            "Attribute-CAM"))
    elif args.type_choose == 'id-thresh':
        mkdir(os.path.join(args.output_dir,"CAM",
            "scores-group-"+args.Datasets+"-"+args.heatmap_method+"-id_thresh-"+str(args.id_thresh)))
        mkdir(os.path.join(args.output_dir,"CAM",
            "scores-group-"+args.Datasets+"-"+args.heatmap_method+"-id_thresh-"+str(args.id_thresh),
            "Identity-CAM"))
        mkdir(os.path.join(args.output_dir,"CAM",
            "scores-group-"+args.Datasets+"-"+args.heatmap_method+"-id_thresh-"+str(args.id_thresh),
            "Attribute-CAM"))
    return None

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
    masks = norm_image(mask).astype(np.uint8)
    # mask->heatmap
    heatmap = cv2.applyColorMap(masks, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    # merge heatmap to original image
    cam = 0.5*heatmap + 0.5*np.float32(image)
    cv2.imwrite(save_path,cam)

def save_attributes_heatmap(image_input,masks,save_dir_path,heatmap_method):
    '''
    Save the attributes heatmap
    '''
    num = masks.shape[0]
    for i in range(0,num):
        save_heatmap(image_input,masks[i],os.path.join(save_dir_path,"Attribute-"+str(i)+'-'+Face_attributes_name[i]+'-'+heatmap_method+'.jpg'))

def main(args):
    # Path save
    make_save_dir(args)

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

    # Heatmap
    cam1 = get_heatmap_method(recognition_net, method=args.heatmap_method)
    cam2 = get_heatmap_method(attribute_net, method=args.heatmap_method)

    # Iterate through each piece of data
    if args.Datasets == 'VGGFace2-train':
        image_dir_path = "Verification/dataset/VGGFace2-train"
    elif args.Datasets == "VGGFace2-test":
        image_dir_path = "Verification/dataset/VGGFace2-test"
    elif args.Datasets == "Celeb-A":
        image_dir_path = "Verification/dataset/CelebA-test"
    
    if args.type_choose == 'topk':
        id_cam_save_path = os.path.join(args.output_dir,"CAM",
            "scores-group-"+args.Datasets+"-"+args.heatmap_method+"-topk-"+str(args.topk),
            "Identity-CAM")
        arrt_cam_save_path = os.path.join(args.output_dir,"CAM",
            "scores-group-"+args.Datasets+"-"+args.heatmap_method+"-topk-"+str(args.topk),
            "Attribute-CAM")
    elif args.type_choose == 'id-thresh':
        id_cam_save_path = os.path.join(args.output_dir,"CAM",
            "scores-group-"+args.Datasets+"-"+args.heatmap_method+"-id_thresh-"+str(args.id_thresh),
            "Identity-CAM")
        arrt_cam_save_path = os.path.join(args.output_dir,"CAM",
            "scores-group-"+args.Datasets+"-"+args.heatmap_method+"-id_thresh-"+str(args.id_thresh),
            "Attribute-CAM")

    num = 1
    for data in tqdm(datas):
        try:
            # Get path
            path1 = os.path.join(os.getcwd(),image_dir_path,data.split(' ')[0])
            path2 = os.path.join(os.getcwd(),image_dir_path,data.split(' ')[1])

            # Read Image
            image1 = Path_Image_Preprocessing("VGGFace2",path1)
            image2 = Path_Image_Preprocessing("VGGFace2",path2)
            
            mask1,id_1,scores1 = cam1(torch.tensor([image1], requires_grad=True).cuda())  # cam mask
            mask2,id_2,scores2 = cam1(torch.tensor([image2], requires_grad=True).cuda())  # cam mask

            # save mask
            save_heatmap(path1, mask1, os.path.join(id_cam_save_path,"n"+str(num)+"-1.jpg"))
            save_heatmap(path2, mask2, os.path.join(id_cam_save_path,"n"+str(num)+"-2.jpg"))

            # Attribute mask
            mask3,id_3,scores3 = cam2(torch.tensor([image1], requires_grad=True).cuda())  # cam mask
            mask4,id_4,scores4 = cam2(torch.tensor([image2], requires_grad=True).cuda())  # cam mask
            
            mkdir(os.path.join(arrt_cam_save_path,"n"+str(num)+"-1"))
            mkdir(os.path.join(arrt_cam_save_path,"n"+str(num)+"-2"))

            save_attributes_heatmap(path1,mask3,os.path.join(arrt_cam_save_path,"n"+str(num)+"-1"),args.heatmap_method)
            save_attributes_heatmap(path2,mask4,os.path.join(arrt_cam_save_path,"n"+str(num)+"-2"),args.heatmap_method)
        except:
            pass
        num += 1

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Interpretability Test')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        choices=['VGGFace2-train','VGGFace2-test','Celeb-A'],
                        default='VGGFace2-test',
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
    parser.add_argument('--heatmap-method', type=str, default='GradCAM',
                        choices=['GradCAM','GradCAM++'],
                        help='Attribute network.')
    parser.add_argument('--thresh', type=float, default=0.6,
                        help='Thresh.')
    parser.add_argument('--type-choose', type=str, default='topk',
                        choices=['topk','id-thresh'],
                        help='Which type choose, topk or id-thresh.')
    parser.add_argument('--topk', type=int, default=1,
                        help='Top k classes.')
    parser.add_argument('--id-thresh', type=float, default=0.2,
                        help='Thresh for choose topk people.')
    parser.add_argument('--output-dir', type=str, default='./results/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)