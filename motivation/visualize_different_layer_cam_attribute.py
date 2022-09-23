# -*- coding: utf-8 -*-  

"""
Created on 2021/09/29

@author: Ruoyu Chen
"""

import argparse
import cv2
import numpy as np

import sys
sys.path.append("../")

from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from Datasets.dataload import Path_Image_Preprocessing, Image_Preprocessing
from interpretability.Semantically_interpretable import Segmantically_Attributes
from utils import *

layer_names = [
    "layer1.0",
    "layer1.1",
    "layer1.2",
    "layer2.0",
    "layer2.1",
    "layer2.2",
    "layer2.3",
    "layer3.0",
    "layer3.1",
    "layer3.2",
    "layer3.3",
    "layer3.4",
    "layer3.5",
    "layer4.0",
    "layer4.1",
    "layer4.2",
    "layer4.2"
    ]

image_paths = [
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n003198/0356_01.jpg", 
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n000538/0199_01.jpg",
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n005919/0118_01.jpg",
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n001052/0357_04.jpg",
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n003299/0161_02.jpg",
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n001255/0034_01.jpg",
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n000220/0186_01.jpg",
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n002921/0254_01.jpg",
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n004681/0011_01.jpg",
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n005390/0240_01.jpg",
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n000888/0172_02.jpg",
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n001871/0405_01.jpg",
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n003511/0053_01.jpg",
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n007993/0455_01.jpg",
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n003474/0331_01.jpg",
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n009238/0174_02.jpg",
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n004177/0085_01.jpg",
    "/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-train/n007313/0212_01.jpg"
]

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
                        default='/home/cry/data1/Counterfactual_interpretable/pre-trained/Face-Attributes2.pth',
                        help='Attribute network, name or path.')
    parser.add_argument('--heatmap-method', type=str, default='GradCAM',
                        choices=['GradCAM','GradCAM++'],
                        help='Attribute network.')

    parser.add_argument('--output-dir', type=str, default='./results/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

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

def save_attributes_heatmap(image_input, masks, save_dir_path,layer_name):
    '''
    Save the attributes heatmap
    '''
    num = masks.shape[0]
    for i in range(0,num):
        save_attribute_dir_path = os.path.join(save_dir_path, Face_attributes_name[i])
        mkdir(save_attribute_dir_path)

        save_path = os.path.join(save_attribute_dir_path, image_input.split("/")[-2] + "_" + image_input.split("/")[-1].split(".")[0] + "-" + Face_attributes_name[i] + "-" + layer_name + ".jpg")

        save_heatmap(image_input, masks[i], save_path)

def main(args):

    # Load Attributes Network
    attribute_net = get_network(None,args.attribute_net)
    if torch.cuda.is_available():
        attribute_net.cuda()
    attribute_net.eval()

    mkdir("./ID_cam_visual_various_layer")

    for layer_name in layer_names:
        cam = GradCAM(attribute_net, layer_name)
        for image_path in image_paths:
            image = Path_Image_Preprocessing("VGGFace2", image_path)
            
            mask, id, scores = cam(torch.tensor([image.numpy()], requires_grad=True).cuda())  # cam mask

            save_image_dir_path = "./Attribute_cam_visual_various_layer/" + image_path.split("/")[-2] + "_" + image_path.split("/")[-1].split(".")[0]
            mkdir(save_image_dir_path)

            save_attributes_heatmap(image_path, mask, save_image_dir_path, layer_name)


    return None

if __name__ == "__main__":
    args = parse_args()
    main(args)
