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
import torchvision.transforms as transforms
from PIL import Image
from interpretability.Semantically_interpretable import Segmantically_Attributes
from utils import *

transforms = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

layer_names = [
    "module.layer1.0",
    "module.layer1.1",
    "module.layer1.2",
    "module.layer2.0",
    "module.layer2.1",
    "module.layer2.2",
    "module.layer2.3",
    "module.layer3.0",
    "module.layer3.1",
    "module.layer3.2",
    "module.layer3.3",
    "module.layer3.4",
    "module.layer3.5",
    "module.layer4.0",
    "module.layer4.1",
    "module.layer4.2",
    # "layer4.2"
    ]

image_paths = [
    "/home/cry/data2/VGGFace2/train_align_arcface/n003198/0356_01.jpg", 
    "/home/cry/data2/VGGFace2/train_align_arcface/n000538/0199_01.jpg",
    "/home/cry/data2/VGGFace2/train_align_arcface/n005919/0118_01.jpg",
    "/home/cry/data2/VGGFace2/train_align_arcface/n001052/0357_04.jpg",
    "/home/cry/data2/VGGFace2/train_align_arcface/n003299/0161_02.jpg",
    "/home/cry/data2/VGGFace2/train_align_arcface/n001255/0034_01.jpg",
    "/home/cry/data2/VGGFace2/train_align_arcface/n000220/0186_01.jpg",
    "/home/cry/data2/VGGFace2/train_align_arcface/n002921/0254_01.jpg",
    "/home/cry/data2/VGGFace2/train_align_arcface/n004681/0011_01.jpg",
    "/home/cry/data2/VGGFace2/train_align_arcface/n005390/0240_01.jpg",
    "/home/cry/data2/VGGFace2/train_align_arcface/n000888/0172_02.jpg",
    "/home/cry/data2/VGGFace2/train_align_arcface/n001871/0405_01.jpg",
    "/home/cry/data2/VGGFace2/train_align_arcface/n003511/0053_01.jpg",
    "/home/cry/data2/VGGFace2/train_align_arcface/n007993/0455_01.jpg",
    "/home/cry/data2/VGGFace2/train_align_arcface/n003474/0331_01.jpg",
    "/home/cry/data2/VGGFace2/train_align_arcface/n009238/0174_02.jpg",
    "/home/cry/data2/VGGFace2/train_align_arcface/n004177/0085_01.jpg",
    "/home/cry/data2/VGGFace2/train_align_arcface/n007313/0212_01.jpg"
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
    cv2.imwrite(save_path, cam)

def Path_Image_Preprocessing(image_path):
    data = Image.open(image_path)
    data = transforms(data)
    data = torch.unsqueeze(data,0)
    return data

def main(args):
    weight_path = os.path.join("ckpt/ArcFace-8631.pth")

    recognition_net = torch.load(weight_path)
    if torch.cuda.is_available():
        recognition_net.cuda()
    recognition_net.eval()

    mkdir("./ID_cam_visual_various_layer-ArcFace")

    for layer_name in layer_names:
        print(layer_name)
        cam = GradCAM(recognition_net, layer_name)
        for image_path in image_paths:
            image = Path_Image_Preprocessing(image_path)
            
            mask,id,scores = cam(image.cuda())  # cam mask

            save_image_dir_path = "./ID_cam_visual_various_layer-ArcFace/" + image_path.split("/")[-2] + "_" + image_path.split("/")[-1].split(".")[0]
            mkdir(save_image_dir_path)

            save_path = os.path.join(save_image_dir_path, image_path.split("/")[-2] + "_" + image_path.split("/")[-1].split(".")[0] + "-" + layer_name + ".jpg")

            save_heatmap(image_path, mask, save_path)

    return None


if __name__ == "__main__":
    args = parse_args()
    main(args)