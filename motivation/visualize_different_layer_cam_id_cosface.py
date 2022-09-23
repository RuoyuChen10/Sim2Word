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
from interpretability.Semantically_interpretable import Segmantically_Attributes
from utils import *

import torchvision.transforms as transforms
from PIL import Image

transforms = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Interpretability Test')
    # general
    parser.add_argument('--recognition-net',
                        type=str,
                        default="CosFace-8631",
                        choices=["ArcFace-8631", "ArcFace-5000", "ArcFace-5000-reduce","CosFace-8631"],
                        help='Face identity recognition network.')
    parser.add_argument('--heatmap-method', type=str, default='GradCAM',
                        choices=['GradCAM','GradCAM++'],
                        help='Attribute network.')
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

def Path_Image_Preprocessing_(image_path):
    data = Image.open(image_path)
    data = transforms(data)
    data = torch.unsqueeze(data,0)
    return data

def main(args):

    if args.recognition_net == "ArcFace-8631":
        weight_path = os.path.join("ckpt/ArcFace-8631.pth")
    elif args.recognition_net == "ArcFace-5000":
        weight_path = os.path.join("ckpt/vggface-5000id.pth")
    elif args.recognition_net == "ArcFace-5000-reduce":
        weight_path = os.path.join("ckpt/vggface-5000id-reduce.pth")
    elif args.recognition_net == "CosFace-8631":
        weight_path = os.path.join("ckpt/CosFace-8631.pth")
        
    recognition_net = torch.load(weight_path)
    layer_name = "layer4.2"
    root_dir = "/home/cry/data2/VGGFace2/train_align_arcface"

    if torch.cuda.is_available():
        recognition_net.cuda()
    recognition_net.eval()

    with open("tes.txt", "r") as f:
        image_paths = f.readlines()

    cam = GradCAM(recognition_net, layer_name)
    for image_path in image_paths:
        gt_id = int(image_path.split(" ")[-1].replace("\n", ""))
        image_path = image_path.split(" ")[0]
        image_path = os.path.join(root_dir, image_path)


        image = Path_Image_Preprocessing_(image_path)
        mask,id,scores = cam(image.cuda())  # cam mask

        print("Predict ID:{}, Ground Truth ID:{}".format(id, gt_id))
        
        save_image_dir_path = os.path.join("./ID_cam_various_arch",args.recognition_net)

        mkdir(save_image_dir_path)
        
        save_path = os.path.join(save_image_dir_path, image_path.split("/")[-2] + "_" + image_path.split("/")[-1].split(".")[0] + "-" + layer_name + ".jpg")

        save_heatmap(image_path, mask, save_path)

    return None


if __name__ == "__main__":
    args = parse_args()
    main(args)