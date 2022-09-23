# -*- coding: utf-8 -*-  

"""
Created on 2021/05/07

@author: Ruoyu Chen
"""

import os
import torch
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import argparse

from mtcnn_pytorch.crop_and_aligned import mctnn_crop_face
from model.model import xCosModel
from utils.util import batch_visualize_xcos

from tqdm import tqdm

transforms_mine = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor(),
    transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
])

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def Read_Datasets(Datasets_type):
    '''
    Read the path in different datasets
    '''
    if Datasets_type == "VGGFace2-train":
        dataset_list = "../../Verification/text/VGGFace2-train.txt"
        if os.path.exists(dataset_list):
            with open(dataset_list, "r") as f:
                datas = f.read().split('\n')
        else:
            raise ValueError("File {} not in path".format(dataset_list))
    elif Datasets_type == "VGGFace2-test":
        dataset_list = "../../Verification/text/VGGFace2-test.txt"
        if os.path.exists(dataset_list):
            with open(dataset_list, "r") as f:
                datas = f.read().split('\n')
        else:
            raise ValueError("File {} not in path".format(dataset_list))
    elif Datasets_type == "Celeb-A":
        dataset_list = "../../Verification/text/CelebA.txt"
        if os.path.exists(dataset_list):
            with open(dataset_list, "r") as f:
                datas = f.read().split('\n')
        else:
            raise ValueError("File {} not in path".format(dataset_list))
    return datas

def norm_image(image):
    """
    Normalization image
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image = 1-image
    image *= 255.
    return np.uint8(image)

def plot_visualization(image_dir_path,data,model,save_path):
    # figure
    plt.figure()

    # Get path
    path1 = os.path.join(image_dir_path,data.split(' ')[0])
    path2 = os.path.join(image_dir_path,data.split(' ')[1])
    if_same = data.split(' ')[2]

    # Read Image
    image1 = Image.open(path1).convert('RGB')
    image2 = Image.open(path2).convert('RGB')

    # mtcnn
    image1 = mctnn_crop_face(image1, BGR2RGB=False)
    image2 = mctnn_crop_face(image2, BGR2RGB=False)

    plt.subplot(2,2,1)
    plt.axis('off')
    plt.imshow(image1)
    plt.subplot(2,2,2)
    plt.axis('off')
    plt.imshow(image2)

    # transforms
    inputs1 = transforms_mine(image1)
    inputs2 = transforms_mine(image2)

    imgs_tensor = torch.stack([inputs1, inputs2])

    data = {}
    data['data_input'] = imgs_tensor.unsqueeze(1).cuda()

    model_output = model(data, scenario="get_feature_and_xcos")

    # matrix
    grid_cos_maps = model_output['grid_cos_maps'].squeeze().detach().cpu().unsqueeze(0).numpy()
    attention_maps = model_output['attention_maps'].squeeze().detach().cpu().unsqueeze(0).numpy()

    # mask
    mask = grid_cos_maps * attention_maps
    mask = norm_image(mask[0])
    mask = cv2.resize(mask,(224,224))

    heatmap = cv2.applyColorMap(np.array(mask), cv2.COLORMAP_JET)

    image1 = cv2.cvtColor(np.asarray(image1),cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor(np.asarray(image2),cv2.COLOR_RGB2BGR)

    after_mask1 = Image.fromarray(cv2.cvtColor(np.uint8(image1*0.5 + heatmap*0.5),cv2.COLOR_BGR2RGB))
    after_mask2 = Image.fromarray(cv2.cvtColor(np.uint8(image2*0.5 + heatmap*0.5),cv2.COLOR_BGR2RGB))

    plt.subplot(2,2,3)
    plt.axis('off') 
    plt.imshow(after_mask1)
    plt.subplot(2,2,4)
    plt.axis('off')
    plt.imshow(after_mask2)

    plt.savefig(save_path)
    plt.close()

def main(args):
    if args.Datasets == 'VGGFace2-train':
        image_dir_path = "/home/cry/data2/VGGFace2/train/"
    elif args.Datasets == "VGGFace2-test":
        image_dir_path = "/home/cry/data2/VGGFace2/test/"
    elif args.Datasets == "Celeb-A":
        image_dir_path = "/home/cry/data2/CelebA/Img/img_align_celeba"
    
    # Model
    model = xCosModel()
    pretrained_path = './pretrained_model/xcos/20200217_accu_9931_Arcface.pth'
    checkpoint = torch.load(pretrained_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.cuda()
    model.eval()
    # Read image path
    datas = Read_Datasets(args.Datasets)

    mkdir(os.path.join(args.output_dir,args.Datasets))
    
    num = 1
    for data in tqdm(datas):
        try:
            save_path = os.path.join(args.output_dir,args.Datasets,str(num)+".jpg")
            plot_visualization(image_dir_path,data,model,save_path)
        except:
            pass
        num = num + 1

def parse_args():
    parser = argparse.ArgumentParser(description='results in xCos Face')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        choices=['VGGFace2-train','VGGFace2-test','Celeb-A'],
                        default='VGGFace2-train',
                        help='Datasets.')
    parser.add_argument('--output-dir', type=str, default='./results/Visualization2',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)