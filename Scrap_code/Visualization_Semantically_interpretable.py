# -*- coding: utf-8 -*-  

"""
Created on 2021/2/3

@author: Ruoyu Chen
"""

import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np

import Datasets.dataload as dl
from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from utils import *

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
        heatmap = cv2.applyColorMap(np.uint8(255 * mask[0]), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap)

        # merge heatmap to original image
        cam = 0.5*heatmap + 0.5*np.float32(image)
    return cam, (heatmap).astype(np.uint8)

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

def generate_relative_dir(name):
    '''
    Generate results and intermediate results

    cam
    |_ name1
    |  |_ Attributes_cam
    |  |_ Joint_cam
    |_ name2
       |_Attributes_cam
       |_Joint_cam
    '''
    mkdir(os.path.join(args.output_dir,"cam"))
    mkdir(os.path.join(args.output_dir,"cam",name))
    mkdir(os.path.join(args.output_dir,"cam",name,"Attributes_cam"))
    mkdir(os.path.join(args.output_dir,"cam",name,"Joint_cam"))

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
        save_heatmap(image_input,masks[i],os.path.join(save_dir_path,"Attribute_"+str(i)+'_'+heatmap_method+'.jpg'))

def Discriminant(ground_truth, counterfactual, image_input, save_path):
    '''
    Discriminant
    '''
    discriminant = ground_truth*(np.max(counterfactual)-counterfactual)
    # Normalization
    discriminant -= np.min(discriminant)
    discriminant /= np.max(discriminant)

    save_heatmap(image_input,discriminant,save_path)

    return discriminant

def Semantic_heatmap(mask1,mask2,image_dir,save_path,heatmap_method,image_input):
    '''
    Semantic heatmap
    '''
    # Read image
    image = cv2.imread(image_dir)
    image = cv2.resize(image,(224,224))
    mask = mask1 * mask2
    num = mask.shape[0]
    for i in range(0,num):
        mask_ = np.float32(cv2.applyColorMap(np.uint8(norm_image(mask[i])), cv2.COLORMAP_JET))
        # cv2.imwrite(os.path.join(save_path,"cam",image_input.split('/')[-1]+"-attribute_mask"+str(i)+'_'+heatmap_method+'.jpg'),norm_image(mask[i]))
        cv2.imwrite(os.path.join(save_path,"cam",image_input.split('/')[-1].split('.')[0],"Joint_cam","Attribute_"+str(i)+'_'+heatmap_method+'.jpg'),0.5*image+0.5*mask_)

def main(args):
    # Generate fold to save model
    name = args.image_input.split('/')[-1].split('.')[0]
    generate_relative_dir(name)
    
    # Read the images as input
    images = dl.Image_precessing(args.image_input)
    inputs = torch.tensor([images], requires_grad=True)

    # Load the networks
    net1 = get_network(args.network1)
    net2 = get_network(None,args.network2)
    
    # Heatmap
    cam1 = get_heatmap_method(net1, method=args.heatmap_method)
    cam2 = get_heatmap_method(net2, method=args.heatmap_method)
    
    # Mask
    mask,id_1,scores1 = cam1(inputs.cuda())  # cam mask
    print("Image from path {} predicted as class {}, confidence coefficient: {}.".format(args.image_input,id_1[0],scores1[0]))
    counter_mask,id_2,scores2 = cam1(inputs.cuda(),args.counter_class)
    print("The counter class set as class {}, confidence coefficient: {}.".format(id_2[0],scores2[0]))

    # save mask
    save_heatmap(args.image_input, mask[0], os.path.join(args.output_dir,"cam",name,args.heatmap_method+'-'+'ground_truth.jpg'))
    save_heatmap(args.image_input, counter_mask[0], os.path.join(args.output_dir,"cam",name,args.heatmap_method+'-'+'counterfactual.jpg'))
    save_heatmap(args.image_input, np.max(counter_mask[0])-counter_mask[0], os.path.join(args.output_dir,"cam",name,args.heatmap_method+'-'+'counterfactual-～.jpg'))
    
    # Discriminant
    discriminant = Discriminant(mask[0], counter_mask[0], args.image_input, os.path.join(args.output_dir,"cam",name,args.heatmap_method+'-'+'Discriminant.jpg'))

    # Attribute mask
    mask2,attribute_id,scores_attr = cam2(inputs.cuda())  # cam mask

    # Save attributes mask
    save_attributes_heatmap(args.image_input,mask2,os.path.join(args.output_dir,"cam",name,"Attributes_cam"),args.heatmap_method)

    cam1.remove_handlers()
    cam2.remove_handlers()
    
    # Semantic heatmap
    Semantic_heatmap(discriminant,mask2,args.image_input,args.output_dir,args.heatmap_method,args.image_input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-input', type=str, default="./images/0016_01.jpg",
                        help='input image path')
    parser.add_argument('--network1', type=str, default='VGGFace2',
                        help='Face identity recognition network.')
    parser.add_argument('--network2', type=str, default='./pre-trained/Face-Attributes2.pth',
                        help='Attribute network, name or path.')
    parser.add_argument('--counter-class', type=int, default=108,
                        help='Counterfactual class.')
    parser.add_argument('--heatmap-method', type=str, default='GradCAM',
                        choices=['GradCAM','GradCAM++'],
                        help='Attribute network.')
    parser.add_argument('--type', type=str, default='Face',
                        choices=['CUB','Face'],
                        help='Which type.')
    parser.add_argument('--output-dir', type=str, default='./results/',
                        help='output directory to save results')
    args = parser.parse_args()
    
    mkdir(args.output_dir)
    
    main(args)