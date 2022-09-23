# -*- coding: utf-8 -*-  

"""
Created on 2021/3/30

@author: Ruoyu Chen
"""

import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np
import json

import Datasets.dataload as dl
from Semantically_interpretable import Segmantically_Attributes
from utils import *

def _get_features_hook(module, input, output):
        global hook_feature 
        hook_feature = output.view(output.size(0), -1)[0]
        # print("feature shape:{}".format(hook_feature.size()))

def _register_hook(net,layer_name):
    for (name, module) in net.named_modules():
        if name == layer_name:
            module.register_forward_hook(_get_features_hook)

def Image_precessing(image):
    '''
    Precessing the input images
        image_dir: single image input path, such as "/home/xxx/10.jpg"
    '''
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])
    assert image is not None
    image = cv2.resize(image,(224,224))
    image = image.astype(np.float32)
    image -= mean_bgr
    # H * W * C   -->   C * H * W
    image = image.transpose(2,0,1)
    return image

def main(args):
    # Path save
    mkdir(os.path.join(args.output_dir,"scores"))
    mkdir(os.path.join(args.output_dir,"scores",args.image_input.split('/')[-1].split('.')[0]+'-'+args.counter_image.split('/')[-1].split('.')[0]))
    mkdir(os.path.join(args.output_dir,"scores",args.image_input.split('/')[-1].split('.')[0]+'-'+args.counter_image.split('/')[-1].split('.')[0],"mask"))
    mkdir(os.path.join(args.output_dir,"scores",args.image_input.split('/')[-1].split('.')[0]+'-'+args.counter_image.split('/')[-1].split('.')[0],"images"))

    # Read Image
    images = dl.Image_precessing(args.image_input)
    
    # Load network
    main_net = get_network(args.main_net)
    attribute_net = get_network(None,args.attribute_net)

    # Segmantically Attributes
    seg_attr = Segmantically_Attributes(main_net,attribute_net)
    seg_attr_interpretable = seg_attr.Segmantically_Attributes_Interpretable(images,args.counter_class,args.heatmap_method)

    seg_attr_interpretable = 1 - seg_attr_interpretable

    # Read the original image
    image = cv2.imread(args.image_input)
    image_c = dl.Image_precessing(args.counter_image)

    # Init
    main_net.eval()

    # Hook
    _register_hook(main_net,"avgpool")

    # get the feature vector
    main_net(torch.tensor([images], requires_grad=True).cuda())
    f_y = hook_feature
    main_net(torch.tensor([image_c], requires_grad=True).cuda())
    f_c = hook_feature

    scores = {}
    scores["thresh"] = args.thresh
    scores["f_y-f_c"] = torch.cosine_similarity(f_y, f_c, dim=0).item()
    attr = {}
    for i in range(0,seg_attr_interpretable.shape[0]):
        # No nan
        if np.max(seg_attr_interpretable[i]) == np.max(seg_attr_interpretable[i]):
            # thresh
            seg_attr_interpretable[i][seg_attr_interpretable[i]<1-args.thresh] = 0
            seg_attr_interpretable[i][seg_attr_interpretable[i]>1-args.thresh] = 1

            inputs = Image_precessing((image.transpose(2,0,1)*seg_attr_interpretable[i]).transpose(1,2,0))
            
            # Save the Visualization Results
            cv2.imwrite(os.path.join(args.output_dir,"scores",args.image_input.split('/')[-1].split('.')[0]+'-'+args.counter_image.split('/')[-1].split('.')[0],"mask","attribute"+str(i)+'.jpg'),
                        seg_attr_interpretable[i]*255)
            cv2.imwrite(os.path.join(args.output_dir,"scores",args.image_input.split('/')[-1].split('.')[0]+'-'+args.counter_image.split('/')[-1].split('.')[0],"images","attribute-o"+str(i)+'.jpg'),
                (image.transpose(2,0,1)*seg_attr_interpretable[i]).transpose(1,2,0))
            
            main_net(torch.tensor([inputs], requires_grad=True).cuda())

            attr["f_attr_"+str(i)] = (torch.cosine_similarity(f_y, hook_feature, dim=0).item(),
                                      torch.cosine_similarity(hook_feature, f_c, dim=0).item())
    scores["attributes-(f_a-f_y,f_a-f_c)"] = attr
    with open(os.path.join(args.output_dir,"scores",args.image_input.split('/')[-1].split('.')[0]+'-'+args.counter_image.split('/')[-1].split('.')[0],"scores.json"), "w") as f:
        f.write(json.dumps(scores, ensure_ascii=False, indent=4, separators=(',', ':')))

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Interpretability Test')
    # general
    parser.add_argument('--image-input', type=str, default="./Verification/VGGFace2-test/n008858/0446_01.jpg",#"./images/n000558-0034_01.jpg",
                        help='input image path')
    parser.add_argument('--main-net',
                        type=str,
                        default='VGGFace2',
                        help='Face identity recognition network.')
    parser.add_argument('--attribute-net',
                        type=str,
                        default='./pre-trained/Face-Attributes2.pth',
                        help='Attribute network, name or path.')
    parser.add_argument('--counter-image', type=str, default="./images/n000424-0060_01.jpg",
                        help='Counterfactual class.')
    parser.add_argument('--counter-class', type=int, default=403,
                        help='Counterfactual class.')
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