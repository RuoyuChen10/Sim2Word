# -*- coding: utf-8 -*-  

"""
Created on 2021/3/31

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

def main(args):
    # Read Image
    images = dl.Image_precessing(args.image_input)
    inputs = torch.tensor([images])
    
    # Load network
    main_net = get_network(args.main_net)
    main_net.eval()

    # CUDA
    if torch.cuda.is_available():
        main_net.cuda()
        inputs = inputs.cuda()

    # Compute the output
    output = main_net(inputs)
    # index of the identity
    index = torch.argmax(output,dim=1)[0]

    softmax = nn.Softmax(dim=1) 

    scores = softmax(output)[0][index]
    print(print("Image from path {} predicted as class {}, confidence coefficient: {}.".format(args.image_input,index,scores)))

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Interpretability Test')
    # general
    parser.add_argument('--image-input', type=str, default="./images/n000805-0012_01.jpg",
                        help='input image path')
    parser.add_argument('--main-net',
                        type=str,
                        default='VGGFace2',
                        help='Face identity recognition network.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)