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

attribute_net = get_network("VGGFace2")
if torch.cuda.is_available():
    attribute_net.cuda()
attribute_net.eval()

layer_name = get_last_conv_name(attribute_net)

cam = GradCAM(attribute_net, layer_name)

inputs = torch.tensor([Path_Image_Preprocessing("VGGFace2","/home/cry/data1/Counterfactual_interpretable/Verification/dataset/VGGFace2-test/n008858/0446_01.jpg")], requires_grad=True)

mask,id_1,scores1 = cam.get_heatmap_single_out(inputs.cuda())  # cam mask

cv2.imwrite("debug.jpg",mask*255)