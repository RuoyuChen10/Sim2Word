# -*- coding: utf-8 -*-  

"""
Created on 2021/3/29

@author: Ruoyu Chen
"""

import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np

from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from utils import *

class Segmantically_Attributes(object):
    """
    Segmantically_Attributes
    """
    def __init__(self,main_net,attribute_net):
        self.main_net = main_net
        self.attribute_net = attribute_net
        if torch.cuda.is_available():
            self.main_net.cuda()
            self.attribute_net.cuda()
        self.main_net.eval()
        self.attribute_net.eval()
    
    def get_last_conv_name(self,net):
        """
        Get the name of last convolutional layer
        """
        layer_name = None
        for name, m in net.named_modules():
            if isinstance(m, nn.Conv2d):
                layer_name = name
        return layer_name

    def get_heatmap_method(self, net, method="GradCAM"):
        '''
        Get the method to generate heatmap
        '''
        layer_name = self.get_last_conv_name(net)
        if method == "GradCAM": 
            cam = GradCAM(net, layer_name)
        elif method == "GradCAM++":
            cam = GradCamPlusPlus(net, layer_name)
        return cam

    def Discriminant(self, ground_truth, counterfactual):
        '''
        Discriminant
        '''
        discriminant = ground_truth*(np.max(counterfactual)-counterfactual)
        # Normalization
        discriminant -= np.min(discriminant)
        discriminant /= np.max(discriminant)

        return discriminant
    
    def Segmantically_Attributes_Interpretable(self,Image,counter_class,heatmap_method="GradCAM"):
        # Image input
        inputs = torch.tensor([Image], requires_grad=True)
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        # Heatmap
        cam1 = self.get_heatmap_method(self.main_net, method=heatmap_method)
        cam2 = self.get_heatmap_method(self.attribute_net, method=heatmap_method)

        # Mask
        mask,id_1,scores1 = cam1(inputs)  # cam mask
        print("Image from predicted as class {}, confidence coefficient: {}.".format(id_1[0],scores1[0]))
        counter_mask,id_2,scores2 = cam1(inputs,counter_class)
        print("The counter class set as class {}, confidence coefficient: {}.".format(id_2[0],scores2[0]))

        # Discriminant
        discriminant = self.Discriminant(mask[0], counter_mask[0])

        # Attribute mask
        mask2, attribute_id, scores_attr = cam2(inputs)  # cam mask

        cam1.remove_handlers()
        cam2.remove_handlers()

        mask = discriminant * mask2
        for i in range(0,mask.shape[0]):
            mask[i] -= np.min(mask[i])
            mask[i] /= np.max(mask[i])
            
        return mask

    def Segmantically_Attributes_Interpretable_v2(self,Image1,Image2,heatmap_method="GradCAM"):
        # Image input
        inputs1 = torch.tensor([Image1], requires_grad=True)
        inputs2 = torch.tensor([Image2], requires_grad=True)
        
        if torch.cuda.is_available():
            inputs1 = inputs1.cuda()
            inputs2 = inputs2.cuda()

        # Heatmap
        cam1 = self.get_heatmap_method(self.main_net, method=heatmap_method)
        cam2 = self.get_heatmap_method(self.attribute_net, method=heatmap_method)

        # Mask
        mask,id_1,scores1 = cam1(inputs1)  # cam mask
        _,id_2,scores2 = cam1(inputs2)  # cam mask

        print("Image1 predicted as class {}, confidence coefficient: {}.".format(id_1[0],scores1[0]))
        print("Image2 predicted as class {}, confidence coefficient: {}.".format(id_2[0],scores2[0]))
        
        if id_1[0] == id_2[0]:
            print("The two images predicted the same class {}".format(id_1[0]))
            discriminant = mask[0]
        else:
            counter_mask,id_2,scores2 = cam1(inputs1,id_2[0])
            print("The counter class set as class {}, confidence coefficient: {}.".format(id_2[0],scores2[0]))

            # Discriminant
            discriminant = self.Discriminant(mask[0], counter_mask[0])

        # Attribute mask
        mask2, attribute_id, scores_attr = cam2(inputs1)  # cam mask

        cam1.remove_handlers()
        cam2.remove_handlers()

        mask = discriminant * mask2
        for i in range(0,mask.shape[0]):
            mask[i] -= np.min(mask[i])
            mask[i] /= np.max(mask[i])
        return mask