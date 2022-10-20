# -*- coding: utf-8 -*-  

"""
Created on 2021/04/21
Update  on 2021/04/28   Add new function: topk_Identity_Segmantically_Attributes_Interpretable(
                                            self,Image1,Image2,topk_num=3,heatmap_method="GradCAM")
Update  on 2021/04/29   Add new function: thresh_Identity_Segmantically_Attributes_Interpretable(
                                            self,Image1,Image2,thresh=0.8,heatmap_method="GradCAM")
Update  on 2021/05/06   Fix a bug, prevent duplicate define CAM
@author: Ruoyu Chen
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from interpretability.grad_cam import GradCAM, GradCamPlusPlus

class Segmantically_Attributes(object):
    """
    Segmantically_Attributes
    """
    def __init__(self,main_net,attribute_net,heatmap_method="GradCAM"):
        self.main_net = main_net
        self.attribute_net = attribute_net
        if torch.cuda.is_available():
            self.main_net.cuda()
            self.attribute_net.cuda()
        self.main_net.eval()
        self.attribute_net.eval()
        self.heatmap_method=heatmap_method
        self.cam1 = self.get_heatmap_method(self.main_net, self.heatmap_method)
        self.cam2 = self.get_heatmap_method(self.attribute_net, self.heatmap_method)
    
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
    
    def Segmantically_Attributes_Interpretable(self,Image,counter_class):
        # Image input
        inputs = torch.tensor([Image], requires_grad=True)
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        # Mask
        mask,id_1,scores1 = self.cam1(inputs)  # cam mask
        print("Image from predicted as class {}, confidence coefficient: {}.".format(id_1[0],scores1[0]))
        counter_mask,id_2,scores2 = self.cam1(inputs,counter_class)
        print("The counter class set as class {}, confidence coefficient: {}.".format(id_2[0],scores2[0]))

        # Discriminant
        discriminant = self.Discriminant(mask[0], counter_mask[0])

        # Attribute mask
        mask2, attribute_id, scores_attr = self.cam2(inputs)  # cam mask

        self.cam1.remove_handlers()
        self.cam2.remove_handlers()

        mask = discriminant * mask2
        for i in range(0,mask.shape[0]):
            mask[i] -= np.min(mask[i])
            mask[i] /= np.max(mask[i])
            
        return mask

    def Segmantically_Attributes_Interpretable_v2(self,Image1,Image2):
        # Image input
        inputs1 = torch.tensor([Image1], requires_grad=True)
        inputs2 = torch.tensor([Image2], requires_grad=True)
        
        if torch.cuda.is_available():
            inputs1 = inputs1.cuda()
            inputs2 = inputs2.cuda()

        # Mask
        mask,id_1,scores1 = self.cam1(inputs1)  # cam mask
        _,id_2,scores2 = self.cam1(inputs2)  # cam mask

        print("Image1 predicted as class {}, confidence coefficient: {}.".format(id_1[0],scores1[0]))
        print("Image2 predicted as class {}, confidence coefficient: {}.".format(id_2[0],scores2[0]))
        
        if id_1[0] == id_2[0]:
            print("The two images predicted the same class {}".format(id_1[0]))
            discriminant = mask[0]
        else:
            counter_mask,id_2,scores2 = self.cam1(inputs1,id_2[0])
            print("The counter class set as class {}, confidence coefficient: {}.".format(id_2[0],scores2[0]))

            # Discriminant
            discriminant = self.Discriminant(mask[0], counter_mask[0])

        # Attribute mask
        mask2, attribute_id, scores_attr = self.cam2(inputs1)  # cam mask

        self.cam1.remove_handlers()
        self.cam2.remove_handlers()

        mask = discriminant * mask2
        for i in range(0,mask.shape[0]):
            mask[i] -= np.min(mask[i])
            mask[i] /= np.max(mask[i])

        return mask

    def Visualization_Attributes_Image(self,seg_attr_mask,image,save_path):
        '''
        Visualize the images after masked by joined attributes cam
        '''
        for i in range(0,seg_attr_mask.shape[0]):
            # No nan
            if np.max(seg_attr_mask[i]) == np.max(seg_attr_mask[i]):
                # thresh
                seg_attr_mask[i][seg_attr_mask[i]<1-args.thresh] = 0
                seg_attr_mask[i][seg_attr_mask[i]>1-args.thresh] = 1

                # Save the Visualization Results
                cv2.imwrite(os.path.join(save_path,"mask","attribute"+str(i)+'.jpg'),
                            seg_attr_mask[i]*255)
                cv2.imwrite(os.path.join(save_path,"images","attribute-o"+str(i)+'.jpg'),
                    (image.transpose(2,0,1)*seg_attr_mask[i]).transpose(1,2,0))
        return None
    
    def topk_average_mask(self, inputs1, inputs2, cam, topk_num=3):
        '''
        get the mask from several classes
        '''
        # Get the topk index of inputs1 as ground truth
        output = self.main_net(inputs1)  # inputs: [bz, c, w, h]
        value,index = torch.topk(output, topk_num, dim=1,largest=True, sorted=True)
        
        aver = []
        for id_ in index[0]:
            # Calculate the mask in different topk class of ground truth
            mask,_,__ = cam(inputs1, id_)  # cam mask
            if np.max(mask) == np.max(mask):
                aver.append(mask)
        index1 = index.cpu().numpy().tolist()
        mask1 = torch.mean(torch.FloatTensor(aver),dim=0)

        # Get the topk index of inputs2 as count class
        output = self.main_net(inputs2)  # inputs: [bz, c, w, h]
        value,index = torch.topk(output, topk_num, dim=1,largest=True, sorted=True)
        
        aver = []
        for id_ in index[0]:
            # Calculate the mask in different topk class of countfactual
            mask,_,__ = cam(inputs1, id_)  # cam mask
            if np.max(mask) == np.max(mask):
                aver.append(mask)
        index2 = index.cpu().numpy().tolist()
        mask2 = torch.mean(torch.FloatTensor(aver),dim=0)

        # the mask shape is [c,w,h]
        return mask1, mask2, index1, index2
    
    def topk_Identity_Segmantically_Attributes_Interpretable(
        self, Image1, Image2, topk_num=3, visualization = False):
        '''
        The Indentity choose the top k classes and stack
        '''
        # Image input
        inputs1 = torch.tensor([Image1], requires_grad=True)
        inputs2 = torch.tensor([Image2], requires_grad=True)
        
        if torch.cuda.is_available():
            inputs1 = inputs1.cuda()
            inputs2 = inputs2.cuda()

        # Mask
        mask1, mask2, index1, index2 = self.topk_average_mask(inputs1, inputs2, self.cam1, topk_num)

        discriminant = self.Discriminant(mask1.numpy(), mask2.numpy())

        # Attribute mask
        mask_attr, attribute_id, scores_attr = self.cam2(inputs1)  # cam mask

        self.cam1.remove_handlers()
        self.cam2.remove_handlers()

        mask = discriminant * mask_attr

        ## Needn't normlization
        # for i in range(0,mask.shape[0]):
        #     mask[i] -= np.min(mask[i])
        #     mask[i] /= np.max(mask[i])
        if visualization == True:
            return mask, index1, index2, attribute_id, scores_attr, mask1.numpy(), mask2.numpy(), discriminant, mask_attr
        else:
            return mask, index1, index2, attribute_id, scores_attr

    def thresh_average_mask(self, inputs1, inputs2, cam, thresh):
        '''
        get the mask from several classes
        '''
        # Get the index of inputs1 output scores sumed bigger than thresh as ground truth
        output = self.main_net(inputs1)  # inputs: [bz, c, w, h]
        output = F.softmax(output[0],dim=0)
        
        # Get the topk index until the topk probability more than the thresh
        topk = 1
        while(True):
            value,index = torch.topk(output, topk, dim=0,largest=True, sorted=True)
            if sum(value) > thresh:
                break
            topk += 1
        index1 = index.cpu().numpy().tolist()
        
        aver = []
        for id_ in index:
            # Calculate the mask in different topk class
            mask,_,__ = cam(inputs1, id_)  # cam mask
            if np.max(mask) == np.max(mask):
                aver.append(mask)

        mask1 = torch.mean(torch.FloatTensor(aver),dim=0)

        # Get the index of inputs1 output scores sumed bigger than thresh as count class
        output = self.main_net(inputs2)  # inputs: [bz, c, w, h]
        output = F.softmax(output[0],dim=0)

        # Get the topk index until the topk probability more than the thresh
        topk = 1
        while(True):
            value,index = torch.topk(output, topk, dim=0,largest=True, sorted=True)
            if sum(value) > thresh:
                break
            topk += 1
        index2 = index.cpu().numpy().tolist()

        aver = []
        for id_ in index:
            # Calculate the mask in different topk class
            mask,_,__ = cam(inputs1, id_)  # cam mask
            if np.max(mask) == np.max(mask):
                aver.append(mask)

        mask2 = torch.mean(torch.FloatTensor(aver),dim=0)

        # the mask shape is [c,w,h]
        return mask1, mask2, index1, index2
    
    def thresh_Identity_Segmantically_Attributes_Interpretable(self,Image1,Image2,thresh=0.5):
        '''
        Using thresh to choose several mask, when the topk classes scores sum large than the thresh.
            Image1: [c,w,h], after preprocessing
            Image2: [c,w,h], after preprocessing
        Output:
            mask: the discriminant result
            index1: the class for image1 predict topk classes that scores sumed more than thresh
            index2: the class for image2 predict topk classes that scores sumed more than thresh
        '''
        assert thresh > 0 and thresh < 1

        # Image input
        inputs1 = torch.tensor([Image1], requires_grad=True)
        inputs2 = torch.tensor([Image2], requires_grad=True)
        
        if torch.cuda.is_available():
            inputs1 = inputs1.cuda()
            inputs2 = inputs2.cuda()

        # Mask
        mask1, mask2, index1, index2 = self.thresh_average_mask(inputs1,inputs2,self.cam1,thresh)

        discriminant = self.Discriminant(mask1.numpy(), mask2.numpy())

        # Attribute mask
        mask_attr, attribute_id, scores_attr = self.cam2(inputs1)  # cam mask

        self.cam1.remove_handlers()
        self.cam2.remove_handlers()

        mask = discriminant * mask_attr

        for i in range(0,mask.shape[0]):
            mask[i] -= np.min(mask[i])
            mask[i] /= np.max(mask[i])
        
        return mask, index1, index2, attribute_id, scores_attr

    def ablation_no_attribute(self,Image1,Image2):
        # Image input
        inputs1 = torch.tensor([Image1], requires_grad=True)
        inputs2 = torch.tensor([Image2], requires_grad=True)
        
        if torch.cuda.is_available():
            inputs1 = inputs1.cuda()
            inputs2 = inputs2.cuda()

        # Mask
        mask1,mask2,index1,index2 = self.topk_average_mask(inputs1,inputs2,self.cam1,1)

        discriminant = self.Discriminant(mask1.numpy(), mask2.numpy())  # [w,h]

        self.cam1.remove_handlers()

        return discriminant, index1, index2
    
    def ablation_no_attribute_v2(self,Image1,Image2):
        # Image input
        inputs1 = torch.tensor([Image1], requires_grad=True)
        inputs2 = torch.tensor([Image2], requires_grad=True)
        
        if torch.cuda.is_available():
            inputs1 = inputs1.cuda()
            inputs2 = inputs2.cuda()

        # Mask
        mask1,mask2,index1,index2 = self.topk_average_mask(inputs1,inputs2,self.cam1,1)

        discriminant = self.Discriminant(mask1.numpy(), mask2.numpy())  # [w,h]

        self.cam1.remove_handlers()

        return discriminant, mask1,mask2

    def ablation_no_counterfactual(self,Image1):
        # Image input
        inputs1 = torch.tensor([Image1], requires_grad=True)
        # inputs2 = torch.tensor([Image2], requires_grad=True)
        
        if torch.cuda.is_available():
            inputs1 = inputs1.cuda()
            # inputs2 = inputs2.cuda()

        # Attribute mask
        mask_attr, attribute_id, scores_attr = self.cam2(inputs1)  # cam mask

        self.cam2.remove_handlers()

        mask = mask_attr

        for i in range(0,mask.shape[0]):
            mask[i] -= np.min(mask[i])
            mask[i] /= np.max(mask[i])
        return mask, attribute_id, scores_attr

    def single_people_id_w_attributes(self, input1, input2):
        """
        Create on 2022/1/14

        Identity attribute map with attributes map
        """
        if torch.cuda.is_available():
            input1 = input1.cuda()
            input2 = input2.cuda()
        
        # Mask
        mask, id, scores = self.cam1(input1)  # cam mask
        mask_attr, attribute_id, scores_attr = self.cam2(input2)

        mask_merge = mask * mask_attr

        # Normalization
        mask_merge -= np.min(mask_merge, (1,2)).reshape((mask_merge.shape[0], 1, 1))
        mask_merge /= (np.max(mask_merge, (1,2)).reshape((mask_merge.shape[0], 1, 1)) + 1e-12)

        return mask_merge, id, scores, attribute_id, scores_attr





