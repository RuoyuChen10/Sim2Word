# -*- coding: utf-8 -*-  

"""
Created on 2021/4/5

@author: Ruoyu Chen
"""

import numpy as np
import argparse
import os
import torch
import random
import cv2

import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from iresnet import iresnet18,iresnet34,iresnet50,iresnet100,iresnet200
from utils import *

import sys

sys.path.append('../')

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

class VGGFace2_verifacation(object):
    def __init__(self,vggnet):
        self.net = vggnet
        self.net.eval()
        self.hook_feature = None
    def _register_hook(self,net,layer_name):
        for (name, module) in net.named_modules():
            if name == layer_name:
                module.register_forward_hook(self._get_features_hook)
    def _get_features_hook(self,module, input, output):
        self.hook_feature = output.view(output.size(0), -1)[0]
    def __call__(self,input):
        self._register_hook(self.net,"avgpool")
        self.net.zero_grad()
        self.net(input)
        return torch.unsqueeze(self.hook_feature, dim=0)

def get_net(net_type):
    arcface_r50_path = "./pretrained/ms1mv3_arcface_r50_fp16/backbone.pth"
    arcface_r100_path = "./pretrained/ms1mv3_arcface_r100_fp16/backbone.pth"
    cosface_r50_path = "./pretrained/glint360k_cosface_r50_fp16_0.1/backbone.pth"
    cosface_r100_path = "./pretrained/glint360k_cosface_r100_fp16_0.1/backbone.pth"
    if net_type == "ArcFace-r50":
        net = iresnet50()
        net.load_state_dict(torch.load(arcface_r50_path))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()
    elif net_type == "ArcFace-r100":
        net = iresnet100()
        net.load_state_dict(torch.load(arcface_r100_path))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()
    elif net_type == "CosFace-r50":
        net = iresnet50()
        net.load_state_dict(torch.load(cosface_r50_path))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()
    elif net_type == "CosFace-r100":
        net = iresnet100()
        net.load_state_dict(torch.load(cosface_r100_path))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()
    elif net_type == "VGGFace2":
        vggnet = get_network('VGGFace2')
        if torch.cuda.is_available():
            vggnet.cuda()
        net = VGGFace2_verifacation(vggnet)
    return net

def Image_Preprocessing(net_type,path):
    if net_type in ['ArcFace-r50','ArcFace-r100',"CosFace-r50","CosFace-r100"]:
        return transforms(Image.open(path).resize((112, 112), Image.BILINEAR))
    elif net_type in ["VGGFace2"]:
        mean_bgr = np.array([91.4953, 103.8827, 131.0912])
        image = cv2.imread(path)
        assert image is not None
        image = cv2.resize(image,(224,224))
        image = image.astype(np.float32)
        image -= mean_bgr
        # H * W * C   -->   C * H * W
        image = image.transpose(2,0,1)
        return torch.tensor(image)

def main(args):
    net = get_net(args.Net_type)

    if args.Datasets == "VGGFace2-test":
        if os.path.exists("./results/text/VGGFace2-test-results.txt"):
            os.remove("./results/text/VGGFace2-test-results.txt")  
        with open("./text/VGGFace2.txt", "r") as f:
            datas = f.read().split('\n')
        for data in tqdm(datas):
            try:
                path1 = os.path.join("./dataset/VGGFace2-test/",data.split(' ')[0])
                path2 = os.path.join("./dataset/VGGFace2-test/",data.split(' ')[1])
                # image1 = Image.open(path1).resize((112, 112), Image.BILINEAR)
                # image2 = Image.open(path2).resize((112, 112), Image.BILINEAR)
                # image1 = transforms(image1)
                # image2 = transforms(image2)
                image1 = Image_Preprocessing(args.Net_type,path1)
                image2 = Image_Preprocessing(args.Net_type,path2)

                output1 = F.normalize(net(torch.unsqueeze(image1, dim=0).cuda()),p=2,dim=1)
                output2 = F.normalize(net(torch.unsqueeze(image2, dim=0).cuda()),p=2,dim=1)

                similar = torch.cosine_similarity(output1[0], output2[0], dim=0).item()
                with open("./results/text/VGGFace2-test-results.txt", 'a') as file:
                    file.write(path1+' '+path2+' '+str(similar)+' '+data.split(' ')[2]+'\n')
            except:
                pass
    elif args.Datasets == "Celeb-A":
        if os.path.exists("./results/text/Celeb-A-results.txt"):
            os.remove("./results/text/Celeb-A-results.txt")
        with open("./text/CelebA.txt", "r") as f:
            datas = f.read().split('\n')
        for data in tqdm(datas):
            try:
                path1 = os.path.join("./dataset/CelebA-test",data.split(' ')[0])
                path2 = os.path.join("./dataset/CelebA-test",data.split(' ')[1])
                image1 = Image_Preprocessing(args.Net_type,path1)
                image2 = Image_Preprocessing(args.Net_type,path2)
                
                output1 = F.normalize(net(torch.unsqueeze(image1, dim=0).cuda()),p=2,dim=1)
                output2 = F.normalize(net(torch.unsqueeze(image2, dim=0).cuda()),p=2,dim=1)

                similar = torch.cosine_similarity(output1[0], output2[0], dim=0).item()
                with open("./results/text/Celeb-A-results.txt", 'a') as file:
                    file.write(path1+' '+path2+' '+str(similar)+' '+data.split(' ')[2]+'\n')
            except:
                pass
    elif args.Datasets == "Celeb-A2":
        if os.path.exists("./results/text/Celeb-A-results2.txt"):
            os.remove("./results/text/Celeb-A-results2.txt")
        with open("./text/CelebA.txt", "r") as f:
            datas = f.read().split('\n')
        for data in tqdm(datas):
            try:
                path1 = os.path.join("./dataset/CelebA-test2",data.split(' ')[0])
                path2 = os.path.join("./dataset/CelebA-test2",data.split(' ')[1])
                image1 = Image_Preprocessing(args.Net_type,path1)
                image2 = Image_Preprocessing(args.Net_type,path2)
                
                output1 = F.normalize(net(torch.unsqueeze(image1, dim=0).cuda()),p=2,dim=1)
                output2 = F.normalize(net(torch.unsqueeze(image2, dim=0).cuda()),p=2,dim=1)

                similar = torch.cosine_similarity(output1[0], output2[0], dim=0).item()
                with open("./results/text/Celeb-A-results2.txt", 'a') as file:
                    file.write(path1+' '+path2+' '+str(similar)+' '+data.split(' ')[2]+'\n')
            except:
                pass

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Verification List')
    # general
    parser.add_argument('--Net-type',
                        type=str,
                        default='ArcFace-r100',
                        choices=['ArcFace-r50','ArcFace-r100',"CosFace-r50","CosFace-r100","VGGFace2"],
                        help='Which network using for face verification.')
    parser.add_argument('--Datasets',
                        type=str,
                        choices=['VGGFace2-test','Celeb-A','Celeb-A2'],
                        default='Celeb-A2',
                        help='Datasets.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)