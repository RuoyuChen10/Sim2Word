# -*- coding: utf-8 -*-  

"""
Created on 2022/1/24

@author: Ruoyu Chen
"""

import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np
import json
import os
import time

import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from interpretability.Semantically_interpretable import Segmantically_Attributes
import torchvision.transforms as transforms
from utils import *

save_location = np.array([
    0,0,0,1,1,
    1,1,1,0,1,
    1,1,1,1,1,
    1,1,1,1,1,
    1,1,1,1,1,
    1,1,1,1,1
])

class VGGFace2Dataset(torch.utils.data.Dataset):
    """
    Read datasets
    """
    def __init__(self, dataset_root, dataset_list):
        self.dataset_root = dataset_root
        self.mean_bgr = np.array([91.4953, 103.8827, 131.0912])

        with open(dataset_list,"r") as file:
            datas = file.readlines()

        self.data = [os.path.join(self.dataset_root, data_) for data_ in datas]

        # self.data = np.random.permutation(data)
        self.transforms = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    def __len__(self):
        return len(self.data)

    def Path_Image_Preprocessing(self, path):
        '''
        Precessing the input images
            image_dir: single image input path, such as "/home/xxx/10.jpg"
        '''
        
        image = cv2.imread(path)
        assert image is not None
        image = cv2.resize(image,(224,224))
        image = image.astype(np.float32)
        image -= self.mean_bgr
        # H * W * C   -->   C * H * W
        image = image.transpose(2,0,1)
        return torch.tensor(image)

    def __getitem__(self,index):
        # Sample
        sample = self.data[index]
        
        # data and label information
        splits = sample.split(' ')
        image_path = splits[0]

        data = Image.open(image_path)
        data = self.transforms(data)

        attribute_data = self.Path_Image_Preprocessing(image_path)

        label = np.int32(splits[1])

        return image_path, data.float(), attribute_data, label

def main(args):
    save_log = os.path.join(args.log_dir, "qunatity-evidential-t-"+str(args.thresh) + "-erase-num-" + str(args.Erase_num) + ".log")
    info = INFO(save_log)

    # Read Dataset
    dataset = VGGFace2Dataset(dataset_root=args.dataset_root,dataset_list=args.dataset_list)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False) # Batch Size Must be 1!

    # Load Recognition Network
    recognition_net = get_network(args.recognition_net)
    if torch.cuda.is_available():
        recognition_net.cuda()
    recognition_net.eval()

    # Load Attributes Network
    attribute_net = get_network(None,args.attribute_net)
    if torch.cuda.is_available():
        attribute_net.cuda()
    attribute_net.eval()
    
    # Load Verification Network
    evidential_net = get_network(args.evidential_net)   # Already cuda() and eval() operation
    evidential_net.eval()

    seg_attr = Segmantically_Attributes(recognition_net, attribute_net, args.heatmap_method)

    original_acc = 0
    erased_acc = 0

    for ii, (image_path, data, attribute_data, label) in enumerate(data_loader):
        since = time.time()

        # Base element
        mask_merge, pre_id, pre_score, attribute_id, attribute_score = seg_attr.single_people_id_w_attributes(data, attribute_data)

        original_acc += (pre_id == label).sum().item()

        # Threshold
        mask_merge[mask_merge<args.thresh] = 0
        mask_merge[mask_merge>=args.thresh] = 1
        mask_merge = 1 - mask_merge

        # selected attributes index
        idx = (save_location * attribute_id + 1) * (np.array(attribute_score)>0.8).astype(np.int)
        idx = np.where(idx==1)[0]

        with torch.no_grad():
            # Evidential Network
            mask_merge = mask_merge[idx]
            mask_input = data * mask_merge.reshape((mask_merge.shape[0],1,mask_merge.shape[1],mask_merge.shape[2]))
            try:
                outputs = evidential_net(
                    torch.cat((data,mask_input),0)
                )

                # evidence = exp_evidence(outputs)
                evidence = relu_evidence(outputs) * 100

                alpha = evidence + 1
                uncertain = args.num_classes / torch.sum(alpha, dim=1, keepdim=True)

                value, indices = torch.topk(uncertain.flatten()[1:], k=len(idx), dim=0, largest=True)

                # Quantity
                if args.Erase_num < (value > uncertain[0]).sum():   # we can choose 
                    if args.Erase_num == 1:
                        mask = mask_merge[indices[:args.Erase_num]]
                    else:
                        mask = (mask_merge[indices[:args.Erase_num]].sum(0)==args.Erase_num).astype(int)
                elif (value > uncertain[0]).sum() > 1:
                    mask = (mask_merge[indices[:(value > uncertain[0]).sum().item()]].sum(0)==(value > uncertain[0]).sum().item()).astype(int)
                elif (value > uncertain[0]).sum() == 1:
                    mask = mask_merge[indices[0]]
                else: 
                    print("mask = 1?")
                    mask = 1

                # Erase
                input_data = data * mask + (mask - 1)
                pre_label =  torch.argmax(recognition_net(input_data.float().cuda())).item()
                erased_acc += (pre_label == label).sum().item()
            except:
                info(image_path[0] + " error!")
                pass
        
        item_time = time.time() - since
        time_str = time.asctime(time.localtime(time.time()))

        info("{} process {}/{}, this term using time {:.4f} s".format(time_str, ii+1,len(data_loader), item_time))
    info("original acc={}".format(original_acc / len(data_loader.dataset)))
    info("erased acc={}".format(erased_acc / len(data_loader.dataset)))

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Interpretability Test')
    # general
    parser.add_argument('--dataset-root', type=str,
                        default='/home/cry/data2/VGGFace2/train_align_arcface',
                        help='')
    parser.add_argument('--dataset-list', type=str,
                        default='./Datasets/VGGFace2-uncertain-quantity-1.txt',
                        # default='./Datasets/quan-test.txt',
                        help='')                    
    parser.add_argument('--recognition-net',
                        type=str,
                        default="VGGFace2-ArcFace",
                        choices=["VGGFace2","VGGFace2-ArcFace"],
                        help='Face identity recognition network.')
    parser.add_argument('--num-classes', type=int,
                        default=8631,
                        help='')
    parser.add_argument('--attribute-net',
                        type=str,
                        default='./pre-trained/Face-Attributes2.pth',
                        help='Attribute network, name or path.')
    parser.add_argument('--evidential-net',
                        type=str,
                        default='VGGFace2-evidential',
                        help='Evidential learning.')
    parser.add_argument('--heatmap-method', type=str, default='GradCAM',
                        choices=['GradCAM','GradCAM++'],
                        help='Attribute network.')
    parser.add_argument('--Erasing-method', type=str, default="black",
                        choices=["black","white","mean","random"],
                        help='Which method to erasing.')
    parser.add_argument('--thresh', type=float, default=0.1,
                        help='Thresh.')
    parser.add_argument('--Erase_num', type=int, default=3,
                        help='Erase attribute num.')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)


