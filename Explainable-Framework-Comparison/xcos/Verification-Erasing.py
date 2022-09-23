# -*- coding: utf-8 -*-  

"""
Created on 2021/05/09

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
import json

from mtcnn_pytorch.crop_and_aligned import mctnn_crop_face
from model.model import xCosModel
from utils.util import batch_visualize_xcos
from verification_network import get_network

from tqdm import tqdm

transforms_mine = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor(),
    transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
])

def Path_Image_Preprocessing(net_type,path):
    '''
    Precessing the input images
        image_dir: single image input path, such as "/home/xxx/10.jpg"
    '''
    if net_type in ['ArcFace-r50','ArcFace-r100',"CosFace-r50","CosFace-r100"]:
        return transforms_mine(Image.open(path))
    elif net_type in ["VGGFace2","VGGFace2-verification"]:
        mean_bgr = np.array([91.4953, 103.8827, 131.0912])
        image = cv2.imread(path)
        assert image is not None
        image = cv2.resize(image,(224,224))
        image = image.astype(np.float32)
        image -= mean_bgr
        # H * W * C   -->   C * H * W
        image = image.transpose(2,0,1)
        return torch.tensor(image)

def Image_Preprocessing(net_type,image):
    '''
    Precessing the input images
        image_dir: single image input path, such as "/home/xxx/10.jpg"
    '''
    if net_type in ['ArcFace-r50','ArcFace-r100',"CosFace-r50","CosFace-r100"]:
        image = Image.fromarray(cv2.cvtColor(np.uint8(image),cv2.COLOR_BGR2RGB))
        return transforms_mine(image)
    elif net_type in ["VGGFace2","VGGFace2-verification"]:
        mean_bgr = np.array([91.4953, 103.8827, 131.0912])
        assert image is not None
        image = cv2.resize(image,(224,224))
        image = image.astype(np.float32)
        image -= mean_bgr
        # H * W * C   -->   C * H * W
        image = image.transpose(2,0,1)
        return torch.tensor(image)

def Erasing(image,mask,method):
    '''
    image: (H,W,3)
    mask: (H,W) saved pixels set 1
    method: black, white, mean, random
    '''
    if method == "black":
        image_mask = (image.transpose(2,0,1)*mask).transpose(1,2,0)
    elif method == "white":
        mask_white = np.ones(mask.shape)*(1-mask)*255
        image_mask = (image.transpose(2,0,1)*mask+mask_white).transpose(1,2,0)
    elif method == "mean":
        mask_mean = np.ones(mask.shape)*(1-mask)*125
        image_mask = (image.transpose(2,0,1)*mask+mask_mean).transpose(1,2,0)
    elif method == "random":
        mask_random = np.ones(mask.shape)*(1-mask) * np.random.randint(0,256,(3,mask.shape[0],mask.shape[1]))
        mask_random = mask_random.transpose(1,2,0)
        image_mask = (image.transpose(2,0,1)*mask).transpose(1,2,0) + mask_random
    return image_mask

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

    return image

def XCos_heatmap(image_path1,image_path2,model):
    '''
    XCos heatmap output
    '''
    # Read image
    img1 = Image.open(image_path1).convert('RGB')
    img2 = Image.open(image_path2).convert('RGB')
    
    # align
    img1 = mctnn_crop_face(img1, BGR2RGB=False)
    img2 = mctnn_crop_face(img2, BGR2RGB=False)

    # transform
    img1 = transforms_mine(img1)
    img2 = transforms_mine(img2)

    # stack
    imgs_tensor = torch.stack([img1, img2])

    data = {}
    data['data_input'] = imgs_tensor.unsqueeze(1).cuda()
    ######
    model_output = model(data, scenario="get_feature_and_xcos")

    grid_cos_maps = model_output['grid_cos_maps'].squeeze().detach().cpu().unsqueeze(0).numpy()
    attention_maps = model_output['attention_maps'].squeeze().detach().cpu().unsqueeze(0).numpy()

    mask = grid_cos_maps*attention_maps
    mask = norm_image(mask[0])
    mask = cv2.resize(mask,(224,224))   # shape (224,224)

    return mask

def Verification_path(image_path1,image_path2,verification_net,args):
    # Read image
    img1 = Image.open(image_path1).convert('RGB')
    img2 = Image.open(image_path2).convert('RGB')
    
    # align
    img1 = cv2.cvtColor(np.asarray(mctnn_crop_face(img1, BGR2RGB=False)),cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.asarray(mctnn_crop_face(img2, BGR2RGB=False)),cv2.COLOR_RGB2BGR)

    img1 = Image_Preprocessing(args.verification_net,img1)
    img2 = Image_Preprocessing(args.verification_net,img2)

    feature1 = verification_net(torch.unsqueeze(img1, dim=0).cuda())
    feature2 = verification_net(torch.unsqueeze(img2, dim=0).cuda())

    similarity = torch.cosine_similarity(feature1[0], feature2[0], dim=0).item()
    return similarity,feature1,feature2

def main(args):
    '''
    Verification in different erasing
    '''
    mkdir(args.output_dir)
    mkdir(os.path.join(args.output_dir,args.Datasets+'-'+args.verification_net+'-'+args.Erasing_method))
    mkdir(os.path.join(args.output_dir,args.Datasets+'-'+args.verification_net+'-'+args.Erasing_method,"Json"))
    mkdir(os.path.join(args.output_dir,args.Datasets+'-'+args.verification_net+'-'+args.Erasing_method,"Mask"))

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
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # Verification network
    verification_net = get_network(args.verification_net)   # Already cuda() and eval() operation

    # Read image path
    datas = Read_Datasets(args.Datasets)

    num = 1
    for data in tqdm(datas):
        try:
            scores = {}
            scores["Datasets"] = args.Datasets
            scores["thresh"] = args.thresh
            scores["verification-net"] = args.verification_net
            scores["erasing method"] = args.Erasing_method
            
            path1 = os.path.join(image_dir_path,data.split(' ')[0])
            path2 = os.path.join(image_dir_path,data.split(' ')[1])
            if_same = data.split(' ')[2]

            scores["image-path1"] = path1
            scores["image-path2"] = path2
            scores["match"] = int(if_same)

            # Verification image
            scores["similarity"],feature1,feature2 = Verification_path(path1,path2,verification_net,args)
            
            mask = XCos_heatmap(path1,path2,model)
            
            mask = 1 - mask

            mask[mask<1-args.thresh] = 0
            mask[mask>1-args.thresh] = 1

            Erasing_image1 = Erasing(cv2.cvtColor(np.asarray(mctnn_crop_face(Image.open(path1).convert('RGB'), BGR2RGB=False)),cv2.COLOR_RGB2BGR),mask,args.Erasing_method)
            Erasing_image2 = Erasing(cv2.cvtColor(np.asarray(mctnn_crop_face(Image.open(path2).convert('RGB'), BGR2RGB=False)),cv2.COLOR_RGB2BGR),mask,args.Erasing_method)

            cv2.imwrite(
                os.path.join(args.output_dir,args.Datasets+'-'+args.verification_net+'-'+args.Erasing_method,"Mask",str(num)+"-n1.jpg"),
                Erasing_image1
            )
            cv2.imwrite(
                os.path.join(args.output_dir,args.Datasets+'-'+args.verification_net+'-'+args.Erasing_method,"Mask",str(num)+"-n2.jpg"),
                Erasing_image2
            )

            inputs1 = Image_Preprocessing(args.verification_net, Erasing_image1)
            inputs2 = Image_Preprocessing(args.verification_net, Erasing_image2)

            feature1_erase = verification_net(torch.unsqueeze(inputs1, dim=0).cuda())
            feature2_erase = verification_net(torch.unsqueeze(inputs2, dim=0).cuda())

            scores["erase_n1--n1"] = torch.cosine_similarity(feature1_erase[0], feature1[0], dim=0).item()
            scores["erase_n1--n2"] = torch.cosine_similarity(feature1_erase[0], feature2[0], dim=0).item()

            scores["erase_n2--n2"] = torch.cosine_similarity(feature2_erase[0], feature2[0], dim=0).item()
            scores["erase_n2--n1"] = torch.cosine_similarity(feature2_erase[0], feature1[0], dim=0).item()

            scores["erase_n1--erase_n2"] = torch.cosine_similarity(feature1_erase[0], feature2_erase[0], dim=0).item()
            with open(os.path.join(args.output_dir,args.Datasets+'-'+args.verification_net+'-'+args.Erasing_method,"Json",str(num)+".json"), "w") as f:
                f.write(json.dumps(scores, ensure_ascii=False, indent=4, separators=(',', ':')))
        except:
            pass
        num = num + 1

def parse_args():
    parser = argparse.ArgumentParser(description='results in xCos Face')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        choices=['VGGFace2-train','VGGFace2-test','Celeb-A'],
                        default='VGGFace2-test',
                        help='Datasets.')
    parser.add_argument('--verification-net',
                        type=str,
                        default='CosFace-r100',
                        choices=['ArcFace-r50','ArcFace-r100',"CosFace-r50","CosFace-r100","VGGFace2-verification"],
                        help='Which network using for face verification.')
    parser.add_argument('--thresh', type=float, default=0.6,
                        help='Thresh.')
    parser.add_argument('--Erasing-method', type=str, default="random",
                        choices=["black","white","mean","random"],
                        help='Which method to erasing.')
    parser.add_argument('--output-dir', type=str, default='./results/Erasing',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)