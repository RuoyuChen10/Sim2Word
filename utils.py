# -*- coding: utf-8 -*-  

"""
Created on 2021/02/03
Update  on 2021/04/30   Add VGGFace2_verification, remove hook

@author: Ruoyu Chen
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from collections import OrderedDict

root_path = "/home/cry/data1/Counterfactual_interpretable/"

Face_attributes_name = np.array([
    "Gender","Age","Race","Bald","Wavy Hair",
    "Receding Hairline","Bangs","Sideburns","Hair color","no beard",
    "Mustache","5 o Clock Shadow","Goatee","Oval Face","Square Face",
    "Round Face","Double Chin","High Cheekbones","Chubby","Obstructed Forehead",
    "Fully Visible Forehead","Brown Eyes","Bags Under Eyes","Bushy Eyebrows","Arched Eyebrows",
    "Mouth Closed","Smiling","Big Lips","Big Nose","Pointy Nose"
])

def Read_Datasets(Datasets_type):
    '''
    Read the path in different datasets
    '''
    if Datasets_type == "VGGFace2-train":
        dataset_list = os.path.join(root_path, "Verification/text/VGGFace2-train.txt")
        # dataset_list = "./Verification/text/VGGFace2-train.txt"
        if os.path.exists(dataset_list):
            with open(dataset_list, "r") as f:
                datas = f.read().split('\n')
        else:
            raise ValueError("File {} not in path".format(dataset_list))
    elif Datasets_type == "VGGFace2-test":
        dataset_list = os.path.join(root_path, "Verification/text/VGGFace2-test.txt")
        # dataset_list = "./Verification/text/VGGFace2-test.txt"
        if os.path.exists(dataset_list):
            with open(dataset_list, "r") as f:
                datas = f.read().split('\n')
        else:
            raise ValueError("File {} not in path".format(dataset_list))
    elif Datasets_type == "Celeb-A":
        dataset_list = os.path.join(root_path, "Verification/text/CelebA.txt")
        # dataset_list = "./Verification/text/CelebA.txt"
        if os.path.exists(dataset_list):
            with open(dataset_list, "r") as f:
                datas = f.read().split('\n')
        else:
            raise ValueError("File {} not in path".format(dataset_list))
    elif Datasets_type == "Test":
        dataset_list = os.path.join(root_path, "External-Experience/The_most_special_attribute/List.txt")
        # dataset_list = "./External-Experience/The_most_special_attribute/List.txt"
        if os.path.exists(dataset_list):
            with open(dataset_list, "r") as f:
                datas = f.read().split('\n')
        else:
            raise ValueError("File {} not in path".format(dataset_list))
    return datas

class VGGFace2_verifacation(object):
    def __init__(self,net):
        self.net = net
        self.net.eval()
        self.hook_feature = None
    def _register_hook(self,net,layer_name):
        for (name, module) in net.named_modules():
            if name == layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
    def _get_features_hook(self,module, input, output):
        self.hook_feature = output.view(output.size(0), -1)[0]
    def remove_handlers(self):
        # The hook should removed or the cuda memory will accumulate
        for handle in self.handlers:
            handle.remove()
    def __call__(self,inputs):
        self.handlers = []
        self._register_hook(self.net,"avgpool")
        self.net.zero_grad()
        self.net(inputs)
        output = torch.unsqueeze(self.hook_feature, dim=0)
        self.remove_handlers()
        return output

def get_network(command,weight_path=None):
    '''
    Get the object network
        command: Type of network
        weight_path: If need priority load the pretrained model?
    '''
    # Load model
    if weight_path is not None and os.path.exists(weight_path):
        model = torch.load(weight_path)
        try:
            # if multi-gpu model:
            model = model.module
        except:
            # just 1 gpu or cpu
            pass
        pretrain = model.state_dict()
        new_state_dict = {}
        for k,v in pretrain.items():
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)
        print("Model parameters: " + weight_path + " has been load!")
        return model
    elif command == "resnet50":
        from models.resnet import resnet50
        print("Model load: ResNet50 as backbone.")
        return resnet50()
    elif command == 'VGGFace2':
        from models.vggface_models.resnet import resnet50
        weight_path = os.path.join(root_path, "pre-trained/resnet50_scratch_weight.pkl")
        # weight_path = "./pre-trained/resnet50_scratch_weight.pkl"
        net = resnet50(num_classes=8631)
        with open(weight_path, 'rb') as f:
            obj = f.read()
            weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        net.load_state_dict(weights, strict=True)
        return net
    elif command == 'VGGFace2-Se':
        from models.vggface_models.senet import senet50
        weight_path = os.path.join(root_path, "pre-trained/senet50_ft_weight.pkl")
        # weight_path = "./pre-trained/resnet50_scratch_weight.pkl"
        net = senet50(num_classes=8631)
        with open(weight_path, 'rb') as f:
            obj = f.read()
            weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        net.load_state_dict(weights, strict=True)
        return net
    elif command == "ArcFace-r50":
        from Verification.iresnet import iresnet50
        arcface_r50_path = os.path.join(root_path, "Verification/pretrained/ms1mv3_arcface_r50_fp16/backbone.pth")
        # arcface_r50_path = "./Verification/pretrained/ms1mv3_arcface_r50_fp16/backbone.pth"
        net = iresnet50()
        net.load_state_dict(torch.load(arcface_r50_path))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()
        return net 
    elif command == "ArcFace-r100":
        from Verification.iresnet import iresnet100
        arcface_r100_path = os.path.join(root_path, "Verification/pretrained/ms1mv3_arcface_r100_fp16/backbone.pth")
        # arcface_r100_path = "./Verification/pretrained/ms1mv3_arcface_r100_fp16/backbone.pth"
        net = iresnet100()
        net.load_state_dict(torch.load(arcface_r100_path))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()
        return net
    elif command == "CosFace-r50":
        from Verification.iresnet import iresnet50
        cosface_r50_path = os.path.join(root_path, "Verification/pretrained/glint360k_cosface_r50_fp16_0.1/backbone.pth")
        # cosface_r50_path = "./Verification/pretrained/glint360k_cosface_r50_fp16_0.1/backbone.pth"
        net = iresnet50()
        net.load_state_dict(torch.load(cosface_r50_path))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()
        return net
    elif command == "CosFace-r100":
        from Verification.iresnet import iresnet100
        cosface_r100_path = os.path.join(root_path, "Verification/pretrained/glint360k_cosface_r100_fp16_0.1/backbone.pth")
        # cosface_r100_path = "./Verification/pretrained/glint360k_cosface_r100_fp16_0.1/backbone.pth"
        net = iresnet100()
        net.load_state_dict(torch.load(cosface_r100_path))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()
        return net
    elif command == "VGGFace2-verification":
        from models.vggface_models.resnet import resnet50
        weight_path = os.path.join(root_path, "pre-trained/resnet50_scratch_weight.pkl")
        # weight_path = "./pre-trained/resnet50_scratch_weight.pkl"
        vggnet = resnet50(num_classes=8631)
        with open(weight_path, 'rb') as f:
            obj = f.read()
            weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        vggnet.load_state_dict(weights, strict=True)

        if torch.cuda.is_available():
            vggnet.cuda()
        net = VGGFace2_verifacation(vggnet)
        return net
    elif command == "VGGFace2-ArcFace":
        """
        create on 2022/1/13

        recognition network
        """
        from models.vggface_arcface import iresnet50_arcface

        pre_trained = os.path.join(root_path, "pre-trained/vggface2-arc.pth")

        model = iresnet50_arcface(people_num=8631)

        if pre_trained is not None and os.path.exists(pre_trained):
            # No related
            model_dict = model.state_dict()
            pretrained_param = torch.load(pre_trained)
            try:
                pretrained_param = pretrained_param.state_dict()
            except:
                pass

            new_state_dict = OrderedDict()
            for k, v in pretrained_param.items():
                if k in model_dict:
                    new_state_dict[k] = v
                    print("Load parameter {}".format(k))
                elif k[7:] in model_dict:
                    new_state_dict[k[7:]] = v
                    print("Load parameter {}".format(k[7:]))

            model_dict.update(new_state_dict)
            model.load_state_dict(model_dict)
            print("Success load pre-trained model {}".format(pre_trained))

        return model
    elif command == "VGGFace2-evidential":
        """
        create on 2022/1/13

        evidential network
        """
        from models.iresnet_edl import iresnet50, iresnet100

        pre_trained = os.path.join(root_path, "pre-trained/vggface-evidential.pth")

        model = iresnet50(people_num = 8631)

        if pre_trained is not None and os.path.exists(pre_trained):
            # No related
            model_dict = model.state_dict()
            pretrained_param = torch.load(pre_trained)
            try:
                pretrained_param = pretrained_param.state_dict()
            except:
                pass

            new_state_dict = OrderedDict()
            for k, v in pretrained_param.items():
                if k in model_dict:
                    new_state_dict[k] = v
                    print("Load parameter {}".format(k))
                elif k[7:] in model_dict:
                    new_state_dict[k[7:]] = v
                    print("Load parameter {}".format(k[7:]))

            model_dict.update(new_state_dict)
            model.load_state_dict(model_dict)
            print("Success load pre-trained model {}".format(pre_trained))
        return model
    
def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def Erasing(image, mask, method):
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

def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

class INFO():
    """
    Logging file
    """
    def __init__(self, save_log):
        self.save_log=save_log
        if os.path.exists(save_log):
            os.remove(save_log)

    def __call__(self, string):
        print(string)
        if self.save_log != None:
            with open(self.save_log,"a") as file:
                file.write(string)
                file.write("\n")