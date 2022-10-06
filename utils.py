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
import cv2

from collections import OrderedDict

from skimage import transform as trans

root_path = "/home/cry/data1/Sim2Word/"

Face_attributes_name = np.array([
    "Gender","Age","Race","Bald","Wavy Hair",
    "Receding Hairline","Bangs","Sideburns","Hair color","no beard",
    "Mustache","5 o Clock Shadow","Goatee","Oval Face","Square Face",
    "Round Face","Double Chin","High Cheekbones","Chubby","Obstructed Forehead",
    "Fully Visible Forehead","Brown Eyes","Bags Under Eyes","Bushy Eyebrows","Arched Eyebrows",
    "Mouth Closed","Smiling","Big Lips","Big Nose","Pointy Nose"
])

Gender = ["Male","Female"]
Age = ["Young","Middle Aged","Senior"]
Race = ["Asian","White","Black"]
Hair_color = ["Black Hair","Blond Hair","Brown Hair","Gray Hair","Unknown Hair"]

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

class FaceAlignment():
    """
    demo of face alignment
    """
    def __init__(self):
        super(FaceAlignment, self).__init__()
        # init templete src
        self.init_src()
    
    def init_src(self):
        """
        src us template, for the five common face position templates.
        """
        src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
        #<--left
        src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                        [45.177, 86.190], [64.246, 86.758]],
                        dtype=np.float32)

        #---frontal
        src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                        [42.463, 87.010], [69.537, 87.010]],
                        dtype=np.float32)

        #-->right
        src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                        [48.167, 86.758], [67.236, 86.190]],
                        dtype=np.float32)

        #-->right profile
        src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                        [55.388, 89.702], [61.257, 89.050]],
                        dtype=np.float32)

        src = np.array([src1, src2, src3, src4, src5])

        arcface_src = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32)

        self.src_map = {112: src, 224: src * 2}
        self.arcface_src = np.expand_dims(arcface_src, axis=0)

    def estimate_norm(self, lmk, image_size=112, mode='arcface'):
        """
        Estimate the warp matrix
        """
        assert lmk.shape == (5, 2)
        tform = trans.SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = float('inf')
        if mode == 'arcface':
            assert image_size == 112
            src = self.arcface_src
        else:
            src = self.src_map[image_size]
        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
            M = tform.params[0:2, :]
            results = np.dot(M, lmk_tran.T)
            results = results.T
            error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
            #         print(error)
            if error < min_error:
                min_error = error
                min_M = M
                min_index = i
        return min_M, min_index
    
    def norm_crop(self, img, landmark, image_size=112, mode='arcface'):
        """
        Warp the face, align
        """
        M, pose_index = self.estimate_norm(landmark, image_size, mode)
        # warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        warped = cv2.warpAffine(img, M, (image_size, image_size),  None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
        return warped

    def __call__(self, img, landmark, image_size=112, mode='arcface'):
        """
        img: The input image, BGR format
        landmark: (5, 2)
        image_size: crop box size
        mode: usually is arcface, other mode can has image_size 224
        """
        wrap = self.norm_crop(img, landmark, image_size=image_size, mode=mode)
        return wrap