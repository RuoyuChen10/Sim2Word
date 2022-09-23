import os
import sys
import torch
import PIL
from PIL import Image
import numpy as np
import cv2

from tqdm import tqdm

sys.path.append('../python')
from xfr.models.whitebox import WhiteboxSTResnet, Whitebox
from xfr.models.resnet import stresnet101
import xfr.models.whitebox
import xfr.models.lightcnn
import xfr.show

sys.path.append('../python/strface')
import strface.detection

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def _detector(imgfile):
    """Faster RCNN face detector wrapper"""
    im = np.array(PIL.Image.open(imgfile))
    if torch.cuda.is_available():
        gpu_index = 0
    else:
        gpu_index = -1
    net = strface.detection.FasterRCNN(model_dir='../python/strface/models/detection', gpu_index=gpu_index, conf_threshold=None, rotate_flags=None, rotate_thresh=None, fusion_thresh=None, test_scales=800, max_size=1300)
    return net(im)

def _blend_saliency_map(img, smap, scale_factor=1.0, gamma=0.3, blur_sigma=0.05):
    """Input is PIL image, smap is real valued saliency map, output is PIL image"""
    img_blend = xfr.show.blend_saliency_map(np.array(img).astype(np.float32)/255.0, smap, blur_sigma=blur_sigma, gamma=gamma, scale_factor=scale_factor)
    return PIL.Image.fromarray(np.uint8(img_blend*255))  # [0,255] PIL image

def _encode_triplet_test_cases(wb, f_probe, f_nonmate, f_mate):
    """Run face detection and network encoding for standardized (mate, non-mate, probe) triplets for testing"""

    

    x_mate = wb.net.encode(wb.net.preprocess(f_mate))
    x_nonmate = wb.net.encode(wb.net.preprocess(f_nonmate))
    img_probe = wb.net.preprocess(f_probe)  # torch tensor

    return (x_mate, x_nonmate, img_probe)

def gen_cam(image_dir, mask):
    """
    Generate heatmap
        :param image: [H,W,C]
        :param mask: [H,W],range 0-1
        :return: tuple(cam,heatmap)
    """
    # Read image
    image = cv2.imread(image_dir)
    image = cv2.resize(image,(112,112))

    # mask->heatmap
    heatmap = cv2.applyColorMap(norm_image(mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)

    # merge heatmap to original image
    cam = 0.5*heatmap + 0.5*np.float32(image)

    return cam

def norm_image(image):
    """
    Normalization image
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image = 1-image
    image *= 255.
    return np.uint8(image)

def truncated_contrastive_triplet_ebp():
    """Truncated contrastive triplet excitation backprop"""
    print('[test_whitebox.truncated_contrastive_triplet_ebp]: Detection and encoding for (mate, non-mate, probe) triplet')
    wb = Whitebox(WhiteboxSTResnet(stresnet101('../models/resnet101v4_28NOV17_train.pth')))
    f_probe = '/home/cry/data2/VGGFace2/train_align/n003198/0356_01.jpg'   # obama
    f_nonmate = '/home/cry/data2/VGGFace2/train_align/n000538/0199_01.jpg'  # obama non-mate: inpainted probe
    f_mate = '/home/cry/data2/VGGFace2/train_align/n003198/0249_01.jpg'  # obama mate
    
    im_probe = Image.open(f_probe)
    im_nonmate = Image.open(f_nonmate)
    im_mate = Image.open(f_mate)
    
    (x_mate, x_nonmate, img_probe) = _encode_triplet_test_cases(wb, im_probe,im_nonmate,im_mate)
    wb.net.set_triplet_classifier((1.0/2500.0)*x_mate, (1.0/2500.0)*x_nonmate)  # rescale encodings to avoid softmax overflow
    img_saliency = wb.truncated_contrastive_ebp(img_probe, k_poschannel=0, k_negchannel=1, percentile=20)

    cam = gen_cam(f_probe, img_saliency)

    print(np.min(img_saliency))
    cv2.imwrite("2.jpg",cam)

if __name__ == "__main__":
    """CPU-only demos (cached output in ./whitebox/*.jpg)"""
    # Baseline whitebox discriminative visualization methods
    if torch.cuda.is_available():
        dev = torch.device(0)
    else:
        dev = torch.device("cpu")

    mkdir("./results")
    VGG_dir = "/home/cry/data2/VGGFace2/train_align/"

    wb = Whitebox(WhiteboxSTResnet(stresnet101('../models/resnet101v4_28NOV17_train.pth')))

    with open("./test.txt", "r") as f:
        datas = f.read().split('\n')

    num=0
    for data in tqdm(datas):
        num += 1
        try:
            f_probe = data.split(' ')[0]
            f_nonmate = data.split(' ')[2]
            f_mate = data.split(' ')[1]

            im_probe = Image.open(os.path.join(VGG_dir,f_probe))
            im_nonmate = Image.open(os.path.join(VGG_dir,f_nonmate))
            im_mate = Image.open(os.path.join(VGG_dir,f_mate))

            (x_mate, x_nonmate, img_probe) = _encode_triplet_test_cases(wb, im_probe,im_nonmate,im_mate)
            wb.net.set_triplet_classifier((1.0/2500.0)*x_mate, (1.0/2500.0)*x_nonmate)  # rescale encodings to avoid softmax overflow
            img_saliency = wb.truncated_contrastive_ebp(img_probe, k_poschannel=0, k_negchannel=1, percentile=20)

            cam = gen_cam(os.path.join(VGG_dir,f_probe), img_saliency)

            mkdir(os.path.join("./results/",str(num)))

            cv2.imwrite(os.path.join("./results/",str(num),"probe.jpg"),cv2.cvtColor(np.asarray(im_probe),cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join("./results/",str(num),"non-mate.jpg"),cv2.cvtColor(np.asarray(im_nonmate),cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join("./results/",str(num),"mate.jpg"),cv2.cvtColor(np.asarray(im_mate),cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join("./results/",str(num),"cam.jpg"),cv2.cvtColor(np.asarray(cam),cv2.COLOR_RGB2BGR))
        except:
            pass