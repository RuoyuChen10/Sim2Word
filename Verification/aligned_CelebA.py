import cv2
import numpy as np
import os
import random
from PIL import Image
from tqdm import tqdm
import math

from mtcnn_pytorch.crop_and_aligned import mctnn_crop_face

CelebA_image_path="/home/cry/data2/CelebA/Img/img_align_celeba"

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

# # load input images and corresponding 5 landmarks
# def load_img_and_box(img_path, detector):
#     #Reading image
#     image = Image.open(img_path)
#     if img_path.split('.')[-1]=='png':
#         image = image.convert("RGB")
#     # BGR
#     img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#     #Detect 5 key point
#     face = detector.detect_faces(img)[0]
#     box = face["box"]
#     image = cv2.imread(img_path)
#     return image, box

# def box_crop(image, box):
#     shape=(224,224)
#     print(image.shape)
#     print((int(np.floor(box[1]-box[3]*0.15)),int(np.ceil(box[1]+box[3]*1.15))))
#     print((int(np.floor(box[0]-box[2]*0.15)),int(np.ceil(box[0]+box[2]*1.15))))

#     if int(np.floor(box[1]-box[3]*0.15)) < 0:
#         top = 0
#         top_ = -int(np.floor(box[1]-box[3]*0.15))
#     else:
#         top = int(np.floor(box[1]-box[3]*0.15))
#         top_ = 0
#     if int(np.ceil(box[1]+box[3]*1.15)) > image.shape[0]:
#         bottom = image.shape[0]
#     else:
#         bottom = int(np.ceil(box[1]+box[3]*1.15))
#     if int(np.floor(box[0]-box[2]*0.15)) < 0:
#         left = 0
#         left_ = -int(np.floor(box[0]-box[2]*0.15))
#     else:
#         left = int(np.floor(box[0]-box[2]*0.15))
#         left_ = 0
#     if int(np.ceil(box[0]+box[2]*1.15)) > image.shape[1]:
#         right = image.shape[1]
#     else:
#         right = int(np.ceil(box[0]+box[2]*1.15))
#     img_zero = np.zeros((int(np.ceil(box[1]+box[3]*1.15))-int(np.floor(box[1]-box[3]*0.15)),
#                          int(np.ceil(box[0]+box[2]*1.15))-int(np.floor(box[0]-box[2]*0.15)),
#                          3))
#     img = image[
#         top:bottom,
#         left:right]
#     img_zero[top_:top_+bottom-top,left_:left_+right-left] = img
#     img = img_zero
#     im_shape = img.shape[:2]
#     ratio = float(shape[0]) / np.min(im_shape)
#     img = cv2.resize(
#         img,
#         dsize=(math.ceil(im_shape[1] * ratio),   # width
#             math.ceil(im_shape[0] * ratio))  # height
#         )
#     new_shape = img.shape[:2]
#     h_start = (new_shape[0] - shape[0])//2
#     w_start = (new_shape[1] - shape[1])//2
#     img = img[h_start:h_start+shape[0], w_start:w_start+shape[1]]
#     return img

with open("./text/CelebA.txt", "r") as f:
    datas = f.read().split('\n')

# with tf.device('gpu:0'):
#     detector = MTCNN()
mkdir(os.path.join("./dataset/CelebA-test2"))

for image_paths in tqdm(datas):
    try:
        path1 = os.path.join(CelebA_image_path,image_paths.split(' ')[0])
        path2 = os.path.join(CelebA_image_path,image_paths.split(' ')[1])
        
        for image_path in [path1,path2]:
            image = Image.open(image_path).convert('RGB')
            image = mctnn_crop_face(image, BGR2RGB=False)
            image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
            # img,box = load_img_and_box(image_path, detector)
            # img_ = box_crop(img, box)
            
            cv2.imwrite(os.path.join("./dataset/CelebA-test2",image_path.split('/')[-1]),image)
    except:
        pass