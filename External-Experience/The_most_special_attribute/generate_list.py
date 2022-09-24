# -*- coding: utf-8 -*-  

"""
Created on 2021/05/20

@author: Ruoyu Chen
"""

import os 
import random

VGGFace2_train_image_path = "/home/cry/data2/VGGFace2/train_align/"

names = os.listdir(VGGFace2_train_image_path)
names.sort()

# image1 = random.choice(os.listdir(os.path.join(VGGFace2_train_image_path,names[0])))
# image1 = os.path.join(names[0],image1)

image1 = "n000805/0012_01.jpg"

for name in names:
    try:
        image2 = random.choice(os.listdir(os.path.join(VGGFace2_train_image_path,name)))
        image2 = os.path.join(name,image2)

        with open("List8.txt","a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
            file.write(image1+" "+image2+"\n")
    except:
        print(name)