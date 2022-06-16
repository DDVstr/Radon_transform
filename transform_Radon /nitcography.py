#! /usr/bin/env python
# -*- coding: utf-8 -*-

# name: Zhornichenko Ilya Alekseevich
# group: TMSS
# course: Image analysis

# Importing libs
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from radon import process as pr
import cv2
import fbp  as f
import random

# method returns converted and cropped image due to conditions
from PIL import Image 
import numpy as np

def mask_settler(img_size):

    mask  = np.zeros((img_size, img_size))
    centered = (img_size/2 + 1)
    radius = int(img_size / 2 + 1)
    
    for i in range(img_size):
        for j in range(img_size ):
            if (i - centered[0]) ** 2 + (j - centered[0]) ** 2 < radius ** 2:
                mask[i][j] = 1
            else:
                mask[i][j] = 0
    return mask



def img_size_hanlder():
    np.set_printoptions(threshold=50)

    img = Image.open(r"/Users/izhora/Desktop/hd.jpeg") 

    print(img.size[0])

    if img.size[0] < img.size[1]:

        left = 0
        top = 0
        right = img.size[0]
        bottom = img.size[0]
    
    else:
        left = 0
        top = 0 
        right = img.size[1]
        bottom = img.size[1]

    
    img_res = img.crop((left, top, right, bottom)) 
    img  = img.crop((left, top, right, bottom)) 
    print(img.size)


    pix = img.load()
    #img_res.show() 
    max_value = np.max(img_res)

    print(max_value)
    img_res = np.array(img_res)
    img_res.shape
    a = float(1/max_value)
    img_res_1 = img_res
    img_res = img_res/255
    return img_res, img_res_1

def normalization(img_res):
    for i in range(img_res.shape[0]): 
        for j in range(0,5,img_res.shape[1]):
            img_res[i][j]=(img_res[i][j]/np.max(img_res)).astype(np.float32)
    return img_res
def normalization_1(img_res):
    for i in range(img_res.shape[0]): 
        for j in range(0,1,img_res.shape[1]):
            img_res[i][j]=(img_res[i][j]+1).astype(np.float32)
    return img_res

#Добавим прорежение свертки с рамп-фильтром
def splitter(img_res,img_res_1):
    img_res[:,::12]=1
    img_res_1[:,::12]=1

def devided(devided_sino):
    dev = np.array(devided_sino)
    dev = dev/255
    print("normalized dev ", dev )
    dev.shape[0]
    for i in range(dev.shape[0]): 
        for j in range(dev.shape[1]):
            if dev[i][j]<0: 
                dev[i][j] = 0
            else:
                dev[i][j] = 1
    return dev

def filt_sino(dev):
    filtSino = f.rmp_filter(dev) 
    filtSino = Image.fromarray(filtSino.astype('uint8'))
    print(filtSino.rotate(180))
    dev_1 = np.array(filtSino)
    return dev_1

def nitcography_project(dev):
    filtSino = f.rmp_filter(dev) 
    theta = np.arange(0,181,1)
    filtSino = Image.fromarray(filtSino.astype('uint8'))
    print(filtSino.rotate(180))
    dev_1 = np.array(filtSino)
    nitkography = f.backproject(dev,theta)
    filtered = np.round((nitkography-np.min(nitkography))/np.ptp(nitkography)*255)
    reconImg = Image.fromarray(filtered.astype('uint8'))
    plt.imshow(reconImg,cmap = 'gray')
    plt.title("nitcography projection method")
    return reconImg

def nit_process(img_file, angles, strs, str_t = None):
    
    if str_t is None:
        str_t = np.sqrt(angles * strs)

    imgs, msk = prepare_image(img_file)
    if imgs is None:
        return None, None
    if len(imgs) == 4:
        imgs = imgs[:3]
    rads = []
    sverts = []
    strings = []
    res_img = []
    
    layer = 1
    for el in imgs: # iterate 1 time is gray-scaled, 3 times if colored
        print("Layer", layer, "out of", len(imgs))
        layer += 1
        rad = radon(el, angles)
        rads.append(rad)
        
        svert = f.rmp_filter(rad)
        sverts.append(np.clip(svert, 0, 1))
        
        string = calculate_strings(svert, strs)
        strings.append(string)
        
        res = get_im(string, msk, str_t)
        
        res_img.append(res)
    
    fin_image = np.array(res_img)
    fin_image = np.moveaxis(fin_image, 0, -1)
    fin_image /= np.max(fin_image)
    
    return fin_image, (rads, sverts, strings)    


