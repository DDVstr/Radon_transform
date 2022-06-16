#! /usr/bin/env python
# -*- coding: utf-8 -*-

# name: Zhornichenko Ilya Alekseevich
# group: TMSS
# course: Image analysis

# Importing libs

from cv2 import detail_ImageFeatures
from pyparsing import delimited_list
import scipy
import scipy.ndimage as misc
import numpy as np
import matplotlib.pyplot as plt
from radon import process
import skimage.transform as tr
from fbp import rmp_filter , backproject, padImage, getProj,delimeter
from PIL import Image
import cv2
import random
import nitcography as nit


def radon_transform(image,_dim):
    R = np.zeros((_dim, len(image)), dtype='float64')
    for s in range(_dim):
        rotation = tr.rotate(image, -s * 180 /_dim).astype('float64')
        R[:, s] = sum(rotation)
    R = R/np.max(R)
    return R

#setting the angleS
delta = 1
theta = np.arange(0,181,delta)

# Read image as 64bit float gray scale
#image_process = process('/Users/izhora/Desktop/dali.jpg')
image = cv2.imread('/Users/izhora/Desktop/dali.jpg')
_dim = image.shape[1]
print("size",_dim)


#radon transform
radon = radon_transform(image,_dim)

print("radon image size", radon.shape)


#fbp reconstruction
myImg = Image.open('/Users/izhora/Desktop/dali.jpg').convert('L')
myImgPad, c0, c1 = padImage(myImg)  
dTheta = 1
theta = np.arange(0,181,dTheta)
mySino = getProj(myImg, theta)  
###############
filtSino = rmp_filter(mySino) 
"""ramp - filtering of sinogram"""
radon_filtered_sino = rmp_filter(radon)

filtered_img = backproject(filtSino, theta)
devided_sino = delimeter(filtSino,2)

#################
filtered = np.round((filtered_img-np.min(filtered_img))/np.ptp(filtered_img)*255) #convert values to integers 0-255
reconImg = Image.fromarray(filtered.astype('uint8'))
filtSino = Image.fromarray(filtered.astype('uint8'))
n0, n1 = myImg.size
reconImg = reconImg.crop((c0, c1, c0+n0, c1+n1))

####____nitcography___#####
nit.img_size_hanlder()
nit.normalization(nit.img_size_hanlder()[0])
nit.normalization_1(nit.img_size_hanlder()[0])
nit.splitter(nit.img_size_hanlder()[0],nit.img_size_hanlder()[1])
nit.devided(devided_sino)
nit.filt_sino(nit.devided(devided_sino))
a = nit.nitcography_project(nit.devided(devided_sino))

"""devided image"""

# Plot the original and the radon transformed image

"""original image"""
plt.subplot(1,4, 1), plt.imshow(image, cmap='gray')
plt.title("original photo")
plt.xticks([]), plt.yticks([])

"""sinogram"""
plt.subplot(1, 4, 2), plt.imshow(radon, cmap='gray') 
plt.title("sinogram")
plt.xticks([]), plt.yticks([])
plt.xlim(0,image.shape[0])
plt.ylim(0,image.shape[1])

"""backprojected image"""
plt.subplot(1,4,3), plt.imshow(radon_filtered_sino, cmap='gray')
plt.title("RMP convolution")
plt.xticks([])
plt.yticks([])

"""ramp-filter sinogram"""
"""backprojected image"""
plt.subplot(1,4,4), plt.imshow(a, cmap='gray')
plt.title("RMP convolution")
plt.xticks([])
plt.yticks([])

plt.show()

"""devided image"""
#plt.subplot(1,5,5), plt.imshow(devided_sino, cmap='gray') #filtSino
#plt.title("devided rmp_filter")
#plt.xticks([])
#plt.yticks([])
#plt.xlim(0,filtSino.shape[1])
#plt.ylim(0,filtSino.shape[0])
#plt.show()

"""
saving sinogram to main dir
def sinogram_dir(radon): 
    
    _sino_dir = '/Users/izhora/Desktop/sino_binary.jpg'
    #default size of sinograms
    img = Image.open(radon) 
    left = 0
    top = 0
    right = 681
    bottom = 681
    radon_res= img.crop((left, top, right, bottom)) 
    cv2.imwrite(_sino_dir, radon_res)  
    return _sino_dir

execute"""
"""sinogram_dir(radon)"""




