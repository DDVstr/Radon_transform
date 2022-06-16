#! /usr/bin/env python
# -*- coding: utf-8 -*-

# name: Zhornichenko Ilya Alekseevich
# group: TMSS
# course: Image analysis

# Importing libs

import numpy as np
from PIL import Image
from PIL import ImageChops
import scipy as sc
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, ifft
import random

#correectly resampled image 
def padImage(img):
    N0, N1 = img.size
    lenDiag = int(np.ceil(np.sqrt(N0**2+N1**2)))
    imgPad = Image.new('L',(lenDiag, lenDiag))
    c0, c1 = int(round((lenDiag-N0)/2)), int(round((lenDiag-N1)/2)) 
    imgPad.paste(img, (c0,c1)) 
    return imgPad, c0, c1
# getting sinogramm
def getProj(img, theta):
    numAngles = len(theta)
    sinogram = np.zeros((img.size[0],numAngles))
    for n in range(numAngles):
        rotImgObj = img.rotate(90-theta[n], resample=Image.BICUBIC)
        sinogram[:,n] = np.sum(rotImgObj, axis=0)
    return sinogram

#ramp filter for sinogram
def rmp_filter(sino):
    a = 0.1
    projLen, numAngles = sino.shape
    step = 2*np.pi/projLen
    w = np.arange(-np.pi, np.pi, step)
    if len(w) < projLen:
        w = np.concatenate([w, [w[-1]+step]]) 
                                              
    rn1 = abs(2/a*np.sin(a*w/2));  
    rn2 = np.sin(a*w/2)/(a*w/2);  
    r = rn1*(rn2)**2;              
    
    filt = fftshift(r)   
    filtered_sinogram = np.zeros((projLen, numAngles))
    for i in range(numAngles):
        projfft = fft(sino[:,i])
        filtProj = projfft*filt
        filtered_sinogram[:,i] = np.real(ifft(filtProj))

    return filtered_sinogram
#fbp reconstruction algoritm
def backproject(sinogram, theta):
    
    imageLen = sinogram.shape[0]
    reconMatrix = np.zeros((imageLen, imageLen))
    
    x = np.arange(imageLen)-imageLen/2 
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    plt.ion()
    fig2, ax = plt.subplots()
    im = plt.imshow(reconMatrix, cmap='gray')

    theta = theta*np.pi/180
    numAngles = len(theta)

    for n in range(numAngles):
        Xrot = X*np.sin(theta[n])-Y*np.cos(theta[n]) 
        XrotCor = np.round(Xrot+imageLen/2) 
        XrotCor = XrotCor.astype('int')
        projMatrix = np.zeros((imageLen, imageLen))
        m0, m1 = np.where((XrotCor >= 0) & (XrotCor <= (imageLen-1))) 
        s = sinogram[:,n] 
        projMatrix[m0, m1] = s[XrotCor[m0, m1]]  
        reconMatrix += projMatrix
    plt.close()
    plt.ioff()
    backprojArray = np.flipud(reconMatrix)
    return backprojArray

def delimeter(ramp_filtered_sino, delimeter):
    ramp_filtered_sino[:,::delimeter] = 255
    return ramp_filtered_sino
