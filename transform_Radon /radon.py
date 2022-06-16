#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Importing libs
import cv2
import numpy as np
from PIL import Image
import PIL.ImageOps
import random

def process(filename):
    binary = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # Preprocessed image output
    thresh_value = 128
    img_binary = cv2.threshold(binary, thresh_value, 255, cv2.IMREAD_GRAYSCALE)[1]
    _dir = '/Users/izhora/Desktop/image_binary.jpg'
    cv2.imwrite(_dir, img_binary)
    image = Image.open(_dir)
    print(image.size)

    if image.size[0] < image.size[1]:

        left = 0
        top = 0
        right = image.size[0]
        bottom = image.size[0]
    
    else:
        left = 0
        top = 0 
        right = image.size[1]
        bottom = image.size[1]
    
    img_res = image.crop((left, top, right, bottom)) 
    # Inverting Image
    inverted_image = PIL.ImageOps.invert(img_res)
    _directory = '/Users/izhora/Desktop/image_binary1.jpg'
    inverted_image.save(_directory)
    cv2.destroyAllWindows()
    # Radon transform applying
    return _directory
