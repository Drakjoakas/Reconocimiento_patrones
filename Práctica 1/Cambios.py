# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 00:17:26 2022

@author: deida
"""

import cv2
import numpy as np
import matplotlib.pylab as plt
from skimage import io
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray
import skimage
from PIL import Image
import skimage


#RGB a grises Scikitimage
path = "lena_color_512.tif"
orig_img = io.imread(path)
grayscale_img = rgb2gray(orig_img)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()
ax[0].imshow(orig_img)
ax[0].set_title("Original image")
ax[1].imshow(grayscale_img, cmap=plt.cm.gray)
ax[1].set_title("Grayscale image")
fig.tight_layout()
plt.show()

#RGB a grises OpenCv
img_gray_mode = cv2.imread("lena_color_512.tif", cv2.IMREAD_GRAYSCALE)
cv2.imshow('diff', img_gray_mode)
cv2.waitKey()




# RGB a YUV OpenCv
img_bgr = cv2.imread("lena_color_512.tif")

img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
cv2.imshow("original", img_yuv)
cv2.waitKey(0)

# RGB a YUV scikit

path = "lena_color_512.tif"
orig_img = io.imread(path)
img=skimage.color.rgb2yuv(orig_img)
plt.imshow(img)
plt.show()

# RGB a HSV
path = "lena_color_512.tif"
orig_img = io.imread(path)
img= skimage.color.rgb2hsv(orig_img)
plt.imshow(img)



# Paletta de colores
path = "lena_color_512.tif"
orig_img = io.imread(path)
plt.imshow(orig_img)
plt.colorbar(cmap = 'rgb')





