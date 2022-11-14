# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 03:25:59 2022

@author: deida
"""
from scipy import misc
import imageio as io
import numpy as np
from PIL import Image
import matplotlib.pylab as plt 
import pydicom as dicom
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#Despliegue Imagen Rosa


img = np.fromfile("rosa800x600.raw", dtype=np.uint8)
img = img.reshape((800,600))
plt.imshow(img)
print("rosa800x600")
print(img.dtype)
print(img.shape)
print(img.size)

#Despliegue Imagen Anonimized

image_path = 'C:/users/deida/Downloads/Imagenes/Imagenes/Anonymized20200210.dcm'
ds = dicom.dcmread(image_path)

plt.imshow(ds.pixel_array)
print("Anonumized")
print(ds.pixel_array.shape)
print(ds.pixel_array.dtype)

#Despliegue LenaColor

I = Image.open("lena_color_512.tif")
img=mpimg.imread('lena_color_512.tif')
imgplot = plt.imshow(img)
plt.imshow(np.asarray(I))
plt.show()
print("lena_color")
print(img.dtype)
print(img.shape)
print(I.format)

#Despliegue IM

image_path = 'C:/users/deida/Downloads/Imagenes/Imagenes/IM-0001-0007.dcm'


ds = dicom.dcmread(image_path)
plt.imshow(ds.pixel_array)
print("dcm2")
print(ds.pixel_array.shape)
print(ds.pixel_array.dtype)

#Despliegue Lake
I = Image.open("lake.tif")
img=mpimg.imread('lake.tif')
imgplot = plt.imshow(img)
print("lake")
print(img.dtype)
print(img.dtype)
print(I.format)

#Despliegue Cameraman
I = Image.open("cameraman.tif")
img=mpimg.imread('cameraman.tif')
imgplot = plt.imshow(img)
print("cameraman")
print(img.dtype)
print(img.shape)
#print(I.size)
print(I.format)


#Despliegue House
I = Image.open("house.tif")
img=mpimg.imread('house.tif')
imgplot = plt.imshow(img)
print("house")
print(img.dtype)
print(img.shape)
print(I.format)

#Despliegue pepers
I = Image.open("peppers_color.tif")
img=mpimg.imread('peppers_color.tif')
print("peppers")
imgplot = plt.imshow(img)
print(img.dtype)
print(img.shape)
print(I.format)




