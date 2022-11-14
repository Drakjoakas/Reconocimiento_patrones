from roipoly import  MultiRoi
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import cv2

MASK_SAVE_PATH = r'mascaras/entrenamiento1/'

# imagen = plt.imread('comida/Entrenamiento1.jpg')
imagen = Image.open('comida/entrenamiento1.jpg')
imagen = imagen.filter(ImageFilter.GaussianBlur(3))
imagen = np.array(imagen)

#Captura Mascara Chiles
plt.imshow(imagen)
multiroi = MultiRoi(roi_names=["chile1","chile2","chile3"])

mask = np.zeros(imagen.shape[:2],dtype='uint8')

for name,roil in multiroi.rois.items():
  tmp = roil.get_mask(imagen[:,:,0])
  mask += tmp.astype('uint8')

chiles = cv2.bitwise_and(imagen,imagen,mask=mask)
plt.imsave(MASK_SAVE_PATH+'chiles.png',chiles)