from roipoly import RoiPoly, MultiRoi
import matplotlib.pyplot as plt
import numpy as np
import cv2

MASK_SAVE_PATH = r'mascaras/entrenamiento1/'

imagen = plt.imread('comida/Entrenamiento1.jpg')
plt.imshow(imagen)

my_roi = RoiPoly(color='r')

mask = my_roi.get_mask(imagen[:,:,0])
mask = mask.astype('uint8')

plt.imshow(imagen)
roi2 = RoiPoly(color='blue')

mask2 = roi2.get_mask(imagen[:,:,0])
mask2 = mask2.astype('uint8')

mask += mask2

huevos = cv2.bitwise_and(imagen,imagen, mask=mask)
huevos = cv2.cvtColor(huevos, cv2.COLOR_RGB2BGR)

cv2.imshow('Huevos',huevos)
cv2.waitKey(0)

huevos = cv2.cvtColor(huevos, cv2.COLOR_BGR2RGB)
plt.imsave(MASK_SAVE_PATH+'huevos.png',huevos)

#Captura máscar plátano
plt.imshow(imagen)
roi = RoiPoly(color='blue')
mask = roi.get_mask(imagen[:,:,0])
mask = mask.astype('uint8')

platano = cv2.bitwise_and(imagen,imagen, mask=mask)
platano = cv2.cvtColor(platano, cv2.COLOR_RGB2BGR)

cv2.imshow('platano',platano)
cv2.waitKey(0)

platano = cv2.cvtColor(platano, cv2.COLOR_BGR2RGB)
plt.imsave(MASK_SAVE_PATH+'platano.png',platano)

#Captura Mascara Chiles

plt.imshow(imagen)
multiroi = MultiRoi(roi_names=["chile1","chile2","chile3"])

mask = np.zeros(imagen.shape[:2],dtype='uint8')

for name,roil in multiroi.rois.items():
  tmp = roil.get_mask(imagen[:,:,0])
  mask += tmp.astype('uint8')

chiles = cv2.bitwise_and(imagen,imagen,mask=mask)
plt.imsave(MASK_SAVE_PATH+'chiles.png',chiles)


