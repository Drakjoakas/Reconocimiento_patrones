from roipoly import RoiPoly, MultiRoi
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage

MASK_SAVE_PATH = r'mascaras/entrenamiento2/'

imagen = plt.imread('comida/Entrenamiento2.jpg')
imagen = ndimage.gaussian_filter(imagen,3)

def crearMascara(elemento:str):
  plt.title('Selecciona '+elemento)
  plt.imshow(imagen)
  multiroi = MultiRoi()

  mask = np.zeros(imagen.shape[:2],dtype='uint8')

  for name,roi in multiroi.rois.items():
    tmp = roi.get_mask(imagen[:,:,0])
    mask += tmp.astype('uint8')

  imagen_filtrada = cv2.bitwise_and(imagen,imagen,mask=mask)
  plt.imsave(MASK_SAVE_PATH+elemento+'.png',imagen_filtrada)


#mascaras huevos
crearMascara('huevos')

#mascara platanos
crearMascara('platanos')

#mascara chiles
crearMascara('chiles')