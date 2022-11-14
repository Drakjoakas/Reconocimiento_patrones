from roipoly import  MultiRoi
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def crearMascara(image: str, destino: str, elemento: str):
  
  if(not os.path.isfile(image)): return
  
  imagen = Image.open(image)
  imagen = imagen.filter(ImageFilter.GaussianBlur(3.))
  imagen = np.array(imagen)
  
  plt.title('Selecciona '+elemento)
  plt.imshow(imagen)
  multiroi = MultiRoi()

  mask = np.zeros(imagen.shape[:2],dtype='uint8')

  for _,roi in multiroi.rois.items():
    tmp = roi.get_mask(imagen[:,:,0])
    mask += tmp.astype('uint8')

  if not os.path.exists(destino):
    os.mkdir(destino)
  imagen_filtrada = cv2.bitwise_and(imagen,imagen,mask=mask)
  plt.imsave(destino+ '/' + elemento+'.png',imagen_filtrada)
  
elementos = ['chile','huevo','platano']
destino   = r'./mascaras_v2/'
imagenes  = [r'Entrenamiento1.jpg',r'Entrenamiento2.jpg']#,r'Entrenamiento3.jpg',r'Entrenamiento4.jpg']

for img in imagenes:
  for elem in elementos:
    crearMascara(r'./comida/'+img,destino+img.split('.')[0],elem)