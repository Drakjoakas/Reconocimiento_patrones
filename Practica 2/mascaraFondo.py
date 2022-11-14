import matplotlib.pyplot as plt
import numpy as np
import cv2

for tr in range(1,5):
  tr = str(tr)
  chile = plt.imread('mascaras_v2/Entrenamiento'+tr+'/chile.png')
  huevo = plt.imread('mascaras_v2/Entrenamiento'+tr+'/huevo.png')
  platano = plt.imread('mascaras_v2/Entrenamiento'+tr+'/platano.png')
  comida = plt.imread('./comida/Entrenamiento'+tr+'.jpg')
  img = chile + huevo + platano   

  # img = img.astype('uint8')
  img0 = img[:,:,0]
  mask = np.zeros(img0.shape,dtype='uint8')

  for i in range(len(img0)):
    for j in range(len(img0)):
      mask[i,j] = 255 if img0[i,j] == 0 else 0
  # msk = cv2.bitwise_and(comida1,comida1,mask=img)

  fondo = cv2.bitwise_and(comida,comida,mask=mask)
  plt.imsave('./mascaras_v2/Entrenamiento'+tr+'/fondo.png',fondo)

