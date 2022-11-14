from roipoly import  MultiRoi
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import numpy as np
import cv2
import os

# Constantes de valores estadisticos
PROMEDIO   = 'promedio'
DEV_STD    = 'desviacion_estandar'
COVARIANZA = 'covarainza'
DET_COV    = 'det_covarianza'
INV_COV    = 'inv_covarianza'
PRIORI     = 'priori'

elementos = ['chile','huevo','platano']
destino   = r'./mascaras_v2/'
imagenes  = [r'Entrenamiento1.jpg',r'Entrenamiento2.jpg',r'Entrenamiento3.jpg',r'Entrenamiento4.jpg']

colorDict = {
  'huevo': [1.0,1.0,1.0],
  'chile': [1.0,0.0,0.0],
  'fondo': [0.0,0.0,0.0],
  'platano': [0.0,1.0,0.0]
}

# Funcion para obtener las mascaras
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
  

# Funcion para obtener los datos de entrenamiento
def getTrainData(elemento: str):
  data = []
  
  for numEntrenamiento in range(1,5):
    img = plt.imread(r'./mascaras_v2/Entrenamiento'+str(numEntrenamiento)+r'/'+elemento+r'.png')
    
    for x  in range(img.shape[0]):
      for y  in range(img.shape[1]):
        if(img[x,y,0] and img[x,y,1] and img[x,y,2]):
          data.append([img[x,y,0],img[x,y,1],img[x,y,2]]) #Cada dato es un vector con valores RGB e [0,1]
        
  
  return np.array(data)

# Funcion para obtener datos estadisticos de un conjunto de datos
def getStatistics(data: np.ndarray, total: float):
  promedio    = data.mean(axis=0)
  cov         = np.cov(data.T)
  desv_std    = np.std(data)
  cov_det     = np.linalg.det(cov)
  cov_inv     = np.linalg.inv(cov)
  prob_priori = data.size / total
  
  return {
    PROMEDIO : promedio,
    COVARIANZA: cov,
    DEV_STD: desv_std,
    DET_COV: cov_det,
    INV_COV: cov_inv,
    PRIORI: prob_priori    
  }
  
# Funcion que despliega la informacion estadistica
def showStatistics(stats: dict)->None:
  print('~'*23);print(PROMEDIO);print('-'*23);print(stats[PROMEDIO])
  print('~'*23);print(COVARIANZA);print('-'*23);print(stats[COVARIANZA])
  print('~'*23);print(DEV_STD);print('-'*23);print(stats[DEV_STD])
  print('~'*23);print(DET_COV);print('-'*23);print(stats[DET_COV])
  print('~'*23);print(INV_COV);print('-'*23);print(stats[INV_COV])
  print('~'*23);print(PRIORI);print('-'*23);print(stats[PRIORI])
  
# Clase del Clasificador de Bayes
class ClasificadorBayes:
  HUEVO_CLAVE   = 'huevo'
  CHILE_CLAVE   = 'chile'
  FONDO_CLAVE   = 'fondo'
  PLATANO_CLAVE = 'platano'

  def __init__(self, huevo_st, chile_st, fondo_st, platano_st ) -> None:
    self.huevo = huevo_st
    self.chile = chile_st
    self.platano = platano_st
    self.fondo = fondo_st
    
  def getBayes(self,vector,stats: dict) -> float:
    return ((-1./2.) * np.dot((vector - stats[PROMEDIO].T) @ stats[INV_COV] , (vector - stats[PROMEDIO]))) - ((1./2.) * np.log(stats[DET_COV])) + (np.log(stats[PRIORI]))
    
  def getClass(self,vector):
    bayes = {
    self.HUEVO_CLAVE   : self.getBayes(vector,self.huevo),
    self.CHILE_CLAVE   : self.getBayes(vector,self.chile),
    self.FONDO_CLAVE   : self.getBayes(vector,self.fondo),
    self.PLATANO_CLAVE : self.getBayes(vector,self.platano)
    }
    
    return max(bayes, key=bayes.get)

# Funcion que evala los pixeles de la imagen y los asigna a una clase    
def evaluarImagen(imagen:str,clasificador: ClasificadorBayes) -> np.ndarray:
  img = plt.imread(imagen) #(600,600,3)
  resultado = np.zeros(img.shape)
  
  for x in range(img.shape[0]):
    for y in range(img.shape[1]):
      clase = clasificador.getClass([img[x,y,0]/255,img[x,y,1]/255,img[x,y,2]/255])
      color = colorDict[clase]
      resultado[x,y] = color

  
  return resultado

# Creamos la carpeta donde se guardaran las mascaras.
if not os.path.exists(destino):
  os.mkdir(destino)

# Obtencion de las mascaras
for img in imagenes:
  for elem in elementos:
    crearMascara(r'./comida/'+img,destino+img.split('.')[0],elem)
    
# Obtencion de los fondos
for tr in range(1,5):
  tr = str(tr)
  chile = plt.imread('mascaras_v2/Entrenamiento'+tr+'/chile.png')
  huevo = plt.imread('mascaras_v2/Entrenamiento'+tr+'/huevo.png')
  platano = plt.imread('mascaras_v2/Entrenamiento'+tr+'/platano.png')
  comida = plt.imread('./comida/Entrenamiento'+tr+'.jpg')
  img = chile + huevo + platano   

  img0 = img[:,:,0]
  mask = np.zeros(img0.shape,dtype='uint8')

  for i in range(len(img0)):
    for j in range(len(img0)):
      mask[i,j] = 255 if img0[i,j] == 0 else 0

  fondo = cv2.bitwise_and(comida,comida,mask=mask)
  plt.imsave('./mascaras_v2/Entrenamiento'+tr+'/fondo.png',fondo)
  
# Obtenemos los datos de entrenamiento de cada clase
fondo   = getTrainData('fondo')
chiles  = getTrainData('chile')
platano = getTrainData('platano')
huevo   = getTrainData('huevo') 

# Sumamos el total de los vectores que tenemos para calcular la probabilidad a priori
total = fondo.shape[0] + chiles.shape[0] + platano.shape[0] + huevo.shape[0]

# Obtenemos las distintas estadisticas
huevo_stats = getStatistics(huevo,total)
fondo_stats = getStatistics(fondo,total)
chiles_stats = getStatistics(chiles,total)
platano_stats = getStatistics(platano,total)

# Desplegamos las estadisticas
print("#"*10,"HUEVO","#"*10)
showStatistics(huevo_stats)
print("#"*10,"FONDO","#"*10)
showStatistics(fondo_stats)
print("#"*10,"CHILE","#"*10)
showStatistics(chiles_stats)
print("#"*10,"PLATANO","#"*10)
showStatistics(platano_stats)

# Creamos nuestro clasificador de bayes
bayes = ClasificadorBayes(huevo_stats,chiles_stats,fondo_stats,platano_stats)

#realizamos las pruebas
prueba = evaluarImagen('./comida/Prueba1.jpg',bayes)
plt.imshow(prueba)

prueba = evaluarImagen('./comida/Prueba2.jpg',bayes)
plt.imshow(prueba)

prueba = evaluarImagen('./comida/Prueba3.jpg',bayes)
plt.imshow(prueba)

# Realizamos el classificador de Bayes usando la biblioteca SciKit Learn

# Juntamos todos los datos de prueba en un solo arreglo
train_data = np.append(chiles,platano,0)
train_data = np.append(train_data,huevo,0)
train_data = np.append(train_data,fondo,0)

train_labels = np.zeros(train_data.shape[0])
labels = ['chile','platano','huevo','fondo']

# 0 chile
# 1 platano
# 2 huevo
# 3 fondo

# Creamos un arreglo de etiquetas para cada valor de los datos de prueba
train_labels[chiles.shape[0]:chiles.shape[0]+platano.shape[0]] = 1
train_labels[chiles.shape[0]+platano.shape[0] :chiles.shape[0]+platano.shape[0]+huevo.shape[0]] = 2
train_labels[chiles.shape[0]+platano.shape[0]+huevo.shape[0]:-1] = 3

# Creamos el modelo y lo entrenamos
gnb = GaussianNB()
gnb.fit(train_data,train_labels)

def prueba2(numPurbea: str):
  img = plt.imread('comida/Prueba'+numPurbea+'.jpg')
  img = img/255

  res = np.zeros(shape=img.shape)
  color = [[1.0,0.0,0.0],[0.0,1.0,0.0],[1.0,1.0,1.0],[0.0,0.0,0.0]]
  
  for x in range(img.shape[0]):
    for y in range(img.shape[1]):
      res[x,y] = color[int(gnb.predict([[img[x,y,0],img[x,y,1],img[x,y,2]]])[0])]
      
  return res

for num in range(1,4):
  prueba2(str(num))