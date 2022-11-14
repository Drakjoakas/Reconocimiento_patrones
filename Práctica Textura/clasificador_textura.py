import numpy as np
import matplotlib.pyplot as plt
import cv2
from operator import itemgetter
from scipy.stats import skew,kurtosis,entropy
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

MEAN = 'mean'
VARIANCE = 'variance'
SKEWNESS = 'skew'
KURTOSIS = 'kurt'
ENTROPY  = 'entropy'
ENERGY   = 'energy'



def sliding_window(image: np.ndarray, stepSize:int, windowSize:tuple):
  """Función que genera las ventana de una imagen de acuerdo a un tamaño de ventana y un paso de tamaño variable.

  Args:
      image (np.ndarray): imagen de donde se obtiene la ventana
      stepSize (int): tamaño del paso que se mueve la ventana
      windowSize (tuple): tamaño de la ventana
  """
  for y in range(0, image.shape[0], stepSize):
    for x in range(0, image.shape[1], stepSize):
      yield(x,y,image[y:y+windowSize[1],x:x+windowSize[0]])
      
      
def getEnergy(image: np.ndarray) -> int:
  """calcula la energía de los pixeles de una imagen

  Args:
      image (np.ndarray): imagen de la cual se obtendrá la energía

  Returns:
      int: energía del arreglo
  """
  energy = 0
  for x in range(image.shape[0]):
    for y in range(image.shape[1]):
      energy += (image[x,y])**2
  return energy


def getFirstOrderStatistics(image: np.ndarray,names=False,stats_list: list = []):
  """obtiene los estadísticos de primer orden de una imagen

  Args:
      image (np.ndarray): imagen de la que se calculan los estadísticps
      names (bool, optional): booleano que indica si se regresan los estadísticos con nombre o no. Defaults to False.
      stats_list (list, optional): lista de estadísticos que se quieran obtener. Defaults to [].

  Returns:
      list: vector de características con estadísticos de primer orden
  """
  stats = {
    'mean': image.mean(),
    'variance': image.var(),
    'skew': skew(image,axis=(0,1)),
    'kurt': kurtosis(image,axis=(0,1)),
    'entropy': entropy(image,axis=(0,1)),
    'energy': getEnergy(image)
  }

  if len(stats_list) > 0:
    return list(itemgetter(*stats_list)(stats))
  return stats if names else list(stats.values())

def getTrainingData(texel: np.ndarray, step=10,windowSz=(10,10),stats_list=[]):
  """obtiene los datos de entrenamiento a partir de un texel y los vectores de características de cada ventana

  Args:
      texel (np.ndarray): imagen de la cual se obtienen las características
      step (int, optional): tamaño del paso de la ventana. Defaults to 10.
      windowSz (tuple, optional): tamaño de la ventana. Defaults to (10,10).
      stats_list (list, optional): lista de estadísticas que se deseen calcular. Defaults to [].

  Returns:
      np.array: Arreglo con vectores de características del texel
  """
  data = []
  for (x,y,window) in sliding_window(texel,stepSize=step,windowSize=windowSz):
    data.append(getFirstOrderStatistics(window,stats_list=stats_list))
  return np.array(data)

def getAllTrainingData(tree_data,mono_data,hoja_data):
  """obtiene los datos de entrenamiento concatenados y sus etiquetas a partir de los datos de entrenamiento de cada texel

  Args:
      tree_data (np.ndarray): datos de entrenamiento del texel de arbol
      mono_data (np.ndarray): datos de entrenamiento del texel del mono
      hoja_data (np.ndarray): datos de entrenamiento del texel de las hojas

  Returns:
      np.ndarray,np.ndarray: ndarray con los datos de entrenamiento y ndarray con las etiquetas de los datos de entrenamiento
  """
  train_data = np.append(tree_data,mono_data,0)
  train_data = np.append(train_data,hoja_data,0)

  # 0 arbol
  # 1 mono
  # 2 hojas
  train_labels = np.zeros(train_data.shape[0])
  train_labels[tree_data.shape[0]:tree_data.shape[0]+mono_data.shape[0]] = 1
  train_labels[tree_data.shape[0]+mono_data.shape[0] :tree_data.shape[0]+mono_data.shape[0]+hoja_data.shape[0]] = 2
  
  return (train_data,train_labels)


def classificadorBayesiano(imagen:np.ndarray,gnb: GaussianNB,windowSize=(10,10),windowStep=10,stats_list=[]):
  """clasifica y colorea una imagen de acuerdo a la clase predicha con un calsificador bayesiano "ingenuo"

  Args:
      imagen (np.ndarray): imagen para clasificar
      gnb (GaussianNB): clasificador gaussiano de bayes "ingenuo"
      windowSize (tuple, optional): tamaño de la ventana del recorrido de la imagen. Defaults to (10,10).
      windowStep (int, optional): tamaño del paso del recorrido de la imagen. Defaults to 10.
      stats_list (list, optional): lista de estadísticas a calcular. Defaults to [].

  Returns:
      np.ndarray: imagen con la clasificación coloreada
  """
  color = [[79, 131, 176],[133, 65, 181],[217, 210, 17]]
  cont_err_win = 0
  cont_err_nan = 0
  res = np.zeros(shape=imagen.shape)
  res = np.dstack([res,res,res])
  
  for(x,y,window) in sliding_window(imagen,windowStep,windowSize):
    
    if window.shape[0] != windowSize[0] or window.shape[1] != windowSize[1]:
      cont_err_win += 1
      continue
    
    
    vector = getFirstOrderStatistics(window,stats_list=stats_list)

    if np.isnan(vector).any():
      cont_err_nan += 1
      continue
    
    color_prediction = color[int(gnb.predict([vector]))]
    for iy in range(y,y+10):
      for ix in range(x,x+10):
        if (ix < res.shape[1] and iy < res.shape[0]):
          res[iy,ix] = color_prediction
    
  return res


def kmeansClassifier(imagen:np.ndarray,kmeans:KMeans,windowSize=(10,10),windowStep=10,stats_list=[]):
  """calsifica y colorea una imagen de acuerdo al aclasificación obtenida por el clasificador kmeans

  Args:
      imagen (np.ndarray): imagen a clasificar
      kmeans (KMeans): clasificador de kmeans
      windowSize (tuple, optional): tamaño de la ventana a analizar. Defaults to (10,10).
      windowStep (int, optional): tamaño del paso de desplazamiento de la ventana. Defaults to 10.
      stats_list (list, optional): lista de estadísticos a clacular. Defaults to [].

  Returns:
      np.ndarray: imagen coloreada con la clasificación
  """
  color = [[79, 131, 176],[133, 65, 181],[217, 210, 17]]
  cont_err_win = 0
  cont_err_nan = 0
  res = np.zeros(shape=imagen.shape)
  res = np.dstack([res,res,res])
  
  for(x,y,window) in sliding_window(imagen,windowStep,windowSize):
    
    if window.shape[0] != windowSize[0] or window.shape[1] != windowSize[1]:
      cont_err_win += 1
      continue
    
    
    vector = getFirstOrderStatistics(window,stats_list=stats_list)

    if np.isnan(vector).any():
      cont_err_nan += 1
      continue
    
    color_prediction = color[int(kmeans.predict([vector]))]
    for iy in range(y,y+10):
      for ix in range(x,x+10):
        if (ix < res.shape[1] and iy < res.shape[0]):
          res[iy,ix] = color_prediction
    
  return res

def neighborsClassifier(imagen:np.ndarray,neigh:KNeighborsClassifier,windowSize=(10,10),windowStep=10,stats_list=[]):
  """clasifica y colorea de acuerdo a la clasificación dada por un clasificador KNN

  Args:
      imagen (np.ndarray): imagen a clasificar
      neigh (KNeighborsClassifier): clasificador de KNN
      windowSize (tuple, optional): tamaño de la ventana a analizar. Defaults to (10,10).
      windowStep (int, optional): tamaño del paso del desplazamiento de la ventana. Defaults to 10.
      stats_list (list, optional): lista de estadísticos a calcular. Defaults to [].

  Returns:
      np.ndarray: imagen coloreada con la clasificación
  """
  color = [[79, 131, 176],[133, 65, 181],[217, 210, 17]]
  cont_err_win = 0
  cont_err_nan = 0
  res = np.zeros(shape=imagen.shape)
  res = np.dstack([res,res,res])
  
  for(x,y,window) in sliding_window(imagen,windowStep,windowSize):
    
    if window.shape[0] != windowSize[0] or window.shape[1] != windowSize[1]:
      cont_err_win += 1
      continue
    
    
    vector = getFirstOrderStatistics(window,stats_list=stats_list)

    if np.isnan(vector).any():
      cont_err_nan += 1
      continue
    
    color_prediction = color[int(neigh.predict([vector]))]
    for iy in range(y,y+10):
      for ix in range(x,x+10):
        if (ix < res.shape[1] and iy < res.shape[0]):
          res[iy,ix] = color_prediction
    
  return res


# Obtención de los texels a partir de una imagen

img = plt.imread('./monos/n0114.jpg')

texel_mono  = cv2.cvtColor(img[260:360,480:580],cv2.COLOR_RGB2GRAY)
texel_arbol = cv2.cvtColor(img[190:270,150:230],cv2.COLOR_RGB2GRAY)
texel_hoja  = cv2.cvtColor(img[150:250,570:670],cv2.COLOR_RGB2GRAY)

fig,axes = plt.subplots(1,3,figsize=(12,4))
ax = axes.ravel()
ax[0].imshow(texel_mono, cmap=plt.cm.gray)
ax[0].set_title("Mono")
ax[1].imshow(texel_arbol, cmap=plt.cm.gray)
ax[1].set_title("Arbol")
ax[2].imshow(texel_hoja, cmap=plt.cm.gray)
ax[2].set_title("Arbol")
fig.tight_layout()
plt.show()

# Obtención de los datos de entrenamiento de cada texel
tree_data = getTrainingData(texel_arbol)
mono_data = getTrainingData(texel_mono)
hoja_data = getTrainingData(texel_hoja)

# Obtenemos los datos de entrenamiento y sus etiquetas con el formato que solicita sklearn
train_data, train_labels = getAllTrainingData(tree_data,mono_data,hoja_data)


# Obtenemos los diferentes clasificadores
# Clasificador Naive Bayes Gaussiano
gnb = GaussianNB()
gnb.fit(train_data,train_labels)

# Clasificador KNN
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train_data,train_labels)

# Clasificador kmeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(train_data)

mono1 = cv2.imread('./monos/n0114.jpg')
mono1 = cv2.cvtColor(mono1,cv2.COLOR_BGR2GRAY)

# Clasificación del mono con distintos clasificadores
mono1_class = classificadorBayesiano(mono1,gnb)
mono2_class = neighborsClassifier(mono1,neigh)
mono3_class = kmeansClassifier(mono1,kmeans)

fig,axes = plt.subplots(1,3,figsize=(24,8))
ax = axes.ravel()
ax[0].imshow(mono1_class/255)
ax[0].set_title("naive Bayes")
ax[1].imshow(mono2_class/255)
ax[1].set_title("K-NN")
ax[2].imshow(mono3_class/255)
ax[2].set_title("K-Means")
fig.tight_layout()
plt.show()

# Clasificando usando diferentes estadísticos
stats1 = [MEAN,VARIANCE,KURTOSIS,ENTROPY]

mono_data_prueba1  = getTrainingData(texel_mono,stats_list=stats1)
arbol_data_prueba1 = getTrainingData(texel_arbol,stats_list=stats1)
hojas_data_prueba1 = getTrainingData(texel_hoja,stats_list=stats1)

train_data_prueba1, train_labels_prueba1 = getAllTrainingData(arbol_data_prueba1,mono_data_prueba1,hojas_data_prueba1)

gnb_prueba1 = GaussianNB()
gnb_prueba1.fit(train_data_prueba1,train_labels_prueba1)

mono_prueba1 = cv2.imread('./monos/n0114.jpg')
mono_prueba1 = cv2.cvtColor(mono_prueba1,cv2.COLOR_BGR2GRAY)

mono_prueba1_class = classificadorBayesiano(mono_prueba1, gnb=gnb_prueba1,stats_list=stats1)

plt.imshow(mono_prueba1_class/255)

# Clasificando variando el tamaño de la ventana
winSz  = (5,5)
stepSz = 5

mono_data_prueba2  = getTrainingData(texel_mono,windowSz=winSz,step=stepSz)
arbol_data_prueba2 = getTrainingData(texel_arbol,windowSz=winSz,step=stepSz)
hojas_data_prueba2 = getTrainingData(texel_hoja,windowSz=winSz,step=stepSz) 

train_data_prueba2, train_labels_prueba2 = getAllTrainingData(arbol_data_prueba2,mono_data_prueba2,hojas_data_prueba2)

gnb_prueba2 = GaussianNB()
gnb_prueba2.fit(train_data_prueba2,train_labels_prueba2)

mono_prueba2 = cv2.imread('./monos/n0035.jpg')
mono_prueba2 = cv2.cvtColor(mono_prueba2,cv2.COLOR_BGR2GRAY)

mono_prueba2_class = classificadorBayesiano(mono_prueba2, gnb=gnb_prueba2,windowSize=winSz,windowStep=stepSz)

plt.imshow(mono_prueba2_class/255)