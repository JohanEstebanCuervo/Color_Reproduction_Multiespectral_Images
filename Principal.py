# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 22:41:25 2021

@author: Johan Cuervo
"""

import numpy as np
import matplotlib.pyplot as plt
from os import system
import os
import funciones_reproduccion_color as fun

#%% borrar todo lo cargado anteriormente
system("cls")

#%% barra de colores para mostrar grafico
color_check = np.array([[116,81,67], [199,147,129], [91,122,156], [90,108,64], [130,128,176], [92,190,172],
              [224,124,47], [68,91,170], [198,82,97], [94,58,106], [159,189,63],  [230,162,39],
              [34,63,147], [67,149,74], [180,49,57], [238,198,32], [193,84,151], [12,136,170],
              [243,238,243], [200,202,202], [161,162,161], [120,121,120], [82,83,83], [49,48,51]])


#%% busqueda de los archivos en las carpetas correspondientes
carpeta1 = 'imgs\Patron'
carpeta2 = 'imgs\mascaras'
lista1 = os.listdir(carpeta1)
lista2 = os.listdir(carpeta2)


#%% mascaras 
mascaras=fun.ext_mascaras(carpeta2, lista2)
  
#%% Organizacion de las imagenes, promedios de parches y espectro
imagenes_patron, colorn, prom, entrada, espectro = fun.Imagenes_Camara(carpeta1 ,lista1, mascaras, color_check)


#%% normalizamos el promedio con respecto al blanco es decir fila 19 se normalizan
pesos_ecu, prom_ecualizado_N = fun.ecualizacion(prom,19,243)


#%% grafica de las firmas espectrales
colorN= color_check/255;
fun.grafica_firmas_espectrales(espectro, prom_ecualizado_N, colorN)


#%%  Reproduccion de color pseudo inversa Promedio
imagenes_patron_ecu = imagenes_patron*pesos_ecu
EcualizadoINV= np.linalg.pinv(prom_ecualizado_N)

C= np.dot(EcualizadoINV,color_check)

Conversion= np.dot(imagenes_patron_ecu,C)
Conversion= Conversion/np.max(Conversion)

Irecons= np.reshape(Conversion,(480,640,3)) 
Factor= (243/255)/np.mean(Irecons[np.where(mascaras[18]==255)])
Irecons*=Factor
plt.imshow(Irecons)
plt.show()


#%% Reproduccion Color con Pseudo Inversa

entrada_ecu = entrada*pesos_ecu
EcualizadoINV2= np.linalg.pinv(entrada_ecu)
C2= np.dot(EcualizadoINV2,colorn)
Conversion2 = np.dot(imagenes_patron_ecu,C2)
Conversion2 = Conversion2/np.max(Conversion2)

Irecons2 = np.reshape(Conversion2,(480,640,3)) 
Factor= (243/255)/np.mean(Irecons2[np.where(mascaras[18]==255)])
Irecons2*=Factor
plt.imshow(Irecons2)
plt.show()

# cv2.imshow('imagen recons', Irecons2)
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 

#%% error

Color_parche_Irecons= (fun.promedio_RGB_parches(Irecons2, mascaras)*255).astype(int)

