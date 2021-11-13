# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 10:16:58 2021

@author: Johan Cuervo
"""


import numpy as np
from os import system
import os
import funciones_reproduccion_color as fun
import cv2
import random
import pandas as pd

#%% borrar todo lo cargado anteriormente
system("cls")


#%% barra de colores para mostrar grafico
color_check = np.array([[116,81,67], [199,147,129], [91,122,156], [90,108,64], [130,128,176], [92,190,172],
              [224,124,47], [68,91,170], [198,82,97], [94,58,106], [159,189,63],  [230,162,39],
              [34,63,147], [67,149,74], [180,49,57], [238,198,32], [193,84,151], [12,136,170],
              [243,238,243], [200,202,202], [161,162,161], [120,121,120], [82,83,83], [49,48,51]])


D65=  np.array([
      [410,	91.486000],
      [450,	117.008000],
      [470,	114.861000],
      [490,	108.811000],
      [505,	108.578000],
      [530,	107.689000],
      [560,	100.000000],
      [590,	88.685600],
      [600,	90.006200],
      [620,	87.698700],
      [630, 83.288600],
      [650,	80.026800],
      [720,	61.604000],
  
      ])

CIE1931 =  np.array([
     
     [410,	0.043510,	0.001210,	0.207400],
     [450,	0.336200,	0.038000,	1.772110],
     [470,	0.195360,	0.090980,	1.287640],
     [490,	0.032010,	0.208020,	0.465180],
     [505,	0.002400,	0.407300,	0.212300],
     [530,	0.165500,	0.862000,	0.042160],
     [560,	0.594500,	0.995000,	0.003900],
     [590,	1.026300,	0.757000,	0.001100],
     [600,	1.062200,	0.631000,	0.000800],
     [620,	0.854450,	0.381000,	0.000190],
     [630,	0.642400,	0.265000,	0.000050],
     [650,	0.283500,	0.107000,	0.000000],
     [720,	0.002899,	0.001047,	0.000000],
     ])

XYZ2RGB= np.array([[3.2406, -1.5372, -0.4986],
         [-0.9689, 1.8758, 0.0415],
         [0.0557, -0.2040, 1.0570],])

#%% Coeficientes

Coef= (CIE1931[:,1:]*(np.ones((3,1))*D65[:,1].T).T).T

#%% busqueda de los archivos en las carpetas correspondientes

#carpeta1 = 'imgs\Patron1'
#carpeta2 = 'imgs\mascaras'
carpeta1 = 'Informacion/patron'
carpeta2 = 'Informacion/mascaras'
lista1 = os.listdir(carpeta1)
lista2 = os.listdir(carpeta2)

#%% mascaras 

mascaras=fun.ext_mascaras(carpeta2, lista2)
  
grupos_imagenes = set(range(1,21))

entrenamiento = random.sample(grupos_imagenes,14)
test = list(set(grupos_imagenes).difference(set(entrenamiento)))

for grupo in entrenamiento:
    #%% Organizacion de las imagenes, promedios de parches y espectro
    
    lista_patron=lista1[15*(grupo-1):15*grupo]
    imagenes_patron, colorn, _ , _, _ = fun.Imagenes_Camara(carpeta1 ,lista_patron, mascaras, color_check)
    colorn=np.array(colorn,dtype='uint8')
    #%%  Reproduccion de color usando CIE

    imagenes_patron_ecu = imagenes_patron[:,:-2].T
    
    xyz = np.dot(Coef,imagenes_patron_ecu).T/13
    maximos=np.max(xyz,axis=0)
    xyz = np.divide(xyz,np.max(xyz,axis=0))
    
    im_Y=np.reshape(xyz[:,1],(480,640))
    
    k= 1/np.mean(im_Y[np.where(mascaras[18]==255)])
    xyz*=k
    
    rgb = fun.recorte(np.dot(XYZ2RGB,xyz.T).T)
    
    im_RGB=np.reshape(rgb,(480,640,3))
    fun.imshow('Imagen reconstruida grupo: '+str(grupo), im_RGB)
    lista_RGB = fun.RGB_IN(im_RGB, mascaras)*255
    lista_RGB = np.array(lista_RGB,dtype='uint8')
    if grupo == entrenamiento[0]:
        Datos_entrenamiento=np.concatenate((lista_RGB,colorn),axis=1)
    else:
        dato_im = np.concatenate((lista_RGB,colorn),axis=1)
        Datos_entrenamiento=np.concatenate((Datos_entrenamiento,dato_im))
        

df = pd.DataFrame(Datos_entrenamiento)
df.to_csv('Datos_entrenamiento.csv',header=None,index=False)

test=list(test)

for grupo in test:
    #%% Organizacion de las imagenes, promedios de parches y espectro
    
    lista_patron=lista1[15*(grupo-1):15*grupo]
    imagenes_patron, colorn, _ , _, _ = fun.Imagenes_Camara(carpeta1 ,lista_patron, mascaras, color_check)
    colorn=np.array(colorn,dtype='uint8')
    #%%  Reproduccion de color usando CIE

    imagenes_patron_ecu = imagenes_patron[:,:-2].T
    
    xyz = np.dot(Coef,imagenes_patron_ecu).T/13
    maximos=np.max(xyz,axis=0)
    xyz = np.divide(xyz,np.max(xyz,axis=0))
    
    im_Y=np.reshape(xyz[:,1],(480,640))
    
    k= 1/np.mean(im_Y[np.where(mascaras[18]==255)])
    xyz*=k
    
    rgb = fun.recorte(np.dot(XYZ2RGB,xyz.T).T)
    
    im_RGB=np.reshape(rgb,(480,640,3))
    fun.imshow('Imagen reconstruida grupo: '+str(grupo), im_RGB)
    lista_RGB = fun.RGB_IN(im_RGB, mascaras)*255
    lista_RGB = np.array(lista_RGB,dtype='uint8')
    if grupo == test[0]:
        Datos_test=np.concatenate((lista_RGB,colorn),axis=1)
    else:
        dato_im = np.concatenate((lista_RGB,colorn),axis=1)
        Datos_test=np.concatenate((Datos_test,dato_im))     


df = pd.DataFrame(Datos_test)
df.to_csv('Datos_test.csv', header=None,index=False)