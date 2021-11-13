# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:04:59 2021

@author: Johan Cuervo
"""

import numpy as np
from os import system
import os
import funciones_reproduccion_color as fun
import cv2
import keras
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

#Coef= CIE1931[:,1:].T

#%% busqueda de los archivos en las carpetas correspondientes

#carpeta1 = 'imgs\Patron1'
#carpeta2 = 'imgs\mascaras'
#carpeta1 = 'informacion/patron'
carpeta1 = 'D:\Documentos\Articulo_Programas_Reproduccion_Color\Informacion\patron'
#carpeta1 = 'c1_renombradas/'
carpeta2 = 'informacion/mascaras'
lista1 = os.listdir(carpeta1)
lista2 = os.listdir(carpeta2)

#%% mascaras 

mascaras=fun.ext_mascaras(carpeta2, lista2)
  
#%% Organizacion de las imagenes, promedios de parches y espectro
grupo=5
lista_patron=lista1[15*(grupo-1):15*grupo]
imagenes_patron, colorn, prom, entrada, espectro = fun.Imagenes_Camara(carpeta1 ,lista_patron, mascaras, color_check)

#%% normalizamos el promedio con respecto al blanco es decir fila 19 se normalizan

pesos_ecu, prom_ecualizado_N = fun.ecualizacion(prom,19,243)

#%%  Reproduccion de color usando CIE

imagenes_patron_ecu = imagenes_patron#*pesos_ecu
imagenes_patron_ecu = imagenes_patron_ecu[:,:-2].T

xyz = np.dot(Coef,imagenes_patron_ecu).T/13
maximos=np.max(xyz,axis=0)
xyz = np.divide(xyz,np.max(xyz,axis=0))

im_Y=np.reshape(xyz[:,1],(480,640))

rgb = fun.recorte(np.dot(XYZ2RGB,xyz.T).T)

im_RGB=np.reshape(rgb,(480,640,3))
fun.imshow('Imagen reproducción CIE 1931',im_RGB)

fun.imwrite('Resultados/Imagenes/Imagen reproduccion CIE 1931.png',im_RGB)
#%% Matriz de transformacion

Coef2=np.divide(Coef.T,maximos).T
mte2rgb= np.dot(XYZ2RGB,Coef2)/13

rgb2=fun.recorte(np.dot(mte2rgb,imagenes_patron_ecu))
im_RGB2=np.reshape(rgb2.T,(480,640,3))

fun.imshow('Reconstrucción con matriz de transformación',im_RGB2)

#%% CMM TRANSFORM linear 

N = np.reshape(imagenes_patron[:,13],(480,640))/255

im_RGB3,Ccm_lineal = fun.CCM_Linear(im_RGB2, colorn, mascaras)

fun.imshow('Imagen mejorada mediante con lineal', im_RGB3)

#%% CMM TRANSFORM Compund 

im_RGB4,Ccm_Compound = fun.CCM_Compound(im_RGB2, colorn, mascaras)

fun.imshow('Imagen mejorada mediante ccm compund', im_RGB4)

#%% CMM TRANSFORM logarithm

im_RGB5,Ccm_Logarithm = fun.CCM_Logarithm(im_RGB2, colorn, mascaras)

fun.imshow('Imagen mejorada mediante ccm logarithm', im_RGB5)

#%% CMM TRANSFORM Polynomial near

im_RGB6,Ccm_Polynomial_N, r2 = fun.CCM_Polynomial_N(im_RGB2,N, colorn, mascaras)
fun.imshow('Imagen mejorada mediante ccm Polynomial con Near', im_RGB6)

#%% CMM TRANSFORM Polynomial

im_RGB7,Ccm_Polynomial, r2 = fun.CCM_Polynomial(im_RGB2, colorn, mascaras)
fun.imshow('Imagen mejorada mediante ccm Polynomial', im_RGB7)
fun.imwrite('Resultados/Imagenes/Imagen reproduccion Polynomial.png',im_RGB7)

media_parches = fun.RGB_IN_mean(im_RGB7, mascaras)*255
#%% Red neuronal
red = keras.models.load_model('Resultados/Variables/Correction_color_neuronal_red2.h5')
rgb= np.reshape(im_RGB,(-1,3))
predic = red.predict(rgb*0.8+0.1)
#predic = (predic*255).astype(int)
im_RGB8 = np.reshape(predic,(480,640,3))/255
fun.imshow('Imagen mejorada mediante Red neuronal', im_RGB8)
fun.imwrite('Resultados/Imagenes/Imagen reproduccion Red neuronal.png',im_RGB8)

#%% Errores

imagenes= [im_RGB, im_RGB3, im_RGB4 , im_RGB5, im_RGB6, im_RGB7, im_RGB8]
errores = fun.Error_de_reproduccion(imagenes, mascaras, color_check)
errores_media = np.mean(errores,axis=1)

#%% Comparacion de parches 
fun.comparacion_color_check('CIE 1931', im_RGB, color_check, mascaras,carpeta='Resultados/Imagenes')
fun.comparacion_color_check('Polynomial', im_RGB6, color_check, mascaras,carpeta='Resultados/Imagenes')








