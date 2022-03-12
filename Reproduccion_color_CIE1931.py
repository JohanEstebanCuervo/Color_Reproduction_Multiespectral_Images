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
#system("cls")
archivo='D:\Documentos\Articulo_Programas_Reproduccion_Color\Resultados\Datos_entrenamiento/Datos_entrenamiento.csv'

numero_imagenes=5

red = keras.models.load_model('Resultados/Variables/Correction_color_neuronal_red_Nim'+str(numero_imagenes)+'.h5')
combinaciones= fun.Read_Variable('Resultados\Variables/'+'combinaciones_mean'+'.pickle')
combinaciones[4]= [1, 4,6,10,11] #se cambia para este codigo al no contar con un led de 410 nm

#%% barra de colores para mostrar grafico
color_check = np.array([[116,81,67], [199,147,129], [91,122,156], [90,108,64], [130,128,176], [92,190,172],
              [224,124,47], [68,91,170], [198,82,97], [94,58,106], [159,189,63],  [230,162,39],
              [34,63,147], [67,149,74], [180,49,57], [238,198,32], [193,84,151], [12,136,170],
              [243,238,243], [200,202,202], [161,162,161], [120,121,120], [82,83,83], [49,48,51]])



#%% busqueda de los archivos en las carpetas correspondientes

carpeta1 = 'Fotos_nuevas/patron'
carpeta2 = 'Fotos_nuevas/mascaras'
lista1 = os.listdir(carpeta1)
lista2 = os.listdir(carpeta2)


#%% mascaras 

mascaras=fun.ext_mascaras(carpeta2, lista2)
  
#%% Organizacion de las imagenes, promedios de parches y espectro
#%% Organizacion de las imagenes, promedios de parches y espectro
grupo=1
lista_patron=lista1[15*(grupo-1):15*grupo]

imagenes_patron,shape_imag = fun.Read_Multiespectral_imag(carpeta1, lista_patron)
pesos_ecu = fun.Pesos_ecualizacion(imagenes_patron[:-3], mascaras[18])
imagenes_patron=(imagenes_patron[:-3].T*pesos_ecu).T/255
#espectro = fun.Read_espectros_Imag(lista_patron)
color_RGB_pixel_ideal = fun.Ideal_Color_Patch_pixel(color_check, mascaras)

im_RGB= fun.ReproduccionCie1931(imagenes_patron,selec_imagenes=combinaciones[numero_imagenes-1])

fun.imshow('Reproducción CIE 1931',im_RGB)


#%% CMM TRANSFORM linear 


im_RGB3,Ccm_lineal = fun.CCM_Linear(im_RGB, color_RGB_pixel_ideal/255, mascaras)
im_RGB3 = fun.recorte(im_RGB3)
fun.imshow('Imagen mejorada mediante ccm lineal', im_RGB3)

#%% CMM TRANSFORM Compund 

im_RGB4,Ccm_Compound = fun.CCM_Compound(im_RGB, color_RGB_pixel_ideal, mascaras)

fun.imshow('Imagen mejorada mediante ccm compund', im_RGB4)

#%% CMM TRANSFORM logarithm

im_RGB5,Ccm_Logarithm = fun.CCM_Logarithm(im_RGB, color_RGB_pixel_ideal, mascaras)

fun.imshow('Imagen mejorada mediante ccm logarithm', im_RGB5)


#%% CMM TRANSFORM Polynomial

im_RGB7,Ccm_Polynomial = fun.CCM_Polynomial(im_RGB, color_RGB_pixel_ideal, mascaras)
fun.imshow('Imagen mejorada mediante ccm Polynomial', im_RGB7)


#%% Red neuronal
rgb= np.reshape(im_RGB,(-1,3))
predic = red.predict(rgb)/255
#predic = (predic*255).astype(int)
im_RGB8 = fun.recorte(np.reshape(predic,np.append(shape_imag,3)))
fun.imshow('Imagen mejorada mediante Red neuronal', im_RGB8)


#%% Errores

imagenes= [im_RGB*255, im_RGB3*255, im_RGB4*255 , im_RGB5*255, im_RGB7*255, im_RGB8*255]
errores = fun.Error_de_reproduccion(imagenes, mascaras, color_check)
errores_media = np.mean(errores,axis=1)


im_Lab = fun.sRGB2Lab(im_RGB)
im_Lab_Linear = fun.sRGB2Lab(im_RGB3)
im_Lab_Polynomial = fun.sRGB2Lab(im_RGB7)

color_check_lab=fun.sRGB2Lab(color_check)
imagenes= [im_Lab, im_Lab_Linear,im_Lab_Polynomial]#, im_RGB4 , im_RGB5, im_RGB7, im_RGB8]
errores_Lab = fun.Error_de_reproduccion(imagenes, mascaras, color_check_lab)
errores_media_Lab = np.mean(errores_Lab,axis=1)
