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




#%% busqueda de los archivos en las carpetas correspondientes

carpeta1 = 'Informacion/patron'
carpeta2 = 'Informacion/mascaras'
lista1 = os.listdir(carpeta1)
lista2 = os.listdir(carpeta2)

#%% mascaras 
mascaras=fun.ext_mascaras(carpeta2, lista2)
color_RGB_pixel_ideal = fun.Ideal_Color_patch_pixel(color_check, mascaras).astype(int)

grupos_imagenes = set(range(1,21))

entrenamiento = random.sample(grupos_imagenes,14)
test = list(set(grupos_imagenes).difference(set(entrenamiento)))

for grupo in entrenamiento:
    #% Reconstruccion
    
    lista_patron=lista1[15*(grupo-1):15*grupo]
    imagenes_patron,shape_imag = fun.Read_Multiespectral_imag(carpeta1, lista_patron,filtro_bi='on') 
    im_RGB= fun.ReproduccionCie1931(imagenes_patron[:-2]/255)
    
    error = fun.Error_de_reproduccion([im_RGB], mascaras, color_check)
    print('error de reconstruccion imagen '+str(grupo)+':'+str(np.mean(error)))
    
    fun.imshow('Imagen reconstruida grupo: '+str(grupo), im_RGB)
    lista_RGB = fun.RGB_IN(im_RGB, mascaras)*255
    lista_RGB = np.array(lista_RGB,dtype='uint8')
    if grupo == entrenamiento[0]:
        Datos_entrenamiento=np.concatenate((lista_RGB,color_RGB_pixel_ideal),axis=1)
    else:
        dato_im = np.concatenate((lista_RGB,color_RGB_pixel_ideal),axis=1)
        Datos_entrenamiento=np.concatenate((Datos_entrenamiento,dato_im))
        

df = pd.DataFrame(Datos_entrenamiento)
df.to_csv('Resultados/Datos_entrenamiento/Datos_train_12_im.csv',header=None,index=False)

test=list(test)

for grupo in test:
     #% Reconstruccion
    
    lista_patron=lista1[15*(grupo-1):15*grupo]
    imagenes_patron,shape_imag = fun.Read_Multiespectral_imag(carpeta1, lista_patron) 
    im_RGB= fun.ReproduccionCie1931(imagenes_patron[:-2]/255)
    
    error = fun.Error_de_reproduccion([im_RGB], mascaras, color_check)
    print('error de reconstruccion imagen '+str(grupo)+':'+str(np.mean(error)))
    
    fun.imshow('Imagen reconstruida grupo: '+str(grupo), im_RGB)
    lista_RGB = fun.RGB_IN(im_RGB, mascaras)*255
    lista_RGB = np.array(lista_RGB,dtype='uint8')
    if grupo == test[0]:
        Datos_test=np.concatenate((lista_RGB,color_RGB_pixel_ideal),axis=1)
    else:
        dato_im = np.concatenate((lista_RGB,color_RGB_pixel_ideal),axis=1)
        Datos_test=np.concatenate((Datos_test,dato_im))     


df = pd.DataFrame(Datos_test)
df.to_csv('Resultados/Datos_entrenamiento/Datos_test_12_im.csv', header=None,index=False)

fun.Write_Variable('Lista_grupos_test_12_im', test)