# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 01:18:34 2021

@author: Johan Cuervo
"""

import numpy as np
from os import system
import os
import funciones_reproduccion_color as fun
import keras

#%% borrar todo lo cargado anteriormente
system("cls")

numero_imagenes=12
archivo='D:\Documentos\Articulo_Programas_Reproduccion_Color\Resultados\Datos_entrenamiento/Datos_train_Nim'+str(numero_imagenes)+'.csv'

combinaciones= fun.Read_Variable('Resultados\Variables/'+'combinaciones_mean'+'.pickle')
#%% barra de colores para mostrar grafico
color_check = np.array([[116,81,67], [199,147,129], [91,122,156], [90,108,64], [130,128,176], [92,190,172],
              [224,124,47], [68,91,170], [198,82,97], [94,58,106], [159,189,63],  [230,162,39],
              [34,63,147], [67,149,74], [180,49,57], [238,198,32], [193,84,151], [12,136,170],
              [243,238,243], [200,202,202], [161,162,161], [120,121,120], [82,83,83], [49,48,51]])



#%% busqueda de los archivos en las carpetas correspondientes

carpeta1 = 'informacion/patron'
carpeta1 = 'D:\Documentos\Articulo_Programas_Reproduccion_Color\Informacion\patron'
carpeta2 = 'informacion/mascaras'
lista1 = os.listdir(carpeta1)
lista2 = os.listdir(carpeta2)


#%% mascaras 

mascaras=fun.ext_mascaras(carpeta2, lista2)
  
#%% Organizacion de las imagenes, promedios de parches y espectro
grupo=1
lista_patron=lista1[15*(grupo-1):15*grupo]

imagenes_patron,shape_imag = fun.Read_Multiespectral_imag(carpeta1, lista_patron)
pesos_ecu = fun.Pesos_ecualizacion(imagenes_patron[:-3], mascaras[18])
imagenes_patron=(imagenes_patron[:-3].T*pesos_ecu).T/255
espectro = fun.Read_espectros_Imag(lista_patron)
color_RGB_pixel_ideal = fun.Ideal_Color_patch_pixel(color_check, mascaras)

im_RGB= fun.ReproduccionCie1931(imagenes_patron,selec_imagenes=combinaciones[numero_imagenes-1])

fun.imshow('Reproducción CIE 1931',im_RGB)

#%% CMM TRANSFORM linear 

Ccm_linear = fun.CCM_Linear_Train(archivo)
fun.Write_Variable('CCM_Linear_Nim_'+str(numero_imagenes), Ccm_linear)


im_RGB2 = fun.CCM_Linear_Test(im_RGB, Ccm_linear)

fun.imshow('Imagen mejorada mediante con lineal', im_RGB2)

#%% CMM TRANSFORM Compund 

Ccm_Compound = fun.CCM_Compound_Train(archivo)
fun.Write_Variable('CCM_Compound_Nim_'+str(numero_imagenes), Ccm_Compound)

im_RGB3 = fun.CCM_Compound_Test(im_RGB, Ccm_Compound)

fun.imshow('Imagen mejorada mediante ccm compund', im_RGB3)

#%% CMM TRANSFORM logarithm

Ccm_Logarithm = fun.CCM_Logarithm_Train(archivo)
fun.Write_Variable('CCM_Logarithm_Nim_'+str(numero_imagenes), Ccm_Logarithm)

im_RGB4 = fun.CCM_Logarithm_Test(im_RGB, Ccm_Logarithm)
fun.imshow('Imagen mejorada mediante ccm logarithm', im_RGB4)

#%% CMM TRANSFORM Polynomial

Ccm_Polynomial = fun.CCM_Polynomial_Train(archivo)
fun.Write_Variable('CCM_Polynomial_Nim_'+str(numero_imagenes), Ccm_Polynomial)
im_RGB5 = fun.CCM_Polynomial_Test(im_RGB, Ccm_Polynomial)

fun.imshow('Imagen mejorada mediante ccm Polynomial', im_RGB5)


#%% Red neuronal2
red = keras.models.load_model('Resultados/Variables/Correction_color_neuronal_red_Nim'+str(numero_imagenes)+'.h5')
rgb= np.reshape(im_RGB,(-1,3))
predic = red.predict(rgb*0.8+0.1)
#predic = (predic*255).astype(int)
im_RGB6 = np.reshape(predic,(480,640,3))/255
fun.imshow('Imagen mejorada mediante Red neuronal', im_RGB6)


#%% Errores

imagenes= [im_RGB, im_RGB2, im_RGB3, im_RGB4 , im_RGB5, im_RGB6]
errores = fun.Error_de_reproduccion(imagenes, mascaras, color_check)
errores_media = np.mean(errores,axis=1)
fun.Write_Variable('errores_de_ReyCorrec_Nim_'+str(numero_imagenes), errores)


