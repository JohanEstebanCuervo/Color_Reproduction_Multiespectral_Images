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
espectro = fun.Read_espectros_Imag(lista_patron)
color_RGB_pixel_ideal = fun.Ideal_Color_patch_pixel(color_check, mascaras)

im_RGB= fun.ReproduccionCie1931(imagenes_patron)

fun.imshow('Imagen reproducción CIE 1931',im_RGB)

fun.imwrite('Resultados/Imagenes/Imagen reproduccion CIE 1931.png',im_RGB)


#%% CMM TRANSFORM linear 


im_RGB3,Ccm_lineal = fun.CCM_Linear(im_RGB, color_RGB_pixel_ideal, mascaras)

fun.imshow('Imagen mejorada mediante ccm lineal', im_RGB3)

#%% CMM TRANSFORM Compund 

im_RGB4,Ccm_Compound = fun.CCM_Compound(im_RGB, color_RGB_pixel_ideal, mascaras)

fun.imshow('Imagen mejorada mediante ccm compund', im_RGB4)

#%% CMM TRANSFORM logarithm

im_RGB5,Ccm_Logarithm = fun.CCM_Logarithm(im_RGB, color_RGB_pixel_ideal, mascaras)

fun.imshow('Imagen mejorada mediante ccm logarithm', im_RGB5)


#%% CMM TRANSFORM Polynomial

im_RGB7,Ccm_Polynomial, r2 = fun.CCM_Polynomial(im_RGB, color_RGB_pixel_ideal, mascaras)
fun.imshow('Imagen mejorada mediante ccm Polynomial', im_RGB7)
fun.imwrite('Resultados/Imagenes/Imagen reproduccion Polynomial.png',im_RGB7)

media_parches = fun.RGB_IN_mean(im_RGB7, mascaras)*255
#%% Red neuronal
red = keras.models.load_model('Resultados/Variables/Correction_color_neuronal_red.h5')
rgb= np.reshape(im_RGB,(-1,3))*255
predic = red.predict(rgb)/255
#predic = (predic*255).astype(int)
im_RGB8 = np.reshape(predic,np.append(shape_imag,3))
fun.imshow('Imagen mejorada mediante Red neuronal', im_RGB8)
fun.imwrite('Resultados/Imagenes/Imagen reproduccion Red neuronal.png',im_RGB8)

#%% Errores

imagenes= [im_RGB, im_RGB3, im_RGB4 , im_RGB5, im_RGB7, im_RGB8]
errores = fun.Error_de_reproduccion(imagenes, mascaras, color_check)
errores_media = np.mean(errores,axis=1)

#%% Comparacion de parches 
fun.comparacion_color_check('CIE 1931', im_RGB, color_check, mascaras,carpeta='Resultados/Imagenes')
fun.comparacion_color_check('Polynomial', im_RGB7, color_check, mascaras,carpeta='Resultados/Imagenes')