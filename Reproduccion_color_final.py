# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 22:24:32 2021

@author: Johan Cuervo
"""


import numpy as np
import os
import funciones_reproduccion_color as fun
import keras

#%% borrar todo lo cargado anteriormente
#system("cls")


numero_imagenes=4

red = keras.models.load_model('Resultados/Variables/Correction_color_neuronal_red_Nim'+str(numero_imagenes)+'.h5')
combinaciones= fun.Read_Variable('Resultados\Variables/'+'combinaciones_mean'+'.pickle')
combinaciones[4]= [1, 4,6,10,11] #se cambia para este codigo al no contar con un led de 410 nm
combinaciones[10]= [1,2,3, 4,5,6,7,8,9,10,11] #se cambia para este codigo al no contar con un led de 410 nm
#%% barra de colores para mostrar grafico
color_check = np.array([[116,81,67], [199,147,129], [91,122,156], [90,108,64], [130,128,176], [92,190,172],
              [224,124,47], [68,91,170], [198,82,97], [94,58,106], [159,189,63],  [230,162,39],
              [34,63,147], [67,149,74], [180,49,57], [238,198,32], [193,84,151], [12,136,170],
              [243,238,243], [200,202,202], [161,162,161], [120,121,120], [82,83,83], [49,48,51]])



#%% busqueda de los archivos en las carpetas correspondientes

carpeta1 = 'informacion/patron'
carpeta2 = 'informacion/mascaras'
lista1 = os.listdir(carpeta1)
lista2 = os.listdir(carpeta2)

imwrite='on'

#%% mascaras 

mascaras=fun.ext_mascaras(carpeta2, lista2)
  
#%% Organizacion de las imagenes, promedios de parches y espectro
grupo=1
lista_patron=lista1[15*(grupo-1):15*grupo]

imagenes_patron,shape_imag = fun.Read_Multiespectral_imag(carpeta1, lista_patron)
pesos_ecu = fun.Pesos_ecualizacion(imagenes_patron[:-3], mascaras[18])
imagenes_patron=(imagenes_patron[:-3].T*pesos_ecu).T/255
color_RGB_pixel_ideal = fun.Ideal_Color_Patch_pixel(color_check, mascaras)

im_RGB= fun.ReproduccionCie1931(imagenes_patron,selec_imagenes=combinaciones[numero_imagenes-1])

fun.imshow('Reproducción CIE 1931',im_RGB)

if imwrite == 'on':
    fun.imwrite('Resultados/Imagenes/reproduccion_CIE_1931_Nim_'+str(numero_imagenes)+'.png',im_RGB)

#%% CMM TRANSFORM linear 

Ccm_linear = fun.Read_Variable('Resultados\Variables/'+'CCM_Linear_Nim_'+str(numero_imagenes)+'.pickle')

im_RGB_Linear = fun.CCM_Linear_Test(im_RGB, Ccm_linear)
fun.imshow('Imagen mejorada mediante con linear', im_RGB_Linear)
if imwrite == 'on':
    fun.imwrite('Resultados/Imagenes/Correcion_Lineal_Nim_'+str(numero_imagenes)+'.png',im_RGB_Linear)

#%% CMM TRANSFORM Compund 

Ccm_Compound = fun.Read_Variable('Resultados\Variables/'+'CCM_Compound_Nim_'+str(numero_imagenes)+'.pickle')

im_RGB_Compound = fun.CCM_Compound_Test(im_RGB, Ccm_Compound)
fun.imshow('Imagen mejorada mediante ccm compund', im_RGB_Compound)
if imwrite == 'on':
    fun.imwrite('Resultados/Imagenes/Correcion_Compound_Nim_'+str(numero_imagenes)+'.png',im_RGB_Compound)

#%% CMM TRANSFORM logarithm

Ccm_Logarithm = fun.Read_Variable('Resultados\Variables/'+'CCM_Logarithm_Nim_'+str(numero_imagenes)+'.pickle')

im_RGB_Logarithm = fun.CCM_Logarithm_Test(im_RGB, Ccm_Logarithm)
fun.imshow('Imagen mejorada mediante ccm logarithm', im_RGB_Logarithm)
if imwrite == 'on':
    fun.imwrite('Resultados/Imagenes/Correcion_Logarithm_Nim_'+str(numero_imagenes)+'.png',im_RGB_Logarithm)

#%% CMM TRANSFORM Polynomial

Ccm_Polynomial = fun.Read_Variable('Resultados\Variables/'+'CCM_Polynomial_Nim_'+str(numero_imagenes)+'.pickle')

im_RGB_Polynomial = fun.CCM_Polynomial_Test(im_RGB, Ccm_Polynomial)
fun.imshow('Imagen mejorada mediante ccm Polynomial', im_RGB_Polynomial)
if imwrite == 'on':
    fun.imwrite('Resultados/Imagenes/Correcion_Polynomial_Nim_'+str(numero_imagenes)+'.png',im_RGB_Polynomial)

#%% Red neuronal

rgb= np.reshape(im_RGB,(-1,3))
predic = red.predict(rgb*0.8+0.1)

im_RGB_NN = fun.recorte(np.reshape(predic,(480,640,3))/255)
fun.imshow('Imagen mejorada mediante Red neuronal', im_RGB_NN)

if imwrite == 'on':
    fun.imwrite('Resultados/Imagenes/Correcion_RedNeuronal_Nim_'+str(numero_imagenes)+'.png',im_RGB_NN)
    
#%% Errores

imagenes= [fun.sRGB2Lab(im_RGB),fun.sRGB2Lab( im_RGB_Linear),fun.sRGB2Lab( im_RGB_Compound),fun.sRGB2Lab( im_RGB_Logarithm), fun.sRGB2Lab(im_RGB_Polynomial),fun.sRGB2Lab( im_RGB_NN)]
errores = fun.Error_de_reproduccion(imagenes, mascaras,fun.sRGB2Lab( color_check))
errores_media = np.mean(errores,axis=1)


#%% Comparacion de parches 
if imwrite == 'on':
    fun.comparacion_color_check('CIE_1931_Nim_'+str(numero_imagenes)+' espectros', im_RGB, color_check, mascaras,carpeta='Resultados/Imagenes')
    fun.comparacion_color_check('Linear_Nim_'+str(numero_imagenes)+' espectros', im_RGB_Linear, color_check, mascaras,carpeta='Resultados/Imagenes')
    fun.comparacion_color_check('Compound_Nim_'+str(numero_imagenes)+' espectros', im_RGB_Compound, color_check, mascaras,carpeta='Resultados/Imagenes')
    fun.comparacion_color_check('Logarithm_Nim_'+str(numero_imagenes)+' espectros', im_RGB_Logarithm, color_check, mascaras,carpeta='Resultados/Imagenes')
    fun.comparacion_color_check('Polynomial_Nim_'+str(numero_imagenes)+' espectros', im_RGB_Polynomial, color_check, mascaras,carpeta='Resultados/Imagenes')
    fun.comparacion_color_check('RedNeuronal_Nim_'+str(numero_imagenes)+' espectros', im_RGB_NN, color_check, mascaras,carpeta='Resultados/Imagenes')