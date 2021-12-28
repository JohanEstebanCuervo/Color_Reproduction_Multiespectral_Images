# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 18:01:59 2021

@author: Johan Cuervo
"""


import numpy as np
from os import system
import os
import funciones_reproduccion_color as fun
import matplotlib.pyplot as plt

#%% borrar todo lo cargado anteriormente
system("cls")
archivo='D:\Documentos\Articulo_Programas_Reproduccion_Color\Resultados\Datos_entrenamiento/Datos_train_Nim7.csv'


type_errors=['mean','max','variance','mean_for_standard','rango']
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
imagenes=[]

for type_error in type_errors:
    combinaciones = fun.Read_Variable('Resultados\Variables/'+'combinaciones_'+type_error+'.pickle')
    
    im_RGB= fun.ReproduccionCie1931(imagenes_patron,selec_imagenes=combinaciones[6])
    #fun.imshow('Reproducción CIE 1931',im_RGB)
    imagenes+=[im_RGB]
    
    

errores=fun.Error_de_reproduccion(imagenes, mascaras, color_check)
media= np.mean(errores,axis=1)



plt.figure(figsize=(8,6))
for i in range(len(type_errors)):
    plt.plot(range(1,25),errores[i])

plt.legend(type_errors)
plt.xlabel('numero de parche')
#plt.savefig('Resultados/Imagenes/Grafica_error_'+nombre+'.pdf', format='pdf')
plt.show()
    
    
    
    
    
    
    