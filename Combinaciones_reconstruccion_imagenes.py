# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 02:15:02 2021

@author: Johan Cuervo
"""


import numpy as np
from os import system
import os
import funciones_reproduccion_color as fun
import matplotlib.pyplot as plt
import pickle

def graficas_error(min_error,nombre):
    plt.figure(figsize=(4,3))
    plt.plot(range(1,13),np.array(min_error),color='black')
    plt.xlabel('Cantidad Im')
    plt.savefig('Resultados/Imagenes/Grafica_error_'+nombre+'.pdf', format='pdf')
    plt.show()
    
    fichero = open('Resultados/Variables/combinaciones_'+nombre+'.pickle','wb')
    pickle.dump(mejores_comb,fichero)
    fichero.close()

#%% borrar todo lo cargado anteriormente
#system("cls")

#%% barra de colores para mostrar grafico
color_check = np.array([[116,81,67], [199,147,129], [91,122,156], [90,108,64], [130,128,176], [92,190,172],
              [224,124,47], [68,91,170], [198,82,97], [94,58,106], [159,189,63],  [230,162,39],
              [34,63,147], [67,149,74], [180,49,57], [238,198,32], [193,84,151], [12,136,170],
              [243,238,243], [200,202,202], [161,162,161], [120,121,120], [82,83,83], [49,48,51]])



#%% busqueda de los archivos en las carpetas correspondientes

carpeta1 = 'informacion/patron'
#carpeta1 = ':\Documentos\Articulo_Programas_Reproduccion_Color\Informacion\patron'
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
#%% Combinaciones

#type_errors=['mean','max','variance','mean_for_standard','rango']
type_errors=['mean']
for type_error in type_errors:
    mejores_comb=[]
    min_error = []
    for cant_imagenes in range(1,13):
        comb, error = fun.mejor_combinacion(imagenes_patron, mascaras, fun.sRGB2Lab(color_check),cant_imagenes,type_error=type_error,OutColorSpace="Lab")
        mejores_comb.append(comb)
        min_error.append(error)
    
    graficas_error(min_error,type_error)






