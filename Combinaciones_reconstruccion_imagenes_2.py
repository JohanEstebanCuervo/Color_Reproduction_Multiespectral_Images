# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 02:15:02 2021

@author: Johan Cuervo
"""


import numpy as np
from os import system
import os
import funciones_reproduccion_color as fun
import itertools
import cv2
import matplotlib.pyplot as plt
import pickle
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
imagenes_patron/=255
#%% Combinaciones
stuff = range(12)
combinaciones=[]
errores_comb=[]
N=1
for Cant_Image in np.linspace(12,N,12-N+1):
    
    subset = list(itertools.combinations(stuff,int(Cant_Image)))
    
    min_error=1000
    a=0
    for i,Comb in enumerate(subset):
        if(i/len(subset)*100>a):
            a+=10
            print('Cant imagenes'+str(int(Cant_Image))+' Avance:' + str("{0:.2f}".format(i/len(subset)*100))+str('%'))
    #%%  Reproduccion de color usando CIE
        
        im_RGB= fun.ReproduccionCie1931(imagenes_patron,selec_imagenes=Comb)
        #im_Lab= cv2.cvtColor(im_RGB, cv2.COLOR_RGB2LAB)
        errores = fun.Error_de_reproduccion([im_RGB], mascaras, color_check)
        error_media = np.mean(errores,axis=1)
        #print(error_media)
        if(error_media<min_error):
            min_error=error_media
            mejor_comb=Comb
        #fun.imshow('Imagen reproducción CIE 1931',im_RGB)
    
    print(" ")
    combinaciones +=[mejor_comb]
    errores_comb +=[min_error]
    #%%  Reproduccion de color usando CIE
    im_RGB= fun.ReproduccionCie1931(imagenes_patron[:-3],selec_imagenes=mejor_comb)
    fun.imshow('IR ERGB CIE 1931 im '+str(int(Cant_Image)),im_RGB)
    fun.imwrite('Resultados/Imagenes\IR ERGB CIE 1931 im '+str(int(Cant_Image))+'.png',im_RGB)
    
    
plt.figure(figsize=(4,3))
plt.plot(np.linspace(12,N,12-N+1).astype(int),np.array(errores_comb),color='black')
plt.title('Error RGB en funcion de las imagenes')
plt.xlabel('Cantidad Im')
plt.savefig('Resultados/Imagenes/Grafica_error_RGB.pdf', format='pdf')
plt.show()

fichero = open('Resultados/Variables/combinaciones_RGB.pickle','wb')
pickle.dump(combinaciones,fichero)
fichero.close()