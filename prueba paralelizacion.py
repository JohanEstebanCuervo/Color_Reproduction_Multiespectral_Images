# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 13:22:19 2021

@author: Johan Cuervo
"""
import numpy as np
from os import system
import os
import funciones_reproduccion_color as fun
import multiprocessing as mp
import itertools
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
pesos_ecu = fun.Pesos_ecualizacion(imagenes_patron[:-3], mascaras[18])
imagenes_patron=(imagenes_patron[:-3].T*pesos_ecu).T/255

Cant_Image=range(9,12)

def mejor_combinacion(imagenes_patron,mascaras,color_check,Cant_Image):
    stuff= range(np.shape(imagenes_patron)[0])
    subset = list(itertools.combinations(stuff,Cant_Image))
    
    min_error=1000
    a=0
    for i,Comb in enumerate(subset):
        if(i/len(subset)*100>a):
            a+=10
            print('Cant imagenes'+str(int(Cant_Image))+' Avance:' + str("{0:.2f}".format(i/len(subset)*100))+str('%'))
    # #%%  Reproduccion de color usando CIE
        
        im_RGB= fun.ReproduccionCie1931(imagenes_patron,selec_imagenes=Comb)
        #im_Lab= cv2.cvtColor(im_RGB, cv2.COLOR_RGB2LAB)
        errores = fun.Error_de_reproduccion([im_RGB], mascaras, color_check)
        error_media = np.mean(errores,axis=1)
        #print(error_media)
        if(error_media<min_error):
            min_error=error_media
            mejor_comb=Comb
        #fun.imshow('Imagen reproducción CIE 1931',im_RGB)
    
    #%%  Reproduccion de color usando CIE
    im_RGB= fun.ReproduccionCie1931(imagenes_patron,selec_imagenes=mejor_comb)
    fun.imshow('IR ERGB CIE 1931 im '+str(int(Cant_Image)),im_RGB)
    # #imwrite('Resultados/Imagenes\IR ERGB CIE 1931 im '+str(int(Cant_Image))+'.png',im_RGB)
    
    return mejor_comb,min_error


#mejor_comb, error = fun.mejor_combinacion(imagenes_patron, mascaras, color_check, cant)

if __name__=='__main__':
    pool   = mp.Pool(processes=mp.cpu_count())
    
    resultados=pool.starmap_async(mejor_combinacion,[(imagenes_patron, mascaras, color_check,cant)for cant in Cant_Image])
    #resultados=pool.starmap_async(mejor_combinacion,[(imagenes_patron, mascaras, color_check,cant)for cant in Cant_Image])
    