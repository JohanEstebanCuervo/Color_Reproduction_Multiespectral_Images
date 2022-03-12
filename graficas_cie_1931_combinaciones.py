# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 22:18:54 2021

@author: Johan Cuervo
"""


import pandas as pd 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import funciones_reproduccion_color as fun
import os

nombre = 'CIETABLES.xls'
carpeta_guardado='Resultados/Imagenes/'
hoja  =  pd.read_excel(nombre , skiprows=4,sheet_name='Table4')

cie = np.array( hoja.iloc[:-1,:4] )
Combinaciones = fun.Read_Variable('Resultados/Variables/combinaciones_mean.pickle')

espectro=np.array([410,450,470,490,505,530,560,590,600,620,630,650,720])

mpl.rc('axes', labelsize=10)
mpl.rc('xtick', labelsize=10)
mpl.rc('ytick', labelsize=10)

matriz_cruce = np.reshape(cie[:,0],(-1,1)).astype(int) @ np.ones((1,len(espectro))) - espectro

indice = np.where(matriz_cruce==0)[0]

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


#%% mascaras 

mascaras=fun.ext_mascaras(carpeta2, lista2)
  
#%% Organizacion de las imagenes, promedios de parches y espectro
grupo=1
lista_patron=lista1[15*(grupo-1):15*grupo]

imagenes_patron,shape_imag = fun.Read_Multiespectral_imag(carpeta1, lista_patron)
pesos_ecu = fun.Pesos_ecualizacion(imagenes_patron[:-3], mascaras[18])
imagenes_patron=(imagenes_patron[:-3].T*pesos_ecu).T/255
espectro = fun.Read_espectros_Imag(lista_patron)
color_RGB_pixel_ideal = fun.Ideal_Color_Patch_pixel(color_check, mascaras)

for i,Comb in enumerate(Combinaciones):
    indices= indice[list(Comb)]
    plt.figure(figsize=(4,3))
    plt.plot(cie[:,0],cie[:,1],color='r')
    plt.plot(cie[:,0],cie[:,2],color='g')
    plt.plot(cie[:,0],cie[:,3],color='b')
    
    m,n,base= plt.stem(cie[indices,0],cie[indices,1], linefmt='black', markerfmt='None', use_line_collection=False)
    plt.setp(base, 'linewidth', 0)
    m,n,base= plt.stem(cie[indices,0],cie[indices,2], linefmt='black', markerfmt='None', use_line_collection=False)
    plt.setp(base, 'linewidth', 0)
    m,n,base= plt.stem(cie[indices,0],cie[indices,3], linefmt='black', markerfmt='None', use_line_collection=False)
    plt.setp(base, 'linewidth', 0)
    
    
    #plt.title('CIE 1931')
    plt.xlabel('$\lambda$ nm')
    plt.legend(('X','Y','Z'))
    plt.savefig(carpeta_guardado+'CIE1931_Nim_'+str(i+1)+'.pdf', format='pdf')
    plt.show()
    
    im_RGB= fun.ReproduccionCie1931(imagenes_patron,selec_imagenes=Comb)

    fun.imshow('Reproducción CIE 1931',im_RGB)
    fun.imwrite('Resultados/Imagenes/reproduccion_CIE_Comb_Nim_'+str(i+1)+'.png',im_RGB)


