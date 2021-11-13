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

C= np.reshape(color_check,(1,-1,3)).astype(int)

C = np.array(C,dtype='uint8')
color_check_Lab = cv2.cvtColor(C, cv2.COLOR_RGB2LAB)
color_check_Lab = np.reshape(color_check_Lab,(-1,3))

D65=  np.array([
      [410,	91.486000],
      [450,	117.008000],
      [470,	114.861000],
      [490,	108.811000],
      [505,	108.578000],
      [530,	107.689000],
      [560,	100.000000],
      [590,	88.685600],
      [600,	90.006200],
      [620,	87.698700],
      [630, 83.288600],
      [650,	80.026800],
      [720,	61.604000],
  
      ])

CIE1931 =  np.array([
     
     [410,	0.043510,	0.001210,	0.207400],
     [450,	0.336200,	0.038000,	1.772110],
     [470,	0.195360,	0.090980,	1.287640],
     [490,	0.032010,	0.208020,	0.465180],
     [505,	0.002400,	0.407300,	0.212300],
     [530,	0.165500,	0.862000,	0.042160],
     [560,	0.594500,	0.995000,	0.003900],
     [590,	1.026300,	0.757000,	0.001100],
     [600,	1.062200,	0.631000,	0.000800],
     [620,	0.854450,	0.381000,	0.000190],
     [630,	0.642400,	0.265000,	0.000050],
     [650,	0.283500,	0.107000,	0.000000],
     [720,	0.002899,	0.001047,	0.000000],
     ])

XYZ2RGB= np.array([[3.2406, -1.5372, -0.4986],
         [-0.9689, 1.8758, 0.0415],
         [0.0557, -0.2040, 1.0570],])

#%% Coeficientes



#Coef= CIE1931[:,1:].T

#%% busqueda de los archivos en las carpetas correspondientes

#carpeta1 = 'imgs\Patron1'
#carpeta2 = 'imgs\mascaras'
carpeta1 = 'informacion/patron'
carpeta1 = 'D:\Documentos\Articulo_Programas_Reproduccion_Color\Informacion\patron'
#carpeta1 = 'c1_renombradas/'
carpeta2 = 'informacion/mascaras'
lista1 = os.listdir(carpeta1)
lista2 = os.listdir(carpeta2)

#%% mascaras 

mascaras=fun.ext_mascaras(carpeta2, lista2)
  
#%% Organizacion de las imagenes, promedios de parches y espectro
grupo=1
lista_patron=lista1[15*(grupo-1):15*grupo]
imagenes_patron, colorn, prom, entrada, espectro = fun.Imagenes_Camara(carpeta1 ,lista_patron, mascaras, color_check)

#%% normalizamos el promedio con respecto al blanco es decir fila 19 se normalizan

pesos_ecu, prom_ecualizado_N = fun.ecualizacion(prom,19,243)
imagenes_patron_ecu = imagenes_patron#*pesos_ecu
imagenes_patron_ecu = imagenes_patron_ecu[:,:-3].T


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
        Coef= (CIE1931[Comb,1:]*(np.ones((3,1))*D65[Comb,1].T).T).T
        xyz = np.dot(Coef,imagenes_patron_ecu[Comb,:]).T/int(Cant_Image)
        maximos=np.max(xyz,axis=0)
        xyz = np.divide(xyz,np.max(xyz,axis=0))
        
        im_Y=np.reshape(xyz[:,1],(480,640))
        
        k= 1/np.mean(im_Y[np.where(mascaras[18]==255)])
        xyz*=k
        
        rgb = fun.recorte(np.dot(XYZ2RGB,xyz.T).T)
        
        im_RGB=np.reshape(rgb,(480,640,3))
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
    Coef= (CIE1931[mejor_comb,1:]*(np.ones((3,1))*D65[mejor_comb,1].T).T).T
    xyz = np.dot(Coef,imagenes_patron_ecu[mejor_comb,:]).T/Cant_Image
    maximos=np.max(xyz,axis=0)
    xyz = np.divide(xyz,np.max(xyz,axis=0))
    
    im_Y=np.reshape(xyz[:,1],(480,640))
    
    k= 1/np.mean(im_Y[np.where(mascaras[18]==255)])
    xyz*=k
    
    rgb = fun.recorte(np.dot(XYZ2RGB,xyz.T).T)
    im_RGB=np.reshape(rgb,(480,640,3))
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

