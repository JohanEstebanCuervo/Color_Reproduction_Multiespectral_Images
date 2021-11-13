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


nombre = 'CIETABLES.xls'
carpeta_guardado='Resultados/Imagenes/'
hoja  =  pd.read_excel(nombre , skiprows=4,sheet_name='Table4')

cie = np.array( hoja.iloc[:-1,:4] )
Combinaciones = fun.Read_Variable('Resultados/Variables/combinaciones_RGB.pickle')

espectro=np.array([410,450,470,490,505,530,560,590,600,620,630,650,720])

mpl.rc('axes', labelsize=10)
mpl.rc('xtick', labelsize=10)
mpl.rc('ytick', labelsize=10)

matriz_cruce = np.reshape(cie[:,0],(-1,1)).astype(int) @ np.ones((1,len(espectro))) - espectro

indice = np.where(matriz_cruce==0)[0]


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
    plt.savefig(carpeta_guardado+'CIE1931_Nim_'+str(12-i)+'.pdf', format='pdf')
    plt.show()


