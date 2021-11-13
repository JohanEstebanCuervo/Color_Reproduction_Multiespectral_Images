# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 16:29:02 2021

@author: Johan Cuervo
"""

import pandas as pd 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


nombre = 'CIETABLES.xls'
carpeta_guardado='Resultados/Imagenes/'
hoja  =  pd.read_excel(nombre , skiprows=4,sheet_name='Table4')
cie = np.array( hoja.iloc[:-1,:4] )

espectro=np.array([410,450,470,490,505,530,560,590,600,620,630,650,720])

mpl.rc('axes', labelsize=10)
mpl.rc('xtick', labelsize=10)
mpl.rc('ytick', labelsize=10)

matriz_cruce = np.reshape(cie[:,0],(-1,1)).astype(int) @ np.ones((1,len(espectro))) - espectro

indices = np.where(matriz_cruce==0)[0]

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
plt.savefig(carpeta_guardado+'CIE1931.pdf', format='pdf')
plt.show()

hoja2  =  pd.read_excel(nombre , skiprows=5,sheet_name='Table1')
D65 = np.array( hoja2.iloc[:,:3] )

plt.figure(figsize=(4,3))
plt.plot(D65[:,0],D65[:,2],color='black')
#plt.title('Standard Iluminant D65')
plt.xlabel('$\lambda$ nm')
plt.legend(('D65'))
plt.savefig(carpeta_guardado+'standard_iluminant_d65.pdf', format='pdf')
plt.show()

cie_d65= (cie[:,1:]*(np.ones((3,1))*D65[16:,2].T).T)

plt.figure(figsize=(4,3))
plt.plot(cie[:,0],cie_d65[:,0],color='r')
plt.plot(cie[:,0],cie_d65[:,1],color='b')
plt.plot(cie[:,0],cie_d65[:,2],color='g')
plt.title('CIE 1931 por Standard Iluminant D65')
plt.xlabel('$\lambda$ nm')
plt.legend(('X*D65','Y*D65','Z*D65'))
plt.show()

