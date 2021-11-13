# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 23:10:40 2021

@author: Johan Cuervo
"""

import sympy as sym
import numpy as np
import funciones_reproduccion_color as fun
import os

carpeta = 'Resultados/Variables'
lista = os.listdir(carpeta)


for nombre in lista:
    if nombre[:4]=='CCM_':
        CCm = fun.Read_Variable(carpeta +'/'+ nombre)
        
        nombre_archivo = nombre[:nombre.find('.')]
        file = open('Resultados/Formulas_Latex/'+nombre_archivo+'.tex','w')
        matrix= sym.latex(sym.Matrix(CCm.round(3)), full_prec=False)
        texto='\n'
        n=0
        
        for i,carac in enumerate(matrix):
            if carac=='\\' and matrix[i+1] == '\\':
              texto+=matrix[n:i+1]+'\\'+' \n'
              n=i+2
        
        texto+=matrix[n:]
        
        file.write('\\begin{equation}\n')
        file.write('\\begin{bmatrix}\n  R_s \\\ G_s \\\ B_s \n\end{bmatrix}=')
        file.write(texto)
        file.write('\n\\begin{bmatrix}\n  R \\\ G \\\ B \\\ 1 \n\end{bmatrix}\n')
        file.write('\end{equation}\n')
        
        file.close()