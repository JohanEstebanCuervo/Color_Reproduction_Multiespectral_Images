# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 23:10:40 2021

@author: Johan Cuervo
"""

import sympy as sym
import numpy as np
import funciones_reproduccion_color as fun
import os
import matplotlib.pyplot as plt
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

#%% Geneacion de tablas de errores y graficas

    if nombre[:11]=='errores_de_':
        errores = fun.Read_Variable(carpeta +'/'+ nombre).round(3)
        
        nombre_archivo = nombre[:nombre.find('.')]
        file = open('Resultados/Formulas_Latex/'+'Tabla_de_'+nombre_archivo+'.tex','w')
        
        
        file.write('\\begin{table}[H]\n')
        file.write('  \caption{Euclidean distance of each patch for '+nombre_archivo[-2:]+' images }\n')
        file.write('  \\begin{center}\n')
        file.write('    \\begin{tabularx}{\\textwidth}{r c c c c c c c c}\n')
        file.write('    \\toprule\n')
        
        num_errores= np.shape(errores)[0]
        nombres = ['Reproduction', 'Linear', 'Compound','Logarithm','Polynomial','Neural Network']
        numero_parche=1
        maximos= np.max(errores,axis=1)
        minimos= np.min(errores,axis=1)
        for fila in range(int(num_errores*3+3)):
            file.write('        ')
            for columna in range(9):
                if(fila%(num_errores+1)==0):
                    if(columna==0):
                        file.write('\\textbf{Patch Number}')
                    else:
                        file.write(' & \\textbf{'+str(numero_parche)+'}')
                        numero_parche+=1
                
                else:
                    if(columna==0):
                        ind=fila%(num_errores+1)-1
                        file.write('\\textbf{'+nombres[ind]+'}')
                    else:
                        ind1= fila%(num_errores+1)-1
                        ind2= int(fila/(num_errores+1))*8+columna-1
                        if(maximos[ind1]==errores[ind1,ind2]):
                            file.write(' &\cellcolor{colorred}{'+str(errores[ind1,ind2])+'}')
                        elif(minimos[ind1]==errores[ind1,ind2]):
                            file.write(' &\cellcolor{colorgreen}{'+str(errores[ind1,ind2])+'}')
                        else:
                            file.write(' &'+str(errores[ind1,ind2]))
                        
            if(fila%(num_errores+1)==0 or fila%(num_errores+1)==6):
                file.write('\\\ \midrule \n')
            
            else:
                file.write('\\\ \n')
        
        file.write('    \\bottomrule\n')
        file.write('    \end{tabularx}\n')
        file.write('  \end{center}\n')
        file.write('\end{table}\n')
        file.close()
        
# =============================================================================
#         Tabla de errores de media
# =============================================================================

        file = open('Resultados/Formulas_Latex/'+'Tabla_medias_'+nombre_archivo+'.tex','w')
        file.write('\\begin{table}[H]\n')
        file.write('  \caption{Average errors of the proposed methods for '+nombre_archivo[-2:]+' images }\n')
        file.write('  \\newcolumntype{C}{>{\\centering\\arraybackslash}X}')
        file.write('    \\begin{tabularx}{\\textwidth}{C C}\n')
        file.write('    \\toprule\n')
        
        medias= np.mean(errores,axis=1).round(3)
        
        for fila in range(num_errores):
            file.write('        ')
            file.write('\\textbf{'+nombres[fila]+'}')
            file.write(' & '+str(medias[fila])+'\\\ \n')
        
        file.write('    \\bottomrule\n')
        file.write('    \end{tabularx}\n')
        file.write('\end{table}\n')
        file.close()
         
        
        plt.figure(figsize=(12,8))
        for i in range(num_errores):
            plt.plot(range(1,25),errores[i])
        plt.xlabel('patch number',fontsize=20)
        plt.ylabel('$\Delta$E',fontsize=20)
        plt.legend(nombres,fontsize=12)
        plt.savefig('Resultados/Imagenes/grafica_error_Nim'+nombre_archivo[-2:]+'.pdf', format='pdf')
        plt.show()
        
        plt.figure(figsize=(12,8))
        for i in [0,num_errores-2,num_errores-1]:
            plt.plot(range(1,25),errores[i])
        plt.xlabel('patch number',fontsize=20)
        plt.ylabel('$\Delta$E',fontsize=20)
        plt.legend(nombres,fontsize=15)
        plt.savefig('Resultados/Imagenes/grafica_error2_Nim'+nombre_archivo[-2:]+'.pdf', format='pdf')
        plt.show()