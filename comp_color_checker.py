'''
Fucion para generar las nuevas comparaciones de cuadros 
'''
import os 
import cv2
import numpy as np

import funciones_reproduccion_color as fun

carpeta = "Resultados/Variables"
errores4 = fun.Read_Variable(carpeta + "/" + 'errores_de_ReyCorrec_Nim_4.pickle')
errores12 = fun.Read_Variable(carpeta + "/" + 'errores_de_ReyCorrec_Nim_4.pickle')

total = errores4 + errores12
total = np.sum(total, axis=0)

max = total.argsort()[-1] + 1
min = total.argsort()[1] + 1
folder = r'C:\Users\cuerv\OneDrive\Documentos\Color_Reproduction_Multiespectral_Images\Resultados\Imagenes'
name = ['peor_parche.png','mejor_parche.png']
for count, par in enumerate([max, min]):
    punto_inicial = np.array([62, 2])
    paso = np.array([122, 62])
    num_path = par

    posx = (num_path - 1) % 6
    posy = (num_path - 1) // 6

    punto_inicial = punto_inicial + paso*[posy, posx]
    print(punto_inicial)
    imagenes_comp =[
        'Comparacion Color_Check - CIE_1931_Nim_12 espectros.png',
        'Comparacion Color_Check - CIE_1931_Nim_4 espectros.png',
        'Comparacion Color_Check - Linear_Nim_12 espectros.png',
        'Comparacion Color_Check - Linear_Nim_4 espectros.png',
        'Comparacion Color_Check - Compound_Nim_12 espectros.png',
        'Comparacion Color_Check - Compound_Nim_4 espectros.png',
        'Comparacion Color_Check - Logarithm_Nim_12 espectros.png',
        'Comparacion Color_Check - Logarithm_Nim_4 espectros.png',
        'Comparacion Color_Check - Polynomial_Nim_12 espectros.png',
        'Comparacion Color_Check - Polynomial_Nim_4 espectros.png',
        'Comparacion Color_Check - RedNeuronal_Nim_12 espectros.png',
        'Comparacion Color_Check - RedNeuronal_Nim_4 espectros.png',
        ]

    imagen = cv2.imread(folder + '/' + imagenes_comp[0])
    parche = imagen[punto_inicial[0] - 60: punto_inicial[0],punto_inicial[1]: punto_inicial[1] + 60, :]

    imagen_result = np.zeros((126,2,3)).astype('uint8')
    column = np.zeros((2,60,3))
    column = np.concatenate((column,parche),axis=0)
    column = np.concatenate((column,np.zeros((2,60,3))),axis=0)
    column = np.concatenate((column,parche),axis=0)
    column = np.concatenate((column,np.zeros((2,60,3))),axis=0)
    column = np.concatenate((column,np.zeros((126,2,3))),axis=1)

    imagen_result = np.concatenate((imagen_result, column), axis=1)
    for i in range(6):
        imagen1 = cv2.imread(folder + '/' +imagenes_comp[i*2])
        imagen2 = cv2.imread(folder + '/' +imagenes_comp[i*2 + 1])
        parche1 = imagen1[punto_inicial[0]: punto_inicial[0] + 60,punto_inicial[1]: punto_inicial[1] + 60, :]
        parche2 = imagen2[punto_inicial[0]: punto_inicial[0] + 60,punto_inicial[1]: punto_inicial[1] + 60, :]

        column = np.zeros((2,60,3))
        column = np.concatenate((column,parche1),axis=0)
        column = np.concatenate((column,np.zeros((2,60,3))),axis=0)
        column = np.concatenate((column,parche2),axis=0)
        column = np.concatenate((column,np.zeros((2,60,3))),axis=0)
        column = np.concatenate((column,np.zeros((126,2,3))),axis=1)

        imagen_result = np.concatenate((imagen_result, column), axis=1)

    imagen_result = imagen_result.astype('uint8')
    imagen_gbr = cv2.cvtColor(imagen_result.astype('uint8'), cv2.COLOR_BGR2RGB)
    cv2.imshow('resultado', imagen_result)
    cv2.imwrite(folder + '/' + name[count], imagen_gbr)
    cv2.waitKey(0) 

    cv2.destroyAllWindows() 
