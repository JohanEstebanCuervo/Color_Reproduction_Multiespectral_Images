# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:04:59 2021

@author: Johan Cuervo
"""

import numpy as np
import os
import funciones_reproduccion_color as fun
import pandas as pd
import keras

#%% borrar todo lo cargado anteriormente
# system("cls")
archivo = r"D:\Documentos\Articulo_Programas_Reproduccion_Color\Resultados\Datos_entrenamiento/Datos_entrenamiento.csv"

numero_imagenes = 4
excel = pd.read_excel('Resultados\Variables\errores_combinaciones.xlsx', sheet_name='N_im 4')

combinaciones = excel.head(10).to_numpy()[:,:-1].astype('int')

#%% barra de colores para mostrar grafico
color_check = np.array(
    [
        [116, 81, 67],
        [199, 147, 129],
        [91, 122, 156],
        [90, 108, 64],
        [130, 128, 176],
        [92, 190, 172],
        [224, 124, 47],
        [68, 91, 170],
        [198, 82, 97],
        [94, 58, 106],
        [159, 189, 63],
        [230, 162, 39],
        [34, 63, 147],
        [67, 149, 74],
        [180, 49, 57],
        [238, 198, 32],
        [193, 84, 151],
        [12, 136, 170],
        [243, 238, 243],
        [200, 202, 202],
        [161, 162, 161],
        [120, 121, 120],
        [82, 83, 83],
        [49, 48, 51],
    ]
)


#%% busqueda de los archivos en las carpetas correspondientes

carpeta1 = "Informacion/patron"
carpeta2 = "Informacion/mascaras"
lista1 = os.listdir(carpeta1)
lista2 = os.listdir(carpeta2)


#%% mascaras

mascaras = fun.ext_mascaras(carpeta2, lista2)

#%% Organizacion de las imagenes, promedios de parches y espectro
grupo = 1
lista_patron = lista1[15 * (grupo - 1) : 15 * grupo]

imagenes_patron, shape_imag = fun.Read_Multiespectral_imag(carpeta1, lista_patron)
pesos_ecu = fun.Pesos_ecualizacion(imagenes_patron[:-3], mascaras[18])
imagenes_patron = (imagenes_patron[:-3].T * pesos_ecu).T / 255
# espectro = fun.Read_espectros_Imag(lista_patron)
color_check_lab = fun.sRGB2Lab(color_check)

for contador in range(10):
    red = keras.models.load_model(
        f"Resultados/Variables/Correction_color_neuronal_red_Nim{numero_imagenes}_comb{contador}.h5"
    )

    im_RGB = fun.ReproduccionCie1931(
        imagenes_patron,
        selec_imagenes=combinaciones[contador],
    )

    fun.imshow("Reproducción CIE 1931", im_RGB)

    rgb = np.reshape(im_RGB, (-1, 3))
    predic = red.predict(rgb * 0.8 + 0.1)

    im_RGB_NN = fun.recorte(np.reshape(predic, (480, 640, 3)) / 255)
    fun.imshow("Imagen mejorada mediante Red neuronal", im_RGB_NN)

    imagenes = [
        fun.sRGB2Lab(im_RGB),
        fun.sRGB2Lab(im_RGB_NN)
    ]

    errores = fun.Error_de_reproduccion(imagenes, mascaras, color_check_lab)
    #print(errores)
    errores_media = np.mean(errores, axis=1)
    print(errores_media)
