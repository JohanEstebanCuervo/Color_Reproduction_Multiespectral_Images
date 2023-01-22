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

nombre = "CIETABLES.xls"
carpeta_guardado = "Resultados/Imagenes/"
hoja = pd.read_excel(nombre, skiprows=4, sheet_name="Table4")

cie = np.array(hoja.iloc[:-1, :4])
Combinaciones = fun.Read_Variable("Resultados/Variables/combinaciones_mean.pickle")

espectro = np.array([410, 450, 470, 490, 505, 530, 560, 590, 600, 620, 630, 650, 720])

mpl.rc("axes", labelsize=10)
mpl.rc("xtick", labelsize=10)
mpl.rc("ytick", labelsize=10)

matriz_cruce = (
    np.reshape(cie[:, 0], (-1, 1)).astype(int) @ np.ones((1, len(espectro))) - espectro
)

indice = np.where(matriz_cruce == 0)[0]

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

carpeta1 = "informacion/patron"
carpeta2 = "informacion/mascaras"
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
espectro = fun.Read_espectros_Imag(lista_patron)
color_RGB_pixel_ideal = fun.Ideal_Color_Patch_pixel(color_check, mascaras)

plt.figure(figsize=(12, 8))
plt.grid()
plt.plot(cie[:, 0], cie[:, 1], color="r")
plt.plot(cie[:, 0], cie[:, 2], color="g")
plt.plot(cie[:, 0], cie[:, 3], color="b")

plt.title('CIE 1931')
plt.xlabel("$\lambda$ nm", fontsize=20)
plt.legend(("X", "Y", "Z"), fontsize=12)
plt.savefig(carpeta_guardado + "CIE1931.pdf", format="pdf")
plt.show()

hoja = pd.read_excel(nombre, skiprows=5, sheet_name="Table1")

d65 = np.array(hoja.iloc[:, :4])

plt.figure(figsize=(12, 8))
plt.grid()
plt.plot(d65[:, 0], d65[:, 2], color="black")

plt.title('Standard Illuminant D65')
plt.xlabel("$\lambda$ nm", fontsize=20)
plt.legend(("D65"), fontsize=12)
plt.savefig(carpeta_guardado + "Illuminant_D65.pdf", format="pdf")
plt.show()

train = [3400, 470, 50, 1, 0.91, 0.7, 0.6]
val = [550, 50, 40, 0.91, 0.7, 0.6, 0.5]
x = range(7)

plt.figure(figsize=(12, 8))
plt.grid()
plt.plot(x, train, color="cyan")
plt.plot(x, val, color="orange")

plt.title('Model loss')
plt.xlabel("Epoch", fontsize=20)
plt.ylabel("Loss", fontsize=20)
plt.legend(("Train","Val"), fontsize=12)
plt.savefig(carpeta_guardado + "Red_loss.pdf", format="pdf")
plt.show()