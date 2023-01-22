# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 20:16:05 2022

@author: cuerv
"""

import cv2
import os

carpeta = r'C:\Users\cuerv\OneDrive\Documentos\PeerJ-True-Color-Reproduction\Resultados\Imagenes'

archivos = os.listdir(carpeta)
imagenes = []
for file in archivos:
    ind = file.index('.')
    if file[ind:] == '.png':
        imagenes.append(file)

for imagen in imagenes:
    image = cv2.imread(carpeta + '/' + imagen)
    ancho = image.shape[1] #columnas
    alto = image.shape[0] # filas

    if ancho == 640 and alto == 480:
        print(imagen)
        M = cv2.getRotationMatrix2D((ancho//2,alto//2),-3,1)
        imageOut = cv2.warpAffine(image,M,(ancho,alto))
        imageOut = imageOut[65:392,65:530,:]
        
        cv2.imwrite(carpeta + '/' + imagen, imageOut)
