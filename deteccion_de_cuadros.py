# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 15:35:03 2021

@author: Johan Cuervo
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import system
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import funciones_reproduccion_color as func
import funciones_deteccion_cuadros as detec


carpeta1 = 'patronTam'
#carpeta1 = 'imgs\Patron0'

lista1 = os.listdir(carpeta1)

carpetaguardado = 'mascarasTam/'

Mascaras, centros, centrosorg = detec.Principal(carpeta1,lista1[:15])

for i,mascara in enumerate(Mascaras):
    if i+1<10:
        cv2.imwrite(carpetaguardado+'mascara0'+str(i+1)+'.png',mascara )
    else:
        cv2.imwrite(carpetaguardado+'mascara'+str(i+1)+'.png',mascara )