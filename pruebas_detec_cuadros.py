# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 16:21:39 2021

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
import funciones_deteccion_cuadros as dtec_


carpeta1 = 'Carta Color\Carta20'
#carpeta1 = 'imgs\Patron0'
#carpeta1 = 'Carta Color\Carta153'
carpeta2 = 'imgs\mascaras'
lista1 = os.listdir(carpeta1)
lista2 = os.listdir(carpeta2)

prueba= dtec_.mascaras(carpeta1,lista1[:13])