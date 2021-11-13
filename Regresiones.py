# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:49:07 2021

@author: Johan Cuervo
"""

import numpy as np
from os import system
import os
import funciones_reproduccion_color as fun
import cv2
import pandas as pd

import keras


datatrain = pd.read_csv('Datos_test.csv', sep=',',names=range(1,7))
datatrain= datatrain.to_numpy()

X= datatrain[:,:3]
X=X/255
Y= datatrain[:,3:]

red = keras.models.load_model('Correction_color_neuronal_red.h5')

predic = red.predict(X)
predic = (predic*255).astype(int)