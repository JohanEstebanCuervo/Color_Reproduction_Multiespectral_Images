# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 01:42:11 2021

@author: Johan Cuervo
"""

import funciones_reproduccion_color as fun
import numpy as np


im_rgb,mascaras,color_check = fun.ReproduccionCie1931('informacion')
im_rgb2,_,_ = fun.ReproduccionCie1931('informacion',grupo=8)
fun.imshow('imagen reconstruida', im_rgb)
fun.imshow('imagen reconstruida', im_rgb2)

imagenes= [im_rgb]
errores = fun.Error_de_reproduccion(imagenes, mascaras, color_check)
errores_media = np.mean(errores,axis=1)