# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:15:32 2021

@author: Johan Cuervo
"""

import funciones_reproduccion_color as fun
import numpy as np

variable = fun.Read_Variable('Resultados/Variables/combinaciones_RGB.pickle')


espectro=np.array([410,450,470,490,505,530,560,590,600,620,630,650,720])

espectros= espectro[list(variable[5])]