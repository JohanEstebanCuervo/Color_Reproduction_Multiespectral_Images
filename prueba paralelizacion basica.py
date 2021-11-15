# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 22:36:29 2021

@author: Johan Cuervo
"""

import numpy as np
from os import system
import os
import funciones_reproduccion_color as fun
import multiprocessing as mp
import itertools
import time


def funcion(x,a):
    print("inicio de proceso: ",a)
    result = x*a
    time.sleep(result)
    print("fin de proceso: ",a)
    return result

x= 1
N=8

resultados2=[]
tic = time.time()

for a in range(N):
    resultados2+= [funcion(x,a)]
    
toc = time.time()

print('tiempo de ejecucion: ',toc-tic)


tic = time.time()

if __name__=='__main__':
    pool   = mp.Pool(mp.cpu_count())
    resultados=pool.starmap(funcion,[(x,a)for a in range(N)])
    #resultados=pool.starmap_async(mejor_combinacion,[(imagenes_patron, mascaras, color_check,cant)for cant in Cant_Image])

toc = time.time()

print('tiempo de ejecucion: ',toc-tic)



