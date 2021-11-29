# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 21:44:32 2021

@author: Johan Cuervo
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import system
import os
carpeta1 = 'carta_conluz/patron'
lista1 = os.listdir(carpeta1)

espectro=['(410)','(450)','(470)','(490)','(505)','(530)','(560)','(590)','(600)','(620)','(630)','(650)','(720)','(840)','(960)']

for i,nombre in enumerate(lista1[:30]):
    imagen= cv2.imread(carpeta1+"/"+nombre)
    modulo = (i+1)%15
    nombreg= 'carta_conluz/patron/'+nombre[:-5]+espectro[modulo-1]+'.png'
    cv2.imwrite(nombreg ,imagen)