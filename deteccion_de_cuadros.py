# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 15:35:03 2021

@author: Johan Cuervo
"""
import cv2
import os
import funciones_deteccion_cuadros as detec


carpeta1 = 'Fotos_nuevas/patron'
#carpeta1 = 'imgs\Patron0'

lista1 = os.listdir(carpeta1)

carpetaguardado = 'Fotos_nuevas/mascaras/'

Mascaras, centros, centrosorg = detec.Principal(carpeta1,lista1[:15])

for i,mascara in enumerate(Mascaras):
    if i+1<10:
        cv2.imwrite(carpetaguardado+'mascara0'+str(i+1)+'.png',mascara )
    else:
        cv2.imwrite(carpetaguardado+'mascara'+str(i+1)+'.png',mascara )