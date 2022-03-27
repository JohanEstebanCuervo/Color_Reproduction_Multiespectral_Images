# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:49:42 2021

@author: Johan Cuervo
"""

import serial
import time
import Funciones_Adquisicion as Fun_Ad


Lista_Leds = ['M02N','M03N','M04N','M05N','M06N','M07N','M08N','M09N','M0AN','M0BN','M0CN','M0DN','M0EN','M0FN']


puerto = Fun_Ad.Serial_Port_Select()  # si se conoce el puerto se puede agregar el parametro port='Nombre del puerto'
comunicacion = serial.Serial(puerto,57600)
comunicacion.isOpen()

sleep= 0.01
for Led in Lista_Leds:

	print(Led)
	comunicacion.isOpen()
	comunicacion.write(Led.encode('utf-8'))
	time.sleep(sleep)

	if comunicacion.inWaiting()==1:
		Check = comunicacion.read()
	          
		print(Check)

	comunicacion.write('W'.encode('utf-8'))

	time.sleep(sleep)
	if comunicacion.inWaiting()==1:
		Check = comunicacion.read()
	          
		print(Check)

comunicacion.close()

