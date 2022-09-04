# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 23:47:43 2021

@author: Johan Cuervo
"""

import serial
import time


Error_check = 0
Lista_Leds = [
    "M02N",
    "M03N",
    "M04N",
    "M05N",
    "M06N",
    "M07N",
    "M08N",
    "M09N",
    "M0AN",
    "M0BN",
    "M0CN",
    "M0DN",
    "M0EN",
    "M0FN",
]

comunicacion = serial.Serial("COM16", 57600)

comunicacion.write("W".encode("utf-8"))
time.sleep(0.5)

if comunicacion.inWaiting() == 1:
    print(2)
    Check = comunicacion.read()

comunicacion.write("W".encode("utf-8"))


comunicacion.close()

# for Led in Lista_Leds:

#         try:
#             bandera=0
#             iteraciones=0

#             while bandera==0 and iteraciones<6:

#                 comunicacion.write(Led.encode('utf-8'))
#                 time.sleep(1e-3)

#                 Check=''
#                 if comunicacion.inWaiting()==1:
#                     print(2)
#                     Check = comunicacion.read()

#                 if Check == b'O':
#                     print("Comando "+Led +" Aceptado")
#                     bandera=1

#                 print(Check)
#                 iteraciones+=1

#             if(bandera==0):
#                 warnings.warn('Error en Comunicación')
#                 warnings.simplefilter('No se recibe respuesta del puerto Serial', UserWarning)
#             comunicacion.write('W'.encode('utf-8'))

#             time.sleep(1e-3)

#             Check=''
#             if comunicacion.inWaiting()==1:
#                 Check = comunicacion.read()
#                 grab_next_image_by_trigger(cam,Led)
#                 #grab_next_image_by_trigger(cam,Led)

#             if Check == b'O':
#                 print("Comando W Aceptado")


#             time.sleep(1)

#         except:
#             print("Comunicación Fracasada")
#             Error_check=1
#             break
