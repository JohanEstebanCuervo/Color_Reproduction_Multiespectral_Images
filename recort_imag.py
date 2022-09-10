import os
import cv2
import numpy as np
folder = "Resultados/Imagenes"

files = os.listdir(folder)

for file in files:
    try:
        ind = file.rindex('.')
        term = file[ind:]
    except ValueError:
        pass
    else:
        if term == '.png':
            imagen = cv2.imread(folder + '/' + file)
            tamanio = np.shape(imagen)
            if tamanio[0] == 480 and tamanio[1] == 640:
                imagen = imagen[60:390,60:540,:]
                #imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                cv2.imwrite(folder + '/' + file, imagen) 