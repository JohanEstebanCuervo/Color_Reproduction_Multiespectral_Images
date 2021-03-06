# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 01:32:26 2021

@author: Johan Cuervo
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

numero_imagenes=12
archivo='D:\Documentos\Articulo_Programas_Reproduccion_Color\Resultados\Datos_entrenamiento/Datos_train_Nim'+str(numero_imagenes)+'.csv'
datatrain = pd.read_csv(archivo, sep=',',names=range(1,7))
datatrain= datatrain.to_numpy()

archivo='D:\Documentos\Articulo_Programas_Reproduccion_Color\Resultados\Datos_entrenamiento/Datos_test_Nim'+str(numero_imagenes)+'.csv'
datatest = pd.read_csv(archivo, sep=',',names=range(1,7))
datatest= datatest.to_numpy()

X_train = datatrain[:,:3]/255*0.8+0.1
X_val =   datatest[:,:3]/255*0.8+0.1
Y_train = datatrain[:,3:]
Y_val =   datatest[:,3:]
#X_train = X_train
#Y_train = Y_train
#X_val = X_val
#Y_val = Y_val

red = Sequential([
    Dense(10, activation='sigmoid', input_shape=(3,)),
    #Dense(10, activation='sigmoid'),
    Dense(3, activation='elu'),
])

red.compile(optimizer='Adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])


hist = red.fit(X_train, Y_train,
          batch_size=20, epochs=7,
          validation_data=(X_val, Y_val))

datatrain = pd.read_csv(archivo, sep=',',names=range(1,7))
datatrain= datatrain.to_numpy()
X_test = datatrain[:,:3]/255*0.8+0.1
Y_test = datatrain[:,3:]
#Y_test=Y_test
predict = red.predict(X_test)

dif= np.sqrt(np.sum((Y_test-predict)**2,axis=1))
prom= np.mean(dif)
print(prom)

red.save('Resultados/Variables/Correction_color_neuronal_red_Nim'+str(numero_imagenes)+'.h5')