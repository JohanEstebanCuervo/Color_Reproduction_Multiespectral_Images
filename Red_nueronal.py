# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 09:05:38 2021

@author: Johan Cuervo
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


datatrain = pd.read_csv('Datos_entrenamiento.csv', sep=',',names=range(1,7))
datatrain= datatrain.to_numpy()

datatest = pd.read_csv('Datos_test.csv', sep=',',names=range(1,7))
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
          batch_size=20, epochs=10,
          validation_data=(X_val, Y_val))

datatrain = pd.read_csv('Datos_test.csv', sep=',',names=range(1,7))
datatrain= datatrain.to_numpy()
X_test = datatrain[:,:3]/255*0.8+0.1
Y_test = datatrain[:,3:]
#Y_test=Y_test
predict = red.predict(X_test)

dif= np.sqrt(np.sum((Y_test-predict)**2,axis=1))
prom= np.mean(dif)
print(prom)

red.save('Correction_color_neuronal_red2.h5')