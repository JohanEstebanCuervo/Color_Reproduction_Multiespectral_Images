# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 01:32:26 2021

@author: Johan Cuervo
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

numero_imagenes = 4
for contador in [8]:
#for contador in range(10):
    archivo = (
        f"Resultados/Datos_entrenamiento/Datos_train_Nim{numero_imagenes}_comb{contador}.csv"
    )
    datatrain = pd.read_csv(archivo, sep=",", names=range(1, 7))
    datatrain = datatrain.to_numpy()

    archivo = (
        f"Resultados/Datos_entrenamiento/Datos_test_Nim{numero_imagenes}_comb{contador}.csv"
    )
    datatest = pd.read_csv(archivo, sep=",", names=range(1, 7))
    datatest = datatest.to_numpy()

    X_train = datatrain[:, :3] / 255 * 0.8 + 0.1
    X_val = datatest[:, :3] / 255 * 0.8 + 0.1
    Y_train = datatrain[:, 3:]
    Y_val = datatest[:, 3:]


    red = Sequential(
        [
            Dense(10, activation="sigmoid", input_shape=(3,)),
            # Dense(10, activation='sigmoid'),
            Dense(3, activation="elu"),
        ]
    )

    red.compile(
        optimizer="Adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
    )


    hist = red.fit(
        X_train, Y_train, batch_size=20, epochs=7, validation_data=(X_val, Y_val)
    )

    datatrain = pd.read_csv(archivo, sep=",", names=range(1, 7))
    datatrain = datatrain.to_numpy()
    X_test = datatrain[:, :3] / 255 * 0.8 + 0.1
    Y_test = datatrain[:, 3:]
    # Y_test=Y_test
    predict = red.predict(X_test)

    dif = np.sqrt(np.sum((Y_test - predict) ** 2, axis=1))
    prom = np.mean(dif)
    print(prom)

    red.save(
        f"Resultados/Variables/Correction_color_neuronal_red_Nim{numero_imagenes}_comb{contador}.h5"
    )

    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper right")
    plt.show()