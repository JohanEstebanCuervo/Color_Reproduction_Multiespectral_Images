import funciones_reproduccion_color as fun
import numpy as np

folder = "Resultados/Variables"
nombre = 'errores_de_ReyCorrec_Nim_4.pickle'

errores = fun.Read_Variable(folder + "/" + nombre).round(3)
print(np.shape(errores))