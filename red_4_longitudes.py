import pandas as pd

excel = pd.read_excel('Resultados\Variables\errores_combinaciones.xlsx', sheet_name='N_im 4')

combinaciones = excel.head(10).to_numpy()[:,:-1].astype('int')
print(combinaciones)