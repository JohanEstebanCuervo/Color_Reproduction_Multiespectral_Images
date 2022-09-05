# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 23:10:40 2021

@author: Johan Cuervo
"""

import sympy as sym
import numpy as np
import funciones_reproduccion_color as fun
import os
import matplotlib.pyplot as plt

carpeta = "Resultados/Variables"
lista = os.listdir(carpeta)

def generate_ccm(folder, nombre):
    CCm = fun.Read_Variable(folder + "/" + nombre)

    nombre_archivo = nombre[: nombre.find(".")]
    file = open("Resultados/Formulas_Latex/" + nombre_archivo + ".tex", "w")
    matrix = sym.latex(sym.Matrix(CCm.round(3)), full_prec=False)
    texto = "\n"
    n = 0

    for i, carac in enumerate(matrix):
        if carac == "\\" and matrix[i + 1] == "\\":
            texto += matrix[n : i + 1] + "\\" + " \n"
            n = i + 2

    texto += matrix[n:]

    file.write("\\begin{equation}\n")
    file.write("\\begin{bmatrix}\n  R_s \\\ G_s \\\ B_s \n\end{bmatrix}=")
    file.write(texto)
    file.write("\n\\begin{bmatrix}\n  R \\\ G \\\ B \\\ 1 \n\end{bmatrix}\n")
    file.write("\end{equation}\n")
    file.close()


def generate_table(folder, nombre):
    errores = fun.Read_Variable(folder + "/" + nombre).round(3)

    nombre_archivo = nombre[: nombre.find(".")]
    file = open(
        "Resultados/Formulas_Latex/" + "Tabla_de_" + nombre_archivo + ".tex", "w"
    )

    file.write("\\begin{table}[H]\n")
    file.write(
        f"  \caption{'{'}\label{'{'}tab:Euclidean distance of each patch for {nombre_archivo[-2:]} images{'}'}"
        + f"Error deltaE of each patch using {nombre_archivo[-2:]} wavelengths{'}'}\n"
    )
    file.write("  \\begin{center}\n")
    file.write("    \\begin{tabularx}{\\textwidth}{r c c c c c c c c}\n")
    file.write("    \\toprule\n")

    num_errores = np.shape(errores)[0]
    nombres = [
        "Reproduction",
        "Linear",
        "Compound",
        "Logarithm",
        "Polynomial",
        "Neural Network",
    ]
    numero_parche = 1
    maximos = np.max(errores, axis=1)
    minimos = np.min(errores, axis=1)
    for fila in range(int(num_errores * 3 + 3)):
        file.write("        ")
        for columna in range(9):
            if fila % (num_errores + 1) == 0:
                if columna == 0:
                    file.write("\\textbf{Patch Number}")
                else:
                    file.write(" & \\textbf{" + str(numero_parche) + "}")
                    numero_parche += 1

            else:
                if columna == 0:
                    ind = fila % (num_errores + 1) - 1
                    file.write("\\textbf{" + nombres[ind] + "}")
                else:
                    ind1 = fila % (num_errores + 1) - 1
                    ind2 = int(fila / (num_errores + 1)) * 8 + columna - 1
                    if maximos[ind1] == errores[ind1, ind2]:
                        file.write(
                            " &\cellcolor{colorred}{"
                            + str(errores[ind1, ind2])
                            + "}"
                        )
                    elif minimos[ind1] == errores[ind1, ind2]:
                        file.write(
                            " &\cellcolor{colorgreen}{"
                            + str(errores[ind1, ind2])
                            + "}"
                        )
                    else:
                        file.write(" &" + str(errores[ind1, ind2]))

        if fila % (num_errores + 1) == 0 or fila % (num_errores + 1) == 6:
            file.write("\\\ \midrule \n")

        else:
            file.write("\\\ \n")

    file.write("    \\bottomrule\n")
    file.write("    \end{tabularx}\n")
    file.write("  \end{center}\n")
    file.write("\end{table}\n")
    file.close()

def extract_num(nombre: str) -> str:
    nombre = nombre[:-7].split('_')
    numbers = [int(val)for val in nombre if val.isdigit()]
    if numbers:
        return str(numbers[0])
    else:
        return '00'

def generate_table_sep(folder, nombre):
    errores_metodos = fun.Read_Variable(folder + "/" + nombre).round(3)
    nombres_metodos = [
        "Reproduction",
        "Linear",
        "Compound",
        "Logarithm",
        "Polynomial",
        "Neural Network",
    ]
    for count, metodo in enumerate(nombres_metodos):
        nombres = [metodo, r'\textbf{\% $\Delta$E}']
        errores = errores_metodos[count,:].reshape((1,-1))
        por_error =  errores / np.sqrt(3* 255**2) * 100
        for i, error in enumerate(por_error[0]):
            por_error[0,i] = round(error, 3)

        errores = np.concatenate((errores, por_error), axis=0)

        nombre_archivo = nombre[: nombre.find(".")] + '_' + metodo
        file = open(
            "Resultados/Formulas_Latex/" + "Tabla_de_" + nombre_archivo + ".tex", "w"
        )

        file.write("\\begin{table}[H]\n")
        file.write(
            r"  \caption{Measured Lab error and \% error of each patch using "
            + metodo
            + f" correction with {extract_num(nombre)} wavelengths "
            + r".}\n"
        )
        file.write("  \\begin{center}\n")
        file.write("    \\begin{tabularx}{\\textwidth}{r c c c c c c c c}\n")
        file.write("    \\toprule\n")

        num_errores = np.shape(errores)[0]

        numero_parche = 1
        maximos = np.max(errores, axis=1)
        minimos = np.min(errores, axis=1)
        for fila in range(int(num_errores * 3 + 3)):
            file.write("        ")
            for columna in range(9):
                if fila % (num_errores + 1) == 0:
                    if columna == 0:
                        file.write("\\textbf{Patch Number}")
                    else:
                        file.write(" & \\textbf{" + str(numero_parche) + "}")
                        numero_parche += 1

                else:
                    if columna == 0:
                        ind = fila % (num_errores + 1) - 1
                        file.write("\\textbf{" + nombres[ind] + "}")
                    else:
                        ind1 = fila % (num_errores + 1) - 1
                        ind2 = int(fila / (num_errores + 1)) * 8 + columna - 1
                        if maximos[ind1] == errores[ind1, ind2]:
                            file.write(
                                " &\cellcolor{colorred}{"
                                + str(errores[ind1, ind2])
                                + "}"
                            )
                        elif minimos[ind1] == errores[ind1, ind2]:
                            file.write(
                                " &\cellcolor{colorgreen}{"
                                + str(errores[ind1, ind2])
                                + "}"
                            )
                        else:
                            file.write(" &" + str(errores[ind1, ind2]))

            if fila % (num_errores + 1) == 0 or fila % (num_errores + 1) == 2:
                file.write("\\\ \midrule \n")

            else:
                file.write("\\\ \n")

        file.write("    \\bottomrule\n")
        file.write("    \end{tabularx}\n")
        file.write("  \end{center}\n")
        file.write("\end{table}\n")
        file.write(f'% medias: {round(np.mean(errores[0]),3)} , {round(np.mean(errores[1]),3)}')
        file.close()

def generate_median(folder, nombre):
    nombre_archivo = nombre[: nombre.find(".")]
    errores = fun.Read_Variable(folder + "/" + nombre).round(3)
    num_errores = np.shape(errores)[0]
    nombres = [
        "Reproduction",
        "Linear",
        "Compound",
        "Logarithm",
        "Polynomial",
        "Neural Network",
    ]

    file = open(
        "Resultados/Formulas_Latex/" + "Tabla_medias_" + nombre_archivo + ".tex",
        "w",
    )
    file.write("\\begin{table}[H]\n")
    file.write(
        r"  \caption{"
        r"\label{tab:Average errors of the proposed methods}" +
        r"Average errors deltaE and \% of the proposed methods using "  +
        f"{nombre_archivo[-2:]} wavelengths{'}'}\n"
    )
    file.write("  \\newcolumntype{C}{>{\\centering\\arraybackslash}X}")
    file.write("    \\begin{tabularx}{\\textwidth}{C C C}\n")
    file.write("    \\toprule\n")
    file.write(r'      \textbf{method} & \textbf{$\Delta E$} & \textbf{\% $\Delta E$} \\ \midrule')
    file.write('\n')

    medias = np.mean(errores, axis=1).round(3)
    por_error =  (medias / np.sqrt(3* 255**2) * 100).round(3)
    for fila in range(num_errores):
        file.write("      ")
        file.write("\\textbf{" + nombres[fila] + "}")
        file.write(" & " + str(medias[fila]))
        file.write(" & " + str(por_error[fila]) + "\\\ \n")

    file.write("    \\bottomrule\n")
    file.write("    \end{tabularx}\n")
    file.write("\end{table}\n")
    file.close()


def generate_graphics(folder, nombre):
    errores = fun.Read_Variable(folder + "/" + nombre).round(3)
    num_errores = np.shape(errores)[0]
    nombre_archivo = nombre[: nombre.find(".")]
    nombres = [
        "Reproduction",
        "Linear",
        "Compound",
        "Logarithm",
        "Polynomial",
        "Neural Network",
    ]

    plt.figure(figsize=(12, 8))
    for i in range(num_errores):
        plt.plot(range(1, 25), errores[i])
    plt.xlabel("patch number", fontsize=20)
    plt.ylabel("$\Delta$E", fontsize=20)
    plt.legend(nombres, fontsize=12)
    plt.savefig(
        "Resultados/Imagenes/grafica_error_Nim" + nombre_archivo[-2:] + ".pdf",
        format="pdf",
    )
    plt.show()

    plt.figure(figsize=(12, 8))
    for i in [0, num_errores - 2, num_errores - 1]:
        plt.plot(range(1, 25), errores[i])
    plt.xlabel("patch number", fontsize=20)
    plt.ylabel("$\Delta$E", fontsize=20)
    plt.legend((nombres[0],nombres[-2],nombres[-1]), fontsize=15)
    plt.savefig(
        "Resultados/Imagenes/grafica_error2_Nim" + nombre_archivo[-2:] + ".pdf",
        format="pdf",
    )
    plt.show()

def generate_graphics_sep(folder, nombre):
    errores = fun.Read_Variable(folder + "/" + nombre).round(3)
    nombres_metodos = [
        "Reproduction",
        "Linear",
        "Compound",
        "Logarithm",
        "Polynomial",
        "Neural Network",
    ]
    for count, metodo in enumerate(nombres_metodos):
        nombre_archivo = nombre[: nombre.find(".")] + '_' + metodo
        plt.figure(figsize=(12, 8))
        for i in [0, count]:
            plt.plot(range(1, 25), errores[i])
        maxi = np.max(errores[count])
        mini = np.min(errores[count])
        plt.plot(range(1, 25), np.ones(np.size(errores[count]))*maxi, color='black', linestyle = '--')
        plt.plot(range(1, 25), np.ones(np.size(errores[count]))*mini, color='black', linestyle = '--')

        plt.xlabel("patch number", fontsize=24)
        plt.ylabel("$\Delta$E", fontsize=24)
        plt.legend((nombres_metodos[0], metodo), fontsize=20)
        plt.savefig(
            "Resultados/Imagenes/grafica_error_Nim" + nombre_archivo + ".pdf",
            format="pdf",
        )

for nombre in lista:
    if nombre[:4] == "CCM_":
        generate_ccm(carpeta, nombre)

    #%% Generacion de tablas de errores y graficas

    if nombre[:11] == "errores_de_":
        generate_table(carpeta, nombre)

        generate_table_sep(carpeta, nombre)

        generate_median(carpeta, nombre)

        generate_graphics(carpeta, nombre)

        generate_graphics_sep(carpeta, nombre)
