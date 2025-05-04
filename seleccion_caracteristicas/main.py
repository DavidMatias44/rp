'''
Utilizando un lenguaje y/o herramienta de programación realice el siguiente programa con un menú para selección
de características mediante los siguientes métodos:
    1. Radio discriminante de Fisher. Después de leer el conjunto de datos, solicitar el número de característica 
       k a seleccionar, donde k < l (l es el número total de características del dataset).
    2. Coeficiente de correlación cruzada. La selección del número k de características es similar al ejercicio 1.
    3. Coeficiente de correlación de Pearson. Aquí se seleccionarán k características de acuerdo a un umbral de 
    correlación (ej: 0.5).
'''

from sklearn.preprocessing import LabelEncoder
import numpy as np
from metodos import *
from utils import *


def main():
    df = cargar_dataset()

    print("Columnas presentes en el DataFrame: ")
    for i, col in enumerate(df.columns):
        print(f"\t[{i}]: {col}")

    while True:
        try:
            label_col = int(input("Número de columna correspondiente a las etiquetas: "))
            if label_col < 0 or label_col >= len(df.columns):
                print("\n\t[Error]: número de columna no válido.\n")
                continue
            break
        except ValueError:
            print("\n\t[Error]: ha ingresado un valor inválido.\n")
            continue

    y_ = df.iloc[:, label_col]
    X_ = df.drop(df.columns[label_col], axis=1)

    y = np.array(y_).T
    X = np.array(X_).T

    if y.dtype == 'object': y = LabelEncoder().fit_transform(y)
    
    l = X.shape[0]
    labels = X_.columns.tolist()

    while True:
        print("\n\n========== SELECCIÓN DE CARACTERÍSTICAS ==========")
        print("Métodos disponibles: ")
        print("\t[1]: Radio discriminante de Fisher.")
        print("\t[2]: Coeficiente de correlación cruzada.")
        print("\t[3]: Coeficiente de correlación de Pearson.")
        print("\t[4]: Salir.")

        try:
            option = int(input("Ingrese una opción: "))
            if option < 1 or option >= 5:
                print("\n\t[Error]: número de opción no válido.")
                continue
        except ValueError:
            print("\n\t[Error]: ha ingresado un valor inválido.")
            continue
        
        if option == 4: break
        if option not in [1, 2, 3]:
            print("\n\t[Error]: opción no válida.")
            continue
        if option == 1:
            k = obtener_param(l)
            res = FDR(X, y, k)
        elif option == 2:
            k = obtener_param(l)
            res = correlacion_cruzada(X.T, y.T, k)
        elif option == 3:
            res = correlacion_pearson(X.T)

        mostrar_resultados(res, labels)


if __name__ == "__main__":
    main()
