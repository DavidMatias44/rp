from tkinter import Tk, filedialog
import pandas as pd


def cargar_dataset():
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Selecciona el dataset",
        filetypes=[
            ("Archivos CSV", "*.csv"),
            ("Archivos Excel", "*.xlsx")
        ]
    )

    if file_path:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        
        print(f"\n\t[Info]: Dataset cargado: {file_path}.\n")
    else:
        print("\n\t[Error]: No seleccionó un archivo.\n")
        exit(-1)
    
    return df
        

def mostrar_resultados(result, labels):
    if isinstance(result[0], tuple):
        print(f"\n\tMejores características (caracteristica, score): {result}.")
        for i, _ in result:
            print(f"\t\t[{i}]: {labels[i]}")
    else:
        print(f"\n\tMejores características {result}.")
        for i in result:
            print(f"\t\t[{i}]: {labels[i]}")


def obtener_param(l):
    while True:
        try:
            k = int(input(f"Ingrese el número de características a seleccionar (k < {l}): "))
            if k < 0 or k >= l:
                print(f"\n\t[Error]: k debe ser menor a {l} y mayor a 0.\n")
                continue
            break
        except ValueError:
            print("\n\t[Error]: no ha ingresado un valor válido.")
    return k

