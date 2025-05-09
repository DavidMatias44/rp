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
        

def obtener_columna_etiquetas(df):
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
    
    print(f"\n\t[Info]: la columna '[{label_col}]: {df.columns[label_col]}' ha sido asignada a los targets.\n")
    return label_col
