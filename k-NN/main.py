import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from utils import *


def main():
    df = cargar_dataset()
    label_col = obtener_columna_etiquetas(df)

    necesitan_encoding = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in necesitan_encoding:
        df[col] = LabelEncoder().fit_transform(df[col])

    y = df.iloc[:, label_col]
    X = df.drop(df.columns[label_col], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ks = range(1, 16)
    errores = []

    print("Calculando errores con diferentes valores de k: ")
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        error = 1 - accuracy_score(y_test, y_pred)
        errores.append(error)
        print(f"\t k: {k:2} => Error: {error:.4f}")

    best_k = ks[np.argmin(errores)]
    min_error = min(errores)
    print(f"\nMejor k: {best_k:2} con error: {min_error:.4f}")

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))
    print("\nPrecisión:", accuracy_score(y_test, y_pred))

    plt.figure(figsize=(10, 6))
    plt.plot(ks, errores, marker='o')
    plt.title('Error vs Valores de k')
    plt.xlabel('Valor de k')
    plt.ylabel('Error (1 - Accuracy)')
    plt.xticks(ks)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
