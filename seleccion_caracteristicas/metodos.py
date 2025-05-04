from itertools import combinations
import numpy as np


def FDR(X, y, k):
    num_features = X.shape[0]
    scores = []

    for i in range(num_features):
        xi = X[i]
        classes = np.unique(y).tolist()

        m = [np.mean(xi[y == c]) for c in classes]
        v = [np.var(xi[y == c], ddof=1) for c in classes]

        combs = list(combinations(range(len(classes)), 2))
        q = [(m[j] - m[l]) ** 2 / (v[j] + v[l]) for j, l in combs]

        scores.append((i, np.sum(q)))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return [(index, round(float(score), 3)) for index, score in scores[:k]]


def correlacion_cruzada(X, y, k):
    res = []

    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        res.append((i, abs(corr)))

    res.sort(key=lambda x: x[1], reverse=True)
    return [(index, round(float(score), 3)) for index, score in res[:k]]


def correlacion_pearson(X, threshold=0.5):
    corr_mat = np.corrcoef(X, rowvar=False)
    n = corr_mat.shape[0]
    res = []

    for i in range(n):
        demasiado = False
        for j in res:
            if abs(corr_mat[i, j]) > threshold:
                demasiado = True
                break
        if not demasiado:
            res.append(i)

    return res
