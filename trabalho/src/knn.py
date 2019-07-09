# Realização dos imports das bibliotecas
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from math import log

contKnn = 0
coutEuclidean = 0

def class_histogram(kn_neighbors, n_classes):
    '''Faz a contagem das classes dos k vizinhos mais próximos.
    Recebe o vetor com os k mais próximos.'''
    
    c = [0] * n_classes
    for i in kn_neighbors:
        c[i] += 1
    return c

def knn(X_train, Y_train, x, k):
    '''Faz a classificação do exemplo x baseado nos k mais próximos em X_train.'''
    global coutEuclidean
    global contKnn

    d = euclidean_distances(x.reshape(1, -1), X_train).reshape(-1)
    
    idx = np.argsort(d)
    hist = class_histogram(Y_train[idx][:k], len(set(Y_train)))
    
    coutEuclidean = coutEuclidean + len(X_train)
    contKnn = contKnn + 1

    return np.argmax(hist)


X, Y = load_digits(return_X_y=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

misses, hits = 0, 0
k = 3

for i, x in enumerate(X_test):

    if knn(X_train, Y_train, x, k) == Y_test[i]:
        hits += 1
    else:
        misses += 1


nlogn = (contKnn * (len(X_train) * log(len(X_train), 2)))

#print n lg n
# print ("n lg n: %d" % nlogn)

#print contador knn
# print("Euclidean: %d" % coutEuclidean)

#total
print("Total: %d" % (nlogn + coutEuclidean + (2*len(X))))

# A acurácia é dada por acertos / (acertos + erros).
print ("Acurácia: %.02f%%" % (hits / float(hits + misses) * 100))