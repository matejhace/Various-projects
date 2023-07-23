import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Klasifikator:

    def __init__(self, popravek=0.1):
        
        self.popravek = popravek
        self._b = 0.0  # y-intercept
        self._utez = None  # utež, ki bo pomnožena s vzorci
        self.napacni_vzorci = [] # za štetje napak

    # prileganje klasifikatorja glede na naše podatke, x so vzorci, y pa oznake teh
    def prilegaj(self, x: np.array, y: np.array, iteracije=10):
        self._b = 0.0
        self._utez = np.zeros(x.shape[1])
        self.napacni_vzorci = []

        for _ in range(iteracije):
            # štej napake
            napacno = 0
            for ix, iy in zip(x, y):
                # posodobi podatke
                posodobi = self.popravek * (iy - self.napovej(ix))
                # posodobi parametre
                self._b += posodobi
                self._utez += posodobi * ix
                napacno += int(posodobi != 0.0)

            self.napacni_vzorci.append(napacno)
            self.napaka_proc = ((y-self.napacni_vzorci[-1])/y)*100
    
    #vrne izhod neurona
    def izhod_neurona(self, x):
        return np.dot(x, self._utez) + self._b

    # vrne 1 če je izhod vzorca pozitiven oz. enak 0, in -1, če je negativen, x so značilke
    def napovej(self, x: np.array):
        return np.where(self.izhod_neurona(x) >= 0, 1, -1)


tocke = pd.read_csv('iris.data', header=None)
tocke.head()

# oznake
y0 = tocke.iloc[:, 4].values
# značilke
x0 = tocke.iloc[:, 0:3].values

x = x0[0:100, 0:2]
y = y0[0:100]

# združi liste, glede na tri različne tipe
x1 = np.concatenate((x0[0:50, 0:2], x0[100:150, 0:2]), axis=0)
y1 = np.concatenate((y0[0:50], y0[100:150]), axis=0)

# vzorci setosa
plt.scatter(x[:50, 0], x[:50, 1], color='yellow', marker='8', label='Setosa')
# vzorci versicolour
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='s', label='Versicolour')
# vzorci Virginica
plt.scatter(x1[50:100, 0], x1[50:100, 1], color='green', marker=11, label='Virginica')

# prikaži legendo
plt.xlabel("dolžina sepal")
plt.ylabel("dolžina petal")
plt.legend(loc='upper right')

plt.show()

from sklearn.model_selection import train_test_split

#binariziraj glede na vrsto rože
y = np.where(y == 'Iris-setosa', 1, -1)
#print(y)

x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

# razdeli podatke
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# nauči model, prvič
classifier = Klasifikator(popravek=0.01)
classifier.prilegaj(x_train, y_train)

# plotaj število napak za vsako iteracijo
plt.plot(range(1, len(classifier.napacni_vzorci) + 1),
         classifier.napacni_vzorci, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Napake')
plt.show()
print(classifier.napaka_proc)

from matplotlib.colors import ListedColormap


def premica(x, y):
    #resolucija za prostor med vrednostmi na grafu
    resolucija = 0.001

    #markerji za graf
    markers = ('o', 'x')
    #barve za lažje ločevanje na grafu
    cmap = ListedColormap(('red', 'blue'))

    #izberemo celotni razpon vrednosti in dodamo oz. odštejemo 0.5 zato, da dobimo vse vrednosti v graf
    x1_min, x1_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    x2_min, x2_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5

    #klasifikator probamo na teh podatkih
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolucija),
                           np.arange(x2_min, x2_max, resolucija))

    #vse točke zberemo znotraj enega lista oz. arraya
    Z = classifier.napovej(np.array([xx1.ravel(), xx2.ravel()]).T)
    #vrnemo obliko
    Z = Z.reshape(xx1.shape)

    #nariši premico
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #nariši vse točke
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=x[y == c1, 0],
                    y=x[y == c1, 1],
                    alpha=0.8,
                    c=cmap(idx),
                    marker=markers[idx],
                    label=c1)
    plt.show()


premica(x_test, y_test)

#---------------------------------------------------------------
# nauči model, drugic
# binarizacija glede na vrsto rože
y1 = np.where(y1 == 'Iris-setosa', 1, -1)

x1[:, 0] = (x1[:, 0] - x1[:, 0].mean()) / x1[:, 0].std()
x1[:, 1] = (x1[:, 1] - x1[:, 1].mean()) / x1[:, 1].std()

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.3, random_state=0)

classifier.prilegaj(x1_train, y1_train)

#število napak za vsako iteracijo
plt.plot(range(1, len(classifier.napacni_vzorci) + 1), classifier.napacni_vzorci, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Napake')
plt.show()
print(classifier.napaka_proc)

premica(x1_test, y1_test)