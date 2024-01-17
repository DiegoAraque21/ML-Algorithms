from Clasificador import Clasificador
import numpy as np
import pandas as pd
from scipy.stats import norm


class ClasificadorKNN(Clasificador):

    def __init__(self, k=1, normalize=True) -> None:
        super().__init__()
        self.k = k
        self.normalize = normalize
        self.predictions = None

    def _calcularMediasDesv(self, datos: np.ndarray, nominalAtributos: list) -> None:

        self.means = np.empty(len(nominalAtributos) - 1)
        self.stds = np.empty(len(nominalAtributos) - 1)

        for i in range(len(nominalAtributos) - 1):

            if nominalAtributos[i] == False:

                # calculate attribute mean
                self.means[i] = np.mean(datos[:, i])

                # calculate attribute standard deviation
                self.stds[i] = np.std(datos[:, i])

    def _normalizarDatos(self, datos: np.ndarray, nominalAtributos: list) -> None:

        for i in range(len(nominalAtributos) - 1):

            if nominalAtributos[i] == False:
                # normalize data by subtracting the mean and dividing by the standard deviation
                datos[:, i] = (datos[:, i] - self.means[i]) / self.stds[i]

    def entrenamiento(self, datosTrain: pd.DataFrame, nominalAtributos: list, diccionario: dict) -> None:

        self.traindata = datosTrain.to_numpy()

        if self.normalize == True:

            # calculate means and standard deviations
            self._calcularMediasDesv(self.traindata, nominalAtributos)

            # normalize continuous attributes
            self._normalizarDatos(self.traindata, nominalAtributos)

    def clasifica(self, datosTest: pd.DataFrame, nominalAtributos: list, diccionario: dict) -> np.ndarray:

        n = datosTest.shape[0]
        testdata = datosTest.to_numpy()

        # initialize predictions array
        predictions = np.empty(n, dtype=int)

        if self.normalize == True:
            # normalize continuous attributes
            self._normalizarDatos(testdata, nominalAtributos)

        for i in range(n):

            # calculate distances between test instance and all training instances
            distances = np.sqrt(
                np.sum((self.traindata[:, :-1] - testdata[i, :-1])**2, axis=1))

            # get indices of the k nearest neighbors
            k_nearest = np.argsort(distances)[:self.k]

            # get classes of the k nearest neighbors
            classes = self.traindata[k_nearest, -1]

            # get the most frequent class
            predictions[i] = np.bincount(classes.astype(int)).argmax()

        self.predictions = predictions
        return predictions

    def confusion_matrix(self, classes) -> np.ndarray:
        # predictions
        predictions = self.predictions

        # number of classes
        nclasses = len(classes)

        # initialize confusion matrix
        confusion_matrix = np.zeros((nclasses, nclasses), dtype=int)

        # get test data without header
        testdata = self.testdata[:, :-1]

        for i in range(len(predictions)):
            confusion_matrix[predictions[i], testdata[i, -1]] += 1

        return confusion_matrix
