from Clasificador import Clasificador
import numpy as np
import pandas as pd
import math


class ClasificadorRegresionLogistica(Clasificador):

    def __init__(self, l_rate=0.5, n_epochs=12, normalize=False) -> None:
        super().__init__()
        self.n_epochs = n_epochs
        self.l_rate = l_rate
        self.normalize = normalize

    def _sigmoid(self, x):

        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0

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

    def entrenamiento(self, datosTrain, nominalAtributos, diccionario):

        # get data in numpy array
        self.traindata = datosTrain.to_numpy()

        if self.normalize == True:

            # calculate means and standard deviations
            self._calcularMediasDesv(self.traindata, nominalAtributos)

            # normalize continuous attributes
            self._normalizarDatos(self.traindata, nominalAtributos)

        # initialize weights vector
        self.weights = np.random.uniform(-0.5, 0.6, len(nominalAtributos) - 1)

        for _ in range(self.n_epochs):

            # optimized version

            products = np.dot(self.traindata[:, :-1], self.weights)

            scores = np.array([self._sigmoid(x) for x in products])

            errors = scores - self.traindata[:, -1]

            self.weights = self.weights - \
                self.l_rate * np.dot(errors, self.traindata[:, :-1])

            # nested loop version
            #for row in self.traindata:

            #    product = self.weights.dot(row[:-1])

            #    score = self._sigmoid(product)

            #    error = score - row[-1]

            #    self.weights = self.weights - self.l_rate * error * row[:-1]

    def clasifica(self, datosTest: pd.DataFrame, nominalAtributos: list, diccionario: dict):

        # get data in numpy array
        self.testdata = datosTest.to_numpy()
        self.classes = self.testdata[:, -1].astype(int)

        # initialize predictions array
        self.predictions = np.empty(len(self.testdata), dtype=int)

        # initialize scores array
        self.scores = np.empty(len(self.testdata))

        if self.normalize == True:
            # normalize continuous attributes
            self._normalizarDatos(self.testdata, nominalAtributos)

        # optimized version

        self.scores = np.array([self._sigmoid(x) for x in np.dot(self.testdata[:, :-1], self.weights)])
        
        self.predictions = np.where(self.scores >= 0.5, 1, 0)

        # nested loop version
        
        #for i in range(len(self.testdata)):
#
        #    product = self.weights.dot(self.testdata[i][:-1])
#
        #    score = self._sigmoid(product)
#
        #    self.scores[i] = score
#
        #    if score >= 0.5:
        #        self.predictions[i] = 1
        #    else:
        #        self.predictions[i] = 0

        return self.predictions

    def confusion_matrix(self):
        predictions = self.predictions

        # initialize confusion matrix, 2x2 because there are only 2 classes
        conf_matrix = np.zeros((2, 2), dtype=int)

        # fill confusion matrix
        for pred, real in zip(predictions, self.classes):
            conf_matrix[real, pred] += 1

        return conf_matrix
    

    def roc_curve(self):

        # Fawcett's algorithm

        # get confusion matrix
        conf_matrix = self.confusion_matrix()

        # sort scores in descending order
        sorted_scores = np.sort(self.scores)[::-1]

        # initialize fp and tp values
        fp = 0
        tp = 0

        # get N and P values
        N = conf_matrix[0].sum()
        P = conf_matrix[1].sum()

        # initialize roc curve points
        roc_curve = []

        # initialize previous score
        prev_score = -1

        # initialize index
        i = 0

        # iterate over sorted scores
        for score in sorted_scores:

            # if score is different from previous score
            if score != prev_score:

                # add point to roc curve
                roc_curve.append((fp/N, tp/P))

                # update previous score
                prev_score = score

            # if score is positive
            if score >= 0.5:

                # update tp
                tp += 1

            # if score is negative
            else:

                # update fp
                fp += 1

            i += 1

        # add last point to roc curve
        roc_curve.append((fp/N, tp/P))

        return roc_curve
    
