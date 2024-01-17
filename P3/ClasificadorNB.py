from Clasificador import Clasificador
import numpy as np
import pandas as pd
from scipy.stats import norm


class ClasificadorNaiveBayes(Clasificador):

    def __init__(self, laplace=False):
        """ Constructor del clasificador Naive Bayes

        Args:
            laplace (bool, optional): indica si se aplica o no la corrección de Laplace. Defaults to False.
        """
        self.prioris = {}
        self.laplace = laplace

    # TODO: implementar
    def entrenamiento(self, datostrain: pd.DataFrame, nominalAtributos: list, diccionario: dict):
        """ Entrena el clasificador Naive Bayes con los datos de entrenamiento

        Args:
            datostrain (pd.DataFrame): matriz numpy o dataframe con los datos de entrenamiento
            nominalAtributos (list): array bool con la indicatriz de los atributos nominales
            diccionario (dict): array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
        """

        # number of rows
        n = datostrain.shape[0]

        # class column
        classcol = datostrain.iloc[:, -1]

        # column names
        colnames = datostrain.columns

        # number of classes
        nclasses = len(classcol.unique())

        # calculate priori probabilities for each class in a dictionary
        self.prioris = (classcol.value_counts() / n).sort_index().to_dict()

        # initialize likelihoods array
        self.likelihoods = np.empty(
            (len(nominalAtributos) - 1, nclasses), dtype=object)

        # calculate likelihoods for each attribute value in a dictionary
        for i in range(len(nominalAtributos) - 1):

            # if the attribute is nominal
            if nominalAtributos[i] == True:

                # for each class
                for c in range(nclasses):

                    # for each value of the attribute
                    for v in range(len(diccionario[colnames[i]])):

                        # get the number of times the value appears in the class
                        num = datostrain.loc[(
                            datostrain.iloc[:, -1] == c) & (datostrain.iloc[:, i] == v), colnames[i]].count()

                        # calculate the likelihood
                        self.likelihoods[i, c] = num / \
                            classcol.value_counts()[c]

                        if self.laplace == True:
                            self.likelihoods[i, c] = (
                                num + 1) / (classcol.value_counts()[c] + len(diccionario[colnames[i]]))

            # if the attribute is numeric
            elif nominalAtributos[i] == False:

                # calculate mean and std and apply gaussian formula to get likelihoods
                for c in range(nclasses):

                    values_ic = datostrain.loc[datostrain.iloc[:, -1]
                                               == c, colnames[i]]
                    self.likelihoods[i, c] = (
                        values_ic.mean(), values_ic.std())

    # TODO: implementar
    def clasifica(self, datostest: pd.DataFrame, nominalAtributos: list, diccionario: dict) -> np.ndarray:
        """ Clasifica los datos de test con el clasificador Naive Bayes

        Args:
            datostest (pd.DataFrame): matriz numpy o dataframe con los datos de test
            nominalAtributos (list): array bool con la indicatriz de los atributos nominales
            diccionario (dict): array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas

        Returns:
            np.ndarray: numpy array con las predicciones (clase estimada para cada fila de test)
        """

        # number of rows
        n = datostest.shape[0]

        # get classes from test data
        self.testdata = datostest.to_numpy()
        self.classes = self.testdata[:, -1].astype(int)

        # number of classes
        nclasses = len(self.prioris)

        # initialize posterioris array
        posterioris = np.empty(
            (len(nominalAtributos) - 1, nclasses), dtype=object)

        # initialize predictions array
        predictions = np.empty(n, dtype=int)

        # get test data without header
        testdata = datostest.iloc[:, :-1].values

        for dat, row in zip(testdata, range(n)):
            for i in range(len(nominalAtributos) - 1):
                if nominalAtributos[i] == True:
                    for c in range(nclasses):
                        posterioris[i, c] = self.likelihoods[i,
                                                             c] * self.prioris[c]
                elif nominalAtributos[i] == False:
                    for c in range(nclasses):
                        posterioris[i, c] = norm.pdf(
                            dat[i], self.likelihoods[i, c][0], self.likelihoods[i, c][1]) * self.prioris[c]

            predictions[row] = np.argmax(np.prod(posterioris, axis=0))

        self.predictions = predictions
        return predictions

    def confusion_matrix(self) -> np.ndarray:
        # predictions
        predictions = self.predictions

        # initialize confusion matrix, 2x2 because there are only 2 classes
        conf_matrix = np.zeros((2, 2), dtype=int)

        # fill confusion matrix
        for pred, real in zip(predictions, self.classes):
            conf_matrix[real, pred] += 1

        return conf_matrix
