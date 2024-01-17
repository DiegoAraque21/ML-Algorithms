from abc import ABCMeta, abstractmethod

from Datos import Datos
from EstrategiaParticionado import EstrategiaParticionado

import numpy as np
import pandas as pd
from scipy.stats import norm


class Clasificador:

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    # TODO: esta funcion debe ser implementada en cada clasificador concreto. Crea el modelo a partir de los datos de entrenamiento
    # datosTrain: matriz numpy o dataframe con los datos de entrenamiento
    # nominalAtributos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def entrenamiento(self, datosTrain, nominalAtributos, diccionario):
        pass

    @abstractmethod
    # TODO: esta funcion debe ser implementada en cada clasificador concreto. Devuelve un numpy array con las predicciones
    # datosTest: matriz numpy o dataframe con los datos de validaci�n
    # nominalAtributos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    # devuelve un numpy array o vector con las predicciones (clase estimada para cada fila de test)
    def clasifica(self, datosTest, nominalAtributos, diccionario):
        pass

    # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
    # TODO: implementar
    def error(self, datos, pred) -> float:
        # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
        # devuelve el error

        # get the number of rows
        n = datos.shape[0]

        # get the number of correct predictions
        correct = datos.iloc[:, -1].eq(pred).sum()

        # calculate the error
        error = 1 - (correct / n)

        return error

    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    # TODO: implementar esta funcion
    def validacion(self, particionado: EstrategiaParticionado, dataset: Datos, clasificador: 'Clasificador', seed=None) -> np.ndarray:
        """ Realiza una clasificacion utilizando una estrategia de particionado determinada

        Args:
            particionado (EstrategiaParticionado): estrategia de particionado a utilizar
            dataset (Datos): dataset encapsulado en un objeto Datos
            clasificador (Clasificador): clasificador a utilizar
            seed (_type_, optional): semilla para la generacion de numeros aleatorios. Defaults to None.

        Returns:
            np.ndarray: numpy array con los errores de cada particion
        """

        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.
        # devuelve el vector con los errores por cada partici�n

        # pasos
        # crear particiones
        # inicializar vector de errores
        # for cada partici�n
        #     obtener datos de train
        #     obtener datos de test
        #     entrenar sobre los datos de train
        #     obtener prediciones de los datos de test (llamando a clasifica)
        #     a�adir error de la partici�n al vector de errores

        # create partitions
        particionado.creaParticiones(dataset.datos, seed)

        # initialize errors vector
        errors = np.empty(len(particionado.particiones))

        # for each partition
        for i in range(len(particionado.particiones)):
            # get train data
            traindata = dataset.extraeDatos(
                particionado.particiones[i].indicesTrain)

            # get test data
            testdata = dataset.extraeDatos(
                particionado.particiones[i].indicesTest)

            # train the classifier
            clasificador.entrenamiento(
                traindata, dataset.nominalAtributos, dataset.diccionarios)

            # get predictions
            predictions = clasificador.clasifica(
                testdata, dataset.nominalAtributos, dataset.diccionarios)

            # get error
            errors[i] = self.error(testdata, predictions)

        return errors


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

        return predictions
