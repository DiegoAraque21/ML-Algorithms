from abc import ABCMeta, abstractmethod

from Datos import Datos
from EstrategiaParticionado import EstrategiaParticionado

import numpy as np

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
            traindata = dataset.extraeDatos(particionado.particiones[i].indicesTrain)

            # get test data
            testdata = dataset.extraeDatos(particionado.particiones[i].indicesTest)

            # train the classifier
            clasificador.entrenamiento(traindata, dataset.nominalAtributos, dataset.diccionarios)

            # get predictions
            predictions = clasificador.clasifica(testdata, dataset.nominalAtributos, dataset.diccionarios)
            
            # get error
            errors[i] = self.error(testdata, predictions)

        return errors

