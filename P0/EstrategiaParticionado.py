from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np


class Particion():

    # This class keeps the list of Train and Test indexes for each
    # partition
    def __init__(self):
        self.indicesTrain = []
        self.indicesTest = []

###############################################################################


class EstrategiaParticionado:

    # Abstract Class
    __metaclass__ = ABCMeta

    # Atributes: must be filled properly for each concrete strategy.
    # They are passed in the constructor
    def __init__(self):
        self.particiones = []

    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada estrategia concreta
    def creaParticiones(self, datos: pd.DataFrame, seed=None):
        pass


###############################################################################

class ValidacionSimple(EstrategiaParticionado):

    def __init__(self, numeroEjecuciones: int, proporcionTest: float):
        super().__init__()
        self.numeroEjecuciones = numeroEjecuciones
        self.proporcionTest = proporcionTest

    # Crea particiones segun el metodo tradicional de division de los datos
    # según el porcentaje deseado y el n�mero de ejecuciones deseado
    # Devuelve una lista de particiones (clase Particion)
    # TODO: implementar
    def creaParticiones(self, datos: pd.DataFrame, seed=None):

        # get the number of rows
        num_rows = datos.shape[0]

        for _ in range(self.numeroEjecuciones):

            # create a new partition
            particion = Particion()

            # shuffle dataset
            datos = datos.sample(frac=1, random_state=seed)

            # get the number of rows for the test set
            num_rows_test = int(num_rows * self.proporcionTest)

            # get the test indexes
            particion.indicesTest = datos.iloc[:num_rows_test].index.tolist()

            # get the train indexes
            particion.indicesTrain = datos.iloc[num_rows_test:].index.tolist()

            # add the partition to the list of partitions
            self.particiones.append(particion)


#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):

    def __init__(self, numeroParticiones: int):
        super().__init__()
        self.numeroParticiones = numeroParticiones

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el
    # de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    # TODO: implementar
    def creaParticiones(self, datos: pd.DataFrame, seed=None):

        # shuffle dataset
        datos = datos.sample(frac=1, random_state=seed)

        # divide dataset into k folds
        folds = np.array_split(datos, self.numeroParticiones)

        for i in range(self.numeroParticiones):

            # create a new partition
            particion = Particion()

            # get train indexes (all folds except the test fold)
            particion.indicesTrain = pd.concat(
                folds[:i] + folds[i+1:]).index.tolist()

            #  get test indexes
            particion.indicesTest = folds[i].index.tolist()

            # add the partition to the list of partitions
            self.particiones.append(particion)
