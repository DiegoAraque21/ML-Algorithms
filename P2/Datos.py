# -*- coding: utf-8 -*-

# coding: utf-8
import pandas as pd


class Datos:

    # Constructor: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero):

        # read file into a numpy array
        df = pd.read_csv(nombreFichero)

        # get column names
        column_names = df.columns.values.tolist()

        # change dtype of Class column to string
        df.iloc[:, -1] = df.iloc[:, -1].astype(str)

        # get dtypes of each column
        dtypes = df.dtypes.to_list()

        # convert dtypes to boolean
        self.nominalAtributos = [True if x == 'object' else False if x == 'int64' or x ==
                                 'float64' else ValueError("Tipo de dato no soportado") for x in dtypes]

        # create dictionary for nominal values
        self.diccionarios = {}

        for i in range(len(self.nominalAtributos)):
            name = column_names[i]
            self.diccionarios[name] = {}

            if self.nominalAtributos[i] == True:

                # get unique values of each column
                uvalues = df.iloc[:, i].unique()

                # lexicographical order
                uvalues.sort()

                # assign a number to each nominal value
                for v in range(len(uvalues)):
                    self.diccionarios[name][uvalues[v]] = v

        # change nominal values to numbers
        for i in range(len(self.nominalAtributos)):
            name = column_names[i]
            if self.nominalAtributos[i] == True:

                # map each nominal value to its corresponding number
                df.iloc[:, i] = df.iloc[:, i].map(self.diccionarios[name])

        # set self.datos to the mapped dataframe
        self.datos = df

        # update each column dtype to int
        for i in range(len(self.nominalAtributos)):
            name = column_names[i]
            if self.nominalAtributos[i] == True:
                self.datos[name] = self.datos[name].astype(int)


    # Devuelve el subconjunto de los datos cuyos índices se pasan como argumento
    def extraeDatos(self, idx):
        return self.datos.iloc[idx, :]
