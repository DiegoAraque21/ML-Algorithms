from Clasificador import Clasificador
import numpy as np
import pandas as pd

class AlgoritmoGenetico(Clasificador):

    def __init__(self, sample_size, n_generations, max_rules):
        self.sample_size = sample_size
        self.n_generations = n_generations
        self.max_rules = max_rules
        self.best_individuals = []
        self.mean_generations = []

    def entrenamiento(self, datosTrain, nominalAtributos, diccionario):

        self.traindata = datosTrain.to_numpy()
        
        # create set of rules
        self._createRules(diccionario)

        # elitism_size, cross-over_size, mutation_size
        elitism_size = 5*self.sample_size//100
        cross_over_size = 90*self.sample_size//100
        mutation_size = 5*self.sample_size//100

        # adjust cross-over size to be even
        if elitism_size + cross_over_size + mutation_size != self.sample_size:
            mutation_size = self.sample_size - elitism_size - cross_over_size

        # for each generation
        for i in range(self.n_generations):
            # check rules
            self._check_rule()

            # calculate fitness
            self._calculate_fitness(diccionario)

            # elitism
            next_generation = {}
            
            # get elite individuals
            elite_individuals = [(k, v) for k, v in sorted(self.fitness.items(), key=lambda item: item[1], reverse=True)]
            elite_index = [x[0] for x in elite_individuals]

            # 5% of the best individuals are copied to the next generation
            for j in range(elitism_size):
                next_generation[j] = self.rules[elite_index[j]]

            # intra-rule crossover
            # 90% of the next generation is generated by crossover
            for idx in range(0, cross_over_size, 2):
                parents = self._roulette_wheel_selection(2)
                descendants = self._intra_rule_crossover(parents)

                next_generation[idx+elitism_size] = descendants[0]
                next_generation[idx+elitism_size+1] = descendants[1]

            # mutation to descendants (given by a percentage)
            mutation_indices = self._roulette_wheel_selection(amount=mutation_size)
            mutated = self._mutate(mutation_indices)

            for idx2 in range(mutation_size):
                next_generation[idx2+elitism_size+cross_over_size] = mutated[idx2]

            # update rules
            self.rules = next_generation

            self.best_individuals.append(self.fitness[elite_index[0]])

            self.mean_generations.append(sum(self.fitness.values())/len(self.fitness.values()))

            print(f'\rGeneration {i+1}  -->  Best fitness: {self.fitness[elite_index[0]]}\tMean fitness: {(sum(self.fitness.values())/len(self.fitness.values()))}', end=' ', flush=True)


        # check rules
        self._check_rule()

        # calculate fitness
        self._calculate_fitness(diccionario)

        # get best individual
        self.best_individual = max(self.fitness.items(), key=lambda item: item[1])

        # get best rules
        self.best_rules = self.rules[self.best_individual[0]]

        print(f"\nThe best individual is {self.best_individual[0]} with a fitness of {self.best_individual[1]}\nThe best set of rules are {self.best_rules}", end='\n\n', flush=True)


    def clasifica(self, datosTest, nominalAtributos, diccionario):

        n = datosTest.shape[0]

        # get test data
        self.testdata = datosTest.to_numpy(dtype=int)

        # initialize predictions
        self.predictions = np.empty(n, dtype=int)

        # for each row in the test data
        for i in range(n):

            r_pred = np.array([], dtype=int)

            # for each rule in the best individual
            for rule in self.best_rules:

                # check if rule is activated
                failed = 0
                for index, value in enumerate(self.testdata[i]):
                    
                    # if it is not the class column
                    if index != len(self.testdata[i])-1:

                        # get starting bit position
                        bitindex = index
                        prev = index - 1

                        # adjust bit offset
                        if prev >= 0:
                            bitindex = self.sizes[prev] + value
                        elif prev < 0:
                            bitindex = index + value

                        # check if bit is activated, once it fails, stop checking
                        if rule[bitindex] == "0":
                            failed = 1
                            break

                # if rule is activated, add class to prediction
                if failed == 0:
                    r_pred = np.append(r_pred, int(rule[-1]))

            # if there are no rules activated, add random class to prediction
            if len(r_pred) == 0:
                r_pred = np.append(r_pred, np.random.randint(0, 2))

            # get most common class in prediction
            self.predictions[i] = np.bincount(r_pred).argmax()

        return self.predictions
            

    def _intra_rule_crossover(self, parents):

        rules_p1 = self.rules[parents[0]].copy()
        rules_p2 = self.rules[parents[1]].copy()

        # get random rule of each parent
        rule_p1 = np.random.choice(rules_p1)
        rule_p2 = np.random.choice(rules_p2)

        # get index of each rule
        index_p1 = rules_p1.index(rule_p1)
        index_p2 = rules_p2.index(rule_p2)

        # get crossing point
        crossing_point = np.random.randint(1, self.nbits)

        #  gemerate btoh descendants
        desc1_rule = rule_p1[:crossing_point] + rule_p2[crossing_point:]
        desc2_rule = rule_p2[:crossing_point] + rule_p1[crossing_point:]

        # replace rules
        desc1 = self.rules[parents[0]].copy()
        desc2 = self.rules[parents[1]].copy()

        desc1[index_p1] = desc1_rule
        desc2[index_p2] = desc2_rule

        return desc1, desc2


    def _mutate(self, individuals):

        cp_rules = []

        # for each individual
        for i in individuals:

            cp_list = self.rules[i].copy()

            for rule in cp_list:

                for bit in range(len(rule)):

                    # flip bit fifty-fifty
                    if np.random.randint(0, 2) == 1:

                        # flip bit
                        if rule[bit] == "0":
                            rule = rule[:bit] + "1" + rule[bit+1:]
                        elif rule[bit] == "1":
                            rule = rule[:bit] + "0" + rule[bit+1:]

            cp_rules.append(cp_list)

        return cp_rules

    def _calculate_fitness(self, diccionario):

        # initialize fitness dictionary
        self.fitness = {}

        # for each individual
        for i in range(self.sample_size):

            count = 0

            for j in range(self.traindata.shape[0]):

                r_pred = np.array([], dtype=int)

                # for each rule in the individual
                for rule in self.rules[i]:

                    # check if rule is activated
                    failed = 0
                    for index, value in enumerate(self.traindata[j]):
                        # if it is not the class column
                        if index != len(self.traindata[j])-1:

                            # get starting bit position
                            bitindex = index
                            prev = index - 1

                            # adjust bit offset
                            if prev >= 0:
                                bitindex = self.sizes[prev] + value
                            elif prev < 0:
                                bitindex = index + value

                            # check if bit is activated, once it fails, stop checking
                            if rule[bitindex] == "0":
                                failed = 1
                                break

                    # if rule is activated, add class to prediction
                    if failed == 0:
                        r_pred = np.append(r_pred, int(rule[-1]))

                # if there are no rules activated, ignore
                if len(r_pred) != 0:

                    # get most common class in prediction
                    r_pred = np.bincount(r_pred).argmax()

                    # if prediction is correct, increase count
                    if r_pred == self.traindata[j][-1]:
                        count += 1

            # calculate fitness
            self.fitness[i] = count/self.traindata.shape[0]

        # order fitness dictionary
        self.fitness = {k: v for k, v in sorted(self.fitness.items(), key=lambda item: item[1], reverse=True)}

    def _roulette_wheel_selection(self, amount=1):
        
        # self fitnets total
        f = sum(list(self.fitness.values()))
        roulette_probs = {}

        for fi in self.fitness.keys():
            roulette_probs[fi] = self.fitness[fi]/f

        values_ind = list(self.rules.keys())
        values_probs = list(roulette_probs.values())

        selection = np.random.choice(values_ind, size=amount, p=values_probs, replace=True)

        return selection


    def _createRules(self, diccionario):

        # initialize dictionary of rules
        self.rules = {}

        # get total number of bits in the rules
        self.sizes = {}

        # get size of each column
        for i, key in enumerate(diccionario.keys()):
            self.sizes[i] = len(diccionario[key])

        self.nbits = sum(self.sizes.values()) - 1

        # for each individual
        for i in range(self.sample_size):

            # initialize list of rules for an individual
            self.rules[i] = list()

            # generate random number of rules for an individual
            m = np.random.randint(1, self.max_rules + 1)

            # generate random rules for an individual
            for _ in range(m):
                rule_array = np.random.randint(0, 2, self.nbits)
                rule = "".join(str(x) for x in rule_array)
                self.rules[i].append(rule)


    def _check_rule(self):

        # generate rule full of zeros
        rule_array = np.zeros(shape=self.nbits, dtype=int)
        zeros = "".join(str(x) for x in rule_array)

        # generate rule full of ones
        rule_array = np.ones(shape=self.nbits, dtype=int)
        ones = "".join(str(x) for x in rule_array)
        
        # check each rule
        for k in self.rules.keys():
            for rule in self.rules[k]:

                # if rule is full of 0s or 1s, remove it
                if rule == zeros or rule == ones:
                    self.rules[k].remove(rule)
            
            # if there are no rules left, generate a new one
            if len(self.rules[k]) == 0:
                rule_array = np.random.randint(0, 2, self.nbits)
                rule = "".join(str(x) for x in rule_array)
                self.rules[k].append(rule)
