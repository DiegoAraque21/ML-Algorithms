import numpy as np
import random as rd

class K_Means():

    def __init__(self, k=3, normalize=False):
        self.n_clusters = k
        self.centroids = {}
        self.curr_clusters = {}
        self.prev_clusters = {}
        self.curr_error = 0
        self.prev_error = 0
        self.predicted = {}
        self.normalize = normalize


    def _euclideanDistanceCalc(self, data1, data2):
        return np.sqrt(np.sum(np.square(data1 - data2)))
        
    def _assignPointsToCentroids(self, data):
        # iterate over data points
        for i in range(len(data)):

            # set min distance to a large number
            min_dist = 9999999999999999999

            corr_j = None
            corr_dist = None
            # iterate over centroids
            for j in range(len(self.centroids)):
                # calculate distance between data point and centroid
                dist = self._euclideanDistanceCalc(data[i], self.centroids[j])
                # if distance is smaller than min distance
                if dist < min_dist:
                    # set min distance to distance
                    min_dist = dist
                    # set cluster to centroid
                    corr_j = j
                    corr_dist = dist
                
            self.curr_clusters[corr_j].append((i, corr_dist, data[i]))
        
    def _calculateNewCentroids(self):
        for i in range(len(self.curr_clusters)):
            # array of distances for each point in cluster
            points = [self.curr_clusters[i][j][2] for j in range(len(self.curr_clusters[i]))]
            # calculate mean of data points
            self.centroids[i] = np.mean(points, axis=0)
            self.prev_clusters[i] = self.curr_clusters[i]
            self.curr_clusters[i] = []

    def _normalize(self, data):
        for i in range(len(data.columns) - 1):
            data.iloc[:, i] = (data.iloc[:, i] - data.iloc[:, i].mean()) / data.iloc[:, i].std()


    def train(self, data, diccionario):

        np.random.seed(100)

        if self.normalize:
            self._normalize(data)

        # data classes
        classes = data.iloc[:, -1]
        # convert data to numpy array
        data = np.array(data.iloc[:, :-1])

        # get k random centroids
        for i in range(self.n_clusters):
            # # generate random index
            # rand = rd.randint(0, len(data)-1)
            # # set centroid to random data point
            # self.centroids[i] = data[rand]
            # self.curr_clusters[i] = []

            # set centroid to random data point
            self.centroids[i] = data[np.random.choice(data.shape[0], 1, replace=False)]
            self.curr_clusters[i] = []
        
        while True:
            # assign points to centroids
            self._assignPointsToCentroids(data)

            # calculate new centroids
            self._calculateNewCentroids()
            
            # set previous error to current error
            for i in range(len(self.prev_clusters)):
                # array of distances for each point in cluster
                distances = [self.prev_clusters[i][j][1] for j in range(len(self.prev_clusters[i]))]
                self.curr_error += sum(distances)
            
            # if error is the same as previous error
            if self.prev_error - self.curr_error == 0:
                break
            else:
                # set previous error to current error
                self.prev_error = self.curr_error
                # set current error to 0
                self.curr_error = 0

        for i in range(len(self.prev_clusters)):
            indices = []
            for j in range(len(self.prev_clusters[i])):
                indices.append(self.prev_clusters[i][j][0])
            
            counts  = np.bincount(classes[indices])
            if len(counts) == 1:
                self.predicted[i] = 0
            self.predicted[i] = np.argmax(counts)

        for i in range(len(self.predicted)):
            c = list(diccionario.keys())[-1]
            c_types = diccionario[c]
            for k in list(c_types.keys()):
                if c_types[k] == self.predicted[i]:
                    self.predicted[i] = k
                    break

        