# Machine Learning Algorithms

On this repository we have the implementation from scratch of several algorithms and their performance. We implemented Naive Bayes, KNN, K-Means, Logistic Regression and a Genetic Algorithm. All of them were tested with different datasets, to check their accuracy and see which one did the classification job better. We also compare the results to the ones returned by different scikit-learn implementations, to see if htey have a correct behaviour or not.

### Naive Bayes

Naive Bayes is a probabilistic classifier based on Bayes' theorem with the assumption that features are independent given the class label.

### KNN

k-Nearest Neighbors is a simple and intuitive algorithm for classification and regression tasks. It works by finding the 'k' nearest data points in the feature space and making predictions based on the majority class or average value. The problem with this algorithm, is that we need to find the best value for k in the problem, because if not we could overfit or underfit in our final solution.

### K-MEANS

k-Means is an unsupervised clustering algorithm that partitions data into 'k' clusters based on similarity. It iteratively assigns data points to clusters and updates cluster centroids until convergence.

### Logistic Regression

Logistic Regression is a widely-used classification algorithm that models the probability of an instance belonging to a particular class. It's particularly effective for binary classification tasks and can be extended to handle multiple classes using techniques like one-vs-rest or one-vs-one. A neural network is based on the sigmoid functions used for this algorithm, but in a much bigger scale, using more neurons in order to make non-linear problems solvable. Therefore logistic regression only works for problems with linear solutions.

### Genetic Algorithm

The Genetic Algorithm is an optimization algorithm inspired by the process of natural selection. It involves evolving a population of candidate solutions through selection, crossover, and mutation operations to find the optimal solution to a problem. 
