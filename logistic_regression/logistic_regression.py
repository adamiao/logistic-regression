"""
This module implements a `multinomial logistic regression` learning algorithm with L2-regularization.
"""

import logging
import numpy as np

LOGGER = logging.getLogger(__name__)


class LogisticRegression:

    def __init__(self, input_data):
        """
        Args:
            input_data: two-dimensional array (numpy or list of lists) where each row of points is of the form
                        [x_1, x_2, ..., x_m, category]. The array was started at `x_1` on purpose because later on a
                        term `x_0` = 1 will have to be added to multiply by the bias, `w_0`. The indexing used for the
                        training dataset is meant to match the indexing used for the weight vector.
        """

        self._instance_initialization(input_data)
        self.weights_fitted = None

    def fit(self, epochs=1000, eta=0.1, eta_reg=0.001):
        """
        This method runs the logistic regression model
        Args:
            epochs: number of times we'll process the entire training dataset with the intent of updating model weights.
            eta: step size used for gradient descent operation.
            eta_reg: step size for L2-regularization term.

        Returns: `w`, a two-dimensional numpy array representing the converged set of weights for the training data
                 input. Each row of `w` is of the form [w_0, w_1, ..., w_m].
        """
        w = self._weight_initialization()
        for _ in range(epochs):
            w = self._gradient_descent(self.input_data, w, eta, eta_reg)

        self.weights_fitted = w
        return w

    def evaluate(self, input_data, weights=None):
        """
        This method will return the classification and prediction based on the input data. For this method we do not
        have to add the category to the last term of each row since we don't know which category the point belongs to.
        Args:
            input_data: numpy array not including the category as the last term
            weights: weights matrix (numpy array)

        Returns:
            classification: numpy array where each row shows the chances of an input point belonging to one of the
                            categories. The rows do not necessarily add up to one.
            predictions: array of indices. These indices are related to the categories as specified by the object
                         `self.category_dictionary`.
        """

        if weights is None:
            weights = self.weights_fitted

        # Make sure the data is a numpy array and insert the number 1 in the beginning of each row for the bias term
        data = np.array(input_data, dtype=float)
        data = np.insert(data, 0, 1.0, axis=1)

        classification = 1.0 / (1.0 + np.exp(-np.matmul(data, weights.T)))
        predictions = np.argmax(classification, axis=1)

        return classification, predictions

    def evaluate_performance(self, input_data, weights=None):
        """
        This method will return the predictions and the confusion matrix for a testing dataset.
        Args:
            input_data: numpy array including the category as the last term for each row
            weights: weights matrix (numpy array)

        Returns:
            classification: numpy array where each row shows the chances of an input point belonging to one of the
                            categories. The rows do not necessarily add up to one.
            confusion_matrix: numpy array where the rows are the category predictions while the columns are the
                              true categories.
        """
        if weights is None:
            weights = self.weights_fitted

        data, data_category, data_parameters = self._data_preparation(input_data)
        classification = 1.0 / (1.0 + np.exp(-np.matmul(data, weights.T)))

        true_categories = [data_parameters['category_dictionary'].get(category) for category in data_category]
        predictions = np.argmax(classification, axis=1)

        confusion_matrix = np.zeros((self.n_categories, self.n_categories))
        for prediction, category in zip(predictions, true_categories):
            if prediction == category:
                confusion_matrix[prediction, prediction] += 1
            else:
                confusion_matrix[prediction, category] += 1

        return classification, confusion_matrix

    def _gradient_descent(self, training_batch, weights, eta, eta_reg):
        """
        Gradient descent method with L2-regularization term
        Args:
            training_batch: numpy array of shape [n, 1 + training_data_dimension] where each row is of the form
                           [x_0, x_1, x_2, ..., x_m, category] and `x_0` = 1 is used to multiply by the bias
            weights: numpy array of shape [n_categories, 1 + training_data_dimension]. The plus `1` is for the bias
            eta: step size for gradient descent
            eta_reg: step size for regularization

        Returns: updated `weights` numpy array with the same shape input shape
        """

        for category, category_idx in self.category_dictionary.items():
            summation = np.zeros(self.weight_dimension)
            for x, x_category in training_batch:
                y = (1 if x_category == category else -1)
                denominator = 1 + np.exp(y * np.dot(weights[category_idx], x))
                summation += y * x / denominator
            weights[category_idx] += eta / self.n * summation - eta_reg * weights[category_idx]

        return weights

    def _weight_initialization(self):
        return np.random.rand(self.n_categories, self.weight_dimension)

    def _instance_initialization(self, input_data):
        data, data_category, data_parameters = self._data_preparation(input_data)
        self.category_dictionary = data_parameters.get('category_dictionary')
        self.n_categories = data_parameters.get('n_categories')  # number of distinct categories
        self.n = data_parameters.get('n')  # number of points in the training data
        self.m = data_parameters.get('m')  # dimension of the input data
        self.weight_dimension = self.m + 1  # equals the dimension of the input data plus 1 for the bias term

        # Here we group the `input_data` as a list of tuples where each tuple is of the form (data, data_category)
        self.input_data = [(x, y) for x, y in zip(data, data_category)]

        return

    @staticmethod
    def _data_preparation(input_data):
        # We have to break `input_data` into `data` (numerical) and respective `data_category` (string or numerical)
        input_data = input_data[1:]  # we will ignore the header from the `input_data`
        data = np.array([x[:-1] for x in input_data], dtype=float)
        data_category = [int(x[-1]) if isinstance(x[-1], float) else x[-1] for x in input_data]
        category_dictionary = {key: val for val, key in enumerate(set(data_category))}
        data_parameters = {'n': np.shape(data)[0],  # number of points in the training data
                           'm': np.shape(data)[1],  # dimension of the input data
                           'n_categories': len(category_dictionary),
                           'category_dictionary': category_dictionary}

        # Here we insert the number 1 in the beginning of each row so we take into consideration the weight bias term
        data = np.insert(data, 0, 1.0, axis=1)
        return data, data_category, data_parameters
