"""
This main script will run the logistic regression model on some input dataset. The main parameters for this code to run
are as follows:

'epochs': (default value: 250)
'eta': batch gradient descent learning parameter (default value: 0.1)
'eta_reg': regularization learning parameter (default value: 0.001)
'test_size': split percentage of training/testing (default value: 0.25)
'seed': used to always get the same shuffle pattern (default value: 4)

"""

from pathlib import Path
from logistic_regression import LOGGER
from logistic_regression.utils.data_utils import DataInput
from logistic_regression.utils.logistic_regression import LogisticRegression


def main(filename=None, delimiter='\t', epochs=250, eta=0.1, eta_reg=0.001, test_size=0.25, seed=4):

    LOGGER.info(f'filename: {filename} | delimiter: "{delimiter}" | epochs: {epochs} | eta: {eta} | '
                f'eta_reg: {eta_reg} | test_size: {test_size} | seed: {seed}')

    if not filename:
        LOGGER.error('This code needs an input file!')
        raise Exception
    else:
        filename = Path.cwd().joinpath(filename)

    # Create a `DataInput` instance for the input data
    data_class = DataInput(filename=filename, delimiter=delimiter)

    # Splitting the full dataset into training and testing
    shuffled_data = data_class.shuffle_input_data(seed=seed)  # first we shuffle the dataset
    training_data, testing_data = DataInput.training_test_split(data=shuffled_data, test_size=test_size)

    # Application of the multinomial logistic model
    model = LogisticRegression(training_data)
    w = model.fit(epochs=epochs, eta=eta, eta_reg=eta_reg)

    # Checking performance of the converged model by looking at the `confusion matrix`
    *_, confusion_matrix = model.evaluate_performance(testing_data, w)
    LOGGER.info(f'Confusion matrix: {confusion_matrix}')
