import logging
import numpy as np

LOGGER = logging.getLogger(__name__)


class DataInput:

    def __init__(self, filename, delimiter='\t'):

        # Input Data Ingestion
        self.filename = filename
        self.delimiter = delimiter.replace(' ', '')  # to prevent user from inserting extra spaces in the delimiter
        self._input_data_initialization()

    def _input_data_initialization(self):
        """
        This method reads a file and creates objects of interest that are associated with its data. In some  sense this
        is an initialization method for this class.

        Returns: a list containing the header of the input data (`self.data_header`), the input data content
                 (`self.data_content`), and the input data address (`self.data_address`).
        """

        # Reading the input data
        try:
            data = []
            with open(self.filename, 'r') as f:
                for line in f.readlines():
                    row = [row_element.strip() for row_element in line.split(self.delimiter)]
                    data.append(row)
        except OSError as err:
            LOGGER.error(f'Program was not able to read input data: {err}')
            raise err

        # Manipulating input data into objects of interest such as `header`, `content`, and `address`
        try:
            self.data_header = [data[0]]
            self.data_content = data[1:]
            self.data_address = hex(id(data))
        except Exception as err:
            LOGGER.error(f'An error occurred while trying to manipulate the input data: {err}')
            raise err

        return

    def ndarray(self):
        """
        Transform the input data into a numpy array.

        Returns: numpy array
        """

        try:
            data = np.array(self.data_header + self.data_content)
        except Exception as err:
            LOGGER.error(f'Error transforming input data into numpy array: {err}')
            raise err
        return data

    def shuffle_input_data(self, seed=4):
        """
        This method will shuffle. If the user is interested they may feed in a `seed` to keep the shuffling
        deterministic.

        Args:
            seed: integer used to keep shuffling deterministic

        Returns: a list representing the shuffled dataset
        """

        # Break the input data into a `header` and its content (`data`)
        data = self.data_content.copy()

        # Set the `seed` value for the shuffling
        np.random.seed(seed)

        # Shuffle the `data`
        np.random.shuffle(data)

        # Return the `data` with its `header`
        return self.data_header + data

    @staticmethod
    def training_test_split(data, test_size=0.25):
        """
        This method splits the input data into testing and training datasets based on a fractional `test_size`
        parameter.

        Args:
            data: a list of lists or a numpy array representing the input data
            test_size: a fractional value between 0 and 1 representing the size of the `testing_data` relative to the
                       input dataset size

        Returns: a pair of datasets within a tuple, representing the `testing_data` and the `training_data`
                 respectfully.
        """

        header, data_content = [data[0]], data[1:]

        # We find the row which splits the data and we break the input data at that point
        splitting_row = round(len(data_content) * test_size)
        testing_data, training_data = data_content[:splitting_row], data_content[splitting_row:]
        testing_data, training_data = header + testing_data, header + training_data  # add the `header` back

        return training_data, testing_data
