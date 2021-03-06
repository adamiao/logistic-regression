EXAMPLE OF MODULE USE
=====================

Here we'll see an example of how I go about using this tool with more control than what the CLI tool provides. I
leverage other packages such as ``pandas`` and ``sklearn`` to facilitate data import and data manipulation.

Pick some example data set to substitute what I used here, ``iris_data.csv``. For reference, the first 5 rows of data
that this data set holds looks like the following:

+--------------+-------------+--------------+-------------+-------------+
| Sepal Length | Sepal Width | Petal Length | Petal Width |   Species   |
+==============+=============+==============+=============+=============+
|     7.0      |     3.2     |      4.7     |     1.4     | versicolor  |
+--------------+-------------+--------------+-------------+-------------+
|     5.1      |     3.5     |      1.4     |     0.2     |   setosa    |
+--------------+-------------+--------------+-------------+-------------+
|     6.4      |     3.2     |      4.5     |     1.5     | versicolor  |
+--------------+-------------+--------------+-------------+-------------+
|     6.3      |     3.3     |      6.0     |     2.5     |  virginica  |
+--------------+-------------+--------------+-------------+-------------+
|     4.9      |     3.0     |      1.4     |     0.2     |   setosa    |
+--------------+-------------+--------------+-------------+-------------+

The code goes as follows:

.. code-block:: python

    import pandas as pd
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    import logistic_regression.logistic_regression as lr

    # Learning parameters. Tuning them will be necessary for you to get a good training.
    epochs, eta, eta_reg = 200, 0.1, 0.001
    test_size = 0.1

    # Read the input file and create a column dictionary. I've used the 'iris_data.csv' as an example for myself.
    df = pd.read_csv('iris_data.csv')
    columns = list(df.columns)
    columns_dct = {'input_columns': columns[:-1],
                   'target_column': columns[-1]}

    # Shuffle the data rows and separate it into training and testing data sets
    df = shuffle(df).reset_index(drop=True)
    df_train, df_test = train_test_split(df, test_size=test_size)

    # Transform the train dataset into a numpy array
    df_train = df_train.to_numpy()

    # Separate the test data into its `input` and `target` parts
    df_test_input = df_test[columns_dct.get('input_columns')].to_numpy()
    df_test_target = df_test[columns_dct.get('target_column')].to_numpy()
    df_test = df_test.to_numpy()

    # Create a logistic regression model and fit it under the input parameters
    model = lr.LogisticRegression(df_train, has_header=False)
    weights = model.fit(epochs=epochs, eta=eta, eta_reg=eta_reg)

    # Now there are two options: evaluate an example input (or list of example inputs) where you don't know the outcome or
    # see how well the model performs on a list of examples where you do know the outcome. Under this second scenario we
    # will examine the `confusion_matrix`. Let's look at the second case.
    classification, confusion_matrix = model.evaluate_performance(df_test, weights=weights, has_header=False)
