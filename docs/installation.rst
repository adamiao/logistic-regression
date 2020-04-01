**MAIN INSTALLATION**

For this tutorial I'll be using the *Anaconda* set of tools (*conda*).

Once you have the important files setup (__init__.py, cli.py, __main__.py, setup.py) you will be
able to proceed.

My suggestion is to create a new conda environment for testing first. Once that's done, go inside the directory containing the *setup.py* file and run the following command:

    **pip install -e .**

This installs  the package in editable mode so that any changes made to the code will get propagated across the system (this is quite good for developing purposes).

Now, whenever you are within the conda environment containing the installed package you will be able to use it.

**USING IT IN OTHER PROJECTS**

This assumes you are using the conda environment where the package is installed.

.. code-block:: python

    from logistic_regression import main_script
