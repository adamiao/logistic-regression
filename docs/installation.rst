PACKAGE INSTALLATION
====================

Before proceeding with the installation of this package, I first suggest the creation of a Python environment. I prefer
to do it this way because it makes it easier to get rid of the package once you don't need it anymore. Uninstalling the
package is also another way of getting rid of the package so that's another option as well. As I said in the main page
of this repository, I use the `conda <https://docs.conda.io/en/latest/>`_ package manager.

Creation of Python Environment
------------------------------

Let's create a Python environment. You may use *Anaconda Powershell Prompt* if you have Anaconda installed. Open it and
run the following command:

.. code-block:: powershell

    conda create -n environment_name python=x.x.x

where ``environment_name`` is the name you want for the environment, and ``x.x.x`` is a placeholder for the version
of Python that should be installed (e.g. python=3.6.9). You may want to include the word ``anaconda`` at the end of the
command if you want it to come with all the default Anaconda packages. Once the environment is created you may
go ahead and activate it by running the following command:

.. code-block:: powershell

    conda activate environment_name

Installation of Package
-----------------------

Now that you're using the right environment, change the directory so that you're in the root of the package (where the
``source.py`` file is located). Now run the following command using *Anaconda Powershell Prompt*:

.. code-block:: powershell

    pip install -e .

This installs  the package in editable mode so that any changes made to the code will get propagated across the system
(this is quite good if you intend to make changes in this code). Now, whenever you are within the conda environment
containing the installed package you will be able to use it.

Importing in Python
-------------------

If you're interested in using this in other personal projects you may import it as you would any other Python package
(assuming you're in an environment where this package is currently installed). For example, to access the main file of
this package in Python:

.. code-block:: python

    import logistic_regression.logistic_regression as lr
