.. contents:: **TABLE OF CONTENTS**
.. section-numbering::

Logistic Regression
===================

The purpose of this project is to implement a *multinomial logistic regression* algorithm from scratch to get a better
understanding of this numerical technique. This project is still under development.

How to Use this Tool
====================

My suggestion is to install this package within a python environment of your choice (on my personal projects I use the
`conda <https://docs.conda.io/en/latest/>`_ package manager). This will allow you to integrate it with other projects
you may be working on.

If this is the route you decide to take, please read the `installation documentation`__.

__ docs/installation.rst

If you're interested only in the heart of the algorithm, you may just use the code within `logistic_regression.py`__
and make the necessary adjustments.

__ logistic_regression/logistic_regression.py

How to Use the CLI
==================

Running the model on an input file
----------------------------------

If you're interested in using the logistic regression on a particular file that you have, ``filename.ext``, then all
you need to do is run the following command:

.. code-block:: powershell

    logistic-regression -f 'relative/filepath/filename.ext' -d 'delimiter'

Note that unless you have a tab delimited file, you must feed the ``delimiter`` option in the CLI command (e.g. ',' for
comma delimited or '|' for pipe delimited files). Furthermore, it is important to keep in mind that you must use the
relative file path from the location where you're calling the command from.

Still under construction!

References
==========

There is a lot of interesting stuff out there regarding this topic! For a surprisingly nice presentation, check out the
information on `Wikipedia <https://en.wikipedia.org/wiki/Multinomial_logistic_regression>`_.

However, the resource I found to be the most useful (not only for this project), was the following book:
`Learning From Data: A Short Course. Y. S. Abu-Mostafa, M. Magdon-Ismail, and H-T. Lin, AMLbook.com, March 2012 <http://www.amlbook.com/>`_
