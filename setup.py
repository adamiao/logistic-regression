from setuptools import setup, find_packages


setup(
    name='logistic_regression',
    version='0.1.0',
    description='The goal of this application is to run a logistic regression model on a given input dataset.',
    author='Alex Damiao',
    install_requires=[
        'click >= 7.0',
        'numpy >= 1.16.5'
    ],
    packages=find_packages(exclude=['docs', 'tests', 'logs']),
    entry_points={
        'console_scripts': ['logistic-regression=logistic_regression.cli:main']
    }
)
