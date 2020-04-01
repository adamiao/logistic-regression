"""Command line interface"""

import click
from logistic_regression import __version__, main_script


@click.command()
@click.option('--version', '-V', is_flag=True, help='Return version number.')
@click.option('--filename', '-f',
              help='Run the main script with a relative input filename. The default delimiter is tab (\t).',
              type=str)
@click.option('--delimiter', '-d',
              help='Add a delimiter for the file in question. The default is tab delimited.',
              type=str)
@click.option('--test-size', '-ts',
              help='Add a number between 0 and 1 representing the fraction of the test size relative to the entire '
                   'dataset.',
              type=float)
def main(version, filename, delimiter, test_size):
    """Entry point console script for package CLI."""

    if version:
        click.echo(__version__)

    if test_size:
        main_script.main()

    if filename and delimiter:
        return main_script.main(filename=filename, delimiter=delimiter)

    if filename:
        return main_script.main(filename=filename)
