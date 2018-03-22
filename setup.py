'''A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
'''

from setuptools import setup, find_packages
from pathlib import Path


# The long_description is just the contents of the README
repo = Path(__file__).resolve().parent
long_description = (repo / 'README.md').read_text()

setup(
    # The name of the project.
    #
    # It determines how users can install the project, e.g.:
    #
    #     $ pip install sampleproject
    #
    # And where it will live on PyPI:
    #
    #     https://pypi.org/project/sampleproject/
    #
    # There are some restrictions on what makes a valid project name:
    # https://packaging.python.org/specifications/core-metadata/#name
    name='ugasolar',

    # Versions should comply with PEP 440: https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.1-dev',

    # Packages included in the project.
    # You can list packages manually, or use `setuptools.find_packages`.
    packages=['ugasolar'],

    # Corresponds to the "Summary" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    description='Utilities for solar radiation data analysis',

    # Corresponds to the "Description" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-optional
    long_description=long_description,

    # Corresponds to the "Home-Page" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url='https://github.com/cbarrick/SolarRadiation',

    # The name and email address of the project owner.
    author='Chris Barrick',
    author_email='cbarrick1@gmail.com',

    # What does the project relate to?
    # Note that this is a string of words separated by whitespace, not a list.
    keywords='solar radiation weather atmosphere ml machine-learning',

    # More formal categorization of the project.
    # For a list of valid classifiers, see https://pypi.python.org/pypi?:action=list_classifiers
    classifiers=[
        # Maturity
        # 'Development Status :: 1 - Planning',
        'Development Status :: 2 - Pre-Alpha',
        # 'Development Status :: 3 - Alpha',
        # 'Development Status :: 4 - Beta',
        # 'Development Status :: 5 - Production/Stable',
        # 'Development Status :: 6 - Mature',
        # 'Development Status :: 7 - Inactive',

        # Audience
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Atmospheric Science',

        # License
        # NOTE: This MUST be kept in sync with LICENSE file.
        # NOTE: The LICENSE file is cannonical.
        'License :: OSI Approved :: MIT License',

        # Supported Python versions
        'Programming Language :: Python :: 3.6',
    ],
)
