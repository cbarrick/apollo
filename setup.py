'''A setuptools based setup module.

See:
- <https://packaging.python.org/>
- <https://setuptools.readthedocs.io/en/latest/>
- <https://github.com/pypa/sampleproject>
'''

from setuptools import setup, find_packages
from pathlib import Path

repo = Path(__file__).resolve().parent
long_description = (repo / 'README.md').read_text()

setup(
    name='apollo',
    version='0.1.0-dev1',
    author='Chris Barrick',
    author_email='cbarrick1@gmail.com',
    url='https://github.com/cbarrick/apollo',
    description='Utilities for solar radiation data analysis',
    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'apollo = apollo.__main__:main',
        ],
    },

    keywords='solar radiation weather atmosphere ml machine-learning',
    classifiers=[
        # Maturity
        # 'Development Status :: 1 - Planning',
        # 'Development Status :: 2 - Pre-Alpha',
        'Development Status :: 3 - Alpha',
        # 'Development Status :: 4 - Beta',
        # 'Development Status :: 5 - Production/Stable',
        # 'Development Status :: 6 - Mature',
        # 'Development Status :: 7 - Inactive',

        # Audience
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Atmospheric Science',

        # # License
        # # NOTE: This MUST be kept in sync with LICENSE file.
        # # NOTE: The LICENSE file is cannonical.
        # 'License :: OSI Approved :: MIT License',

        # Supported Python versions
        'Programming Language :: Python :: 3.6',
    ],
)
