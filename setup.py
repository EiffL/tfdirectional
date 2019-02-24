
import datetime
import os
import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

setup(
    name='tfdirectional',
    description='Directional distributions in Tensorflow',
    license='Apache 2.0',
    packages=find_packages(),

    exclude_package_data={'': ['BUILD', '*.h', '*.cc']},
    zip_safe=False,
)
