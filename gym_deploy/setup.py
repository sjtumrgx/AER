from setuptools import find_packages
from distutils.core import setup

setup(
    name='aer_wtw_deploy',
    version='1.0.0',
    author='Gabriel Margolis',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='gmargo@mit.edu',
    description='Toolkit for Go2 sim-to-real policy deployment.'
)
