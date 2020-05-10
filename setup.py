#!/usr/bin/env python

from setuptools import setup

with open('README.rst', 'r') as f:
    readme = f.read()

setup(
    name='FileBacked',
    version='2.0.0',
    description='Simple file-backed HDF5 storage for Python objects',
    long_description=readme,
    maintainer='Eivind Fonn',
    maintainer_email='evfonn@gmail.com',
    url='https://github.com/TheBB/FileBacked',
    py_modules=['filebacked'],
    install_requires=[
        'dill',
        'numpy',
        'typing_inspect',
    ],
)
