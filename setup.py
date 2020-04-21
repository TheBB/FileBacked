#!/usr/bin/env python

from setuptools import setup

setup(
    name='FileBacked',
    version='0.0.0',
    maintainer='Eivind Fonn',
    maintainer_email='evfonn@gmail.com',
    modules=['filebacked'],
    install_requires=[
        'dill',
        'numpy',
        'typing_inspect',
    ],
)
