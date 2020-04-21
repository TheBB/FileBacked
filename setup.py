#!/usr/bin/env python

from setuptools import setup

setup(
    name='FileBacked',
    version='1.0.0',
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
