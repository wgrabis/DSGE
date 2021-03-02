# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

setup(
    name='DSGE solver',
    long_description=readme,
    packages=find_packages(exclude=('tests'))
)