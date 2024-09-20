# File: setup.py

from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='datapalette',
      version="1.0",
      description="A comprehensive toolkit for creating, preprocessing, and augmenting image datasets for various deep learning tasks.",
      packages=find_packages(),
      install_requires=requirements,
      test_suite='tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/datapalette-run'],
      zip_safe=False)