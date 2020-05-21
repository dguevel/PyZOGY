from setuptools import setup
from os import path
from glob import glob

setup(
    name='PyZOGY',
    version='0.0.1',
    author='David Guevel',
    author_email='guevel.david@gmail.com',
	entry_points = {'console_scripts': ['pyzogy = PyZOGY.__main__ : main']},
    license='LICENSE.txt',
    description='PyZOGY is a Python implementation of the ZOGY algorithm.',
    install_requires=['numpy>=1.12', 'astropy', 'scipy', 'statsmodels', 'matplotlib', 'sep'],
    packages=['PyZOGY'],
	test_suite = 'PyZOGY.test'
)
