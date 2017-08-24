from distutils.core import setup
from os import path
from glob import glob

setup(
    name='PyZOGY',
    version='1.0.0',
    author='David Guevel',
    author_email='guevel.david@gmail.com',
    scripts=glob(path.join('bin/pyzogy')),
    license='LICENSE.txt',
    description='PyZOGY is a Python implementation of the ZOGY algorithm.',
    requires=['numpy', 'astropy', 'scipy', 'statsmodels', 'matplotlib', 'sep'],
    packages=['PyZOGY'],
)