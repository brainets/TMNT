from distutils.core import setup

AUTHOR = "Vinicius Lima"
MAINTAINER = "Vinicius Lima"
EMAIL = 'vinicius.lima.cordeiro@gmail.com'
KEYWORDS = "Multiplex Network Analysis"
DESCRIPTION = ("Temporal and Multiplex Networks Toolbox")

setup(
    name='TMNT',
    version='0.1dev',
    packages=['tmnt'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.txt').read(),
)
