import hybgp
from setuptools import setup, find_packages


setup(
    name='hybgp',
    version=hybgp.__version__,
    test_suite='hybgp.tests',
    packages=find_packages(), install_requires=['deap', 'sympy', 'numpy']
)