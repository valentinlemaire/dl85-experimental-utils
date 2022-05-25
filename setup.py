from setuptools import find_packages, setup

setup(
    name='experimental-utils',
    version='0.1',
    packages=find_packages(),
    requires=['pandas', 'numpy', 'sklearn', 'scipy', 'skgarden'],
)