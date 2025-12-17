from setuptools import setup, find_packages

setup(
    name='nexusforgecompress',
    version='0.3.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.0',
        'blosc>=1.11.1',
    ],
    extras_require={
        'full': ['neuralcompression>=0.2.0'],
    },
    description='Prototype for lossless AI data compression',
    author='Quintin Reynecke',
    license='MIT',
)