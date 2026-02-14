from setuptools import setup, find_packages

setup(
    name='nexusforgecompress',
    version='2.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.0',
        'blosc>=1.11.1',
        'rich',
        'numba',
        'requests',
        'flask',
    ],
    extras_require={
        'full': ['neuralcompression>=0.2.0'],
    },
    description='Neural File System for AI data compression and deduplication',
    author='Quintin Reynecke',
    license='MIT',
)
