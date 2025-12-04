from setuptools import setup, find_packages

setup(
    name='nexusforgecompress',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.0',
        'zipnn>=0.1.1',  # For AI tensors
        # 'neural-compression>=0.3.0',  # Optional for entropy; fallback if unavailable
    ],
    extras_require={
        'entropy': ['neural_compression>=0.3.0'],  # Optional for v0.2+; install from GitHub
    },
    description='Prototype for lossless AI data compression',
    author='Quintin Reynecke',
    license='MIT',
)