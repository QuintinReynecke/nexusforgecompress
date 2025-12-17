# Technology Stack

This document outlines the core technologies, libraries, and tools used in the `nexusforgecompress-prototype` project.

## Core Libraries

| Library | Version | Purpose |
|---|---|---|
| [numpy](https://numpy.org/) | `>=1.26.0` | The fundamental package for scientific computing with Python. Used for all tensor and numerical operations. |
| [blosc](https://www.blosc.org/) | `>=1.11.1` | A high-performance compressor optimized for binary data, especially effective for numerical and AI datasets. Replaced `zipnn` in v0.3.0. |
| [neuralcompression](https://github.com/facebookresearch/NeuralCompression) | `>=0.2.0` | (Optional, via `[full]` extra) Provides advanced entropy coding techniques, such as arithmetic coding. |

## Development & Build Tools

| Tool | Purpose |
|---|---|
| [Python](https://www.python.org/) | The primary programming language for the project. |
| [pip](https://pip.pypa.io/en/stable/) | Used for installing and managing Python packages. |
| [setuptools](https://setuptools.pypa.io/en/latest/) | Used for building and distributing the Python package. |
| [GitHub Actions](https://github.com/features/actions) | Used for continuous integration (CI) to automatically run tests on pull requests. |