# Technology Stack

This document outlines the core technologies, libraries, and tools used in the `nexusforgecompress-prototype` project.

## Core Libraries

| Library | Version | Purpose |
|---|---|---|
| [numpy](https://numpy.org/) | `>=1.26.0` | The fundamental package for scientific computing with Python. Used for all tensor and numerical operations. |
| [zipnn](https://github.com/zipnn/zipnn) | `>=0.1.0` | The core compression library for handling AI tensors. This is a critical component that provides the underlying compression algorithm. It requires a C++ compiler for installation. |

## Future Dependencies (Roadmap)

The following libraries are not used in the current version but are planned for future development milestones.

| Library | Version | Purpose |
|---|---|---|
| [neural-compression](https://github.com/facebookresearch/NeuralCompression) | `>=0.3.0` | Planned for `v0.2` to integrate advanced, learned entropy coders for higher compression ratios. |

## Development & Build Tools

| Tool | Purpose |
|---|---|
| [Python](https://www.python.org/) | The primary programming language for the project. |
| [pip](https://pip.pypa.io/en/stable/) | Used for installing and managing Python packages. |
| [setuptools](https://setuptools.pypa.io/en/latest/) | Used for building and distributing the Python package. |
| [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) | A system-level prerequisite for compiling the C++ extensions required by the `zipnn` library. |
