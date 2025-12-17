# Project Structure

This document outlines the directory structure and the purpose of key files in the project.

```
nexusforgecompress-prototype/
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions workflow for Continuous Integration.
├── bench/
│   ├── download_model.sh   # Script to download models for benchmarking.
│   └── run_bench.py        # Runs benchmark tests.
├── examples/
│   └── example.py          # Demonstrates basic usage of the library.
├── nfc_prototype/
│   ├── __init__.py         # Makes 'nfc_prototype' a Python package.
│   ├── core.py             # Core compression/decompression logic, with v0.2 entropy coding.
│   └── utils.py            # Utility functions.
├── tests/
│   ├── test_core.py        # Core unit tests.
│   ├── v010_test.py        # Tests for v0.1.0 features.
│   ├── v020_test.py        # Tests for v0.2.0 features.
│   └── v030_test.py        # Tests for v0.3.0 streaming features.
├── .gitignore              # Specifies files to be ignored by Git.
├── CHANGELOG.md            # A log of all changes for each version.
├── CONTRIBUTING.md         # Guidelines for contributing to the project.
├── LICENSE                 # Project license file.
├── README.md               # General information and instructions for the project.
├── requirements.txt        # A list of Python packages required by the project.
├── SECURITY.md             # Security policy and reporting instructions.
├── setup.py                # script for building and distributing the package.
├── structure.md            # This file, explaining the project structure.
└── TECHNOLOGY_STACK.md     # A document listing the project's technologies.
```