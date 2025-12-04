Changelog
v0.1.1 (2025-12-04)

- Fixed TypeError: ZipNN.compress() missing 1 required positional argument: 'data' by calling `compress` and `decompress` methods on an instance of `zipnn.ZipNN` rather than directly on the class.

v0.1.0 (2025-12-03)

- Initial prototype with ZipNN, robust .nfc, streaming, and tests.
- Added documentation for Windows C++ build tool prerequisites to README.md.