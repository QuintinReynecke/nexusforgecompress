# Changelog

## v0.5.0 (2025-12-17)
- **Fix:** Corrected Blosc filters test in `v050_test.py` to use integer representation `4` for `BLOSCDELTA` instead of the non-existent `blosc.BLOSCDELTA` attribute.

## v0.3.0 (2025-12-17)
- **Fix:** Removed unexpected `chunk_size` argument from `decompress_stream` calls in `v010_test.py`.
- **Fix:** Updated metadata `compression_stack` assertion in `v020_test.py` to correctly expect `["blosc_zstd"]`.
- **Fix:** Explicitly set `current_offset = 34` in `decompress` method within `nfc_prototype/core.py` to resolve `JSONDecodeError` in `v030_test.py` by ensuring accurate metadata parsing.
- **Fix:** Restored the test logic for `v020_test.py` after an accidental deletion, ensuring the correct application of `sys.path.insert(0, ...)` and `metadata_start_offset = 34` for proper local package loading and metadata handling.
- **Fix:** Ensured `v030_test.py` correctly loads the local `nfc_prototype` module by adding `sys.path.insert(0, ...)` to prioritize local package loading.
- **Fix:** Corrected `v020_test.py` to prioritize loading the local `nfc_prototype` package over the installed `site-packages` version by using `sys.path.insert(0, ...)` instead of `sys.path.append(...)`. This ensures test execution reflects changes in local development files.
- **Fix:** Refined the calculation of `self.calculated_header_len` in `nfc_prototype/core.py` by removing a duplicate `2 +` for reserved bytes. This ensures the header length is precisely 34 bytes, correctly resolving metadata parsing issues during decompression, preventing `ValueError: Metadata length is larger than remaining binary data` and `json.decoder.JSONDecodeError` by guaranteeing metadata is sliced from the correct offset.
- **Major Change:** Replaced the core compression library from `zipnn` to `blosc`.
  - This decision was made due to persistent, unresolvable errors with `zipnn`'s numpy array handling.
  - `blosc` is a mature, high-performance compressor optimized for numerical data and is expected to be more reliable.
- **Refactor:** The `core.py` module has been completely rewritten to use `blosc`.
- **Fix:** This change resolves the `ValueError: Support only uint32 with NumPy format` error.
- **Build:** `setup.py` dependencies have been updated to remove `zipnn`, `neuralcompression`, `torch`, and `scipy`, and add `blosc`.
- **Feature:** Refactored `decompress_stream` in `nfc_prototype/core.py` for true streaming, enabling efficient processing of large compressed files without loading the entire file into memory.
- **Feature:** Added Blosc codec selection (`zstd`, `lz4`, `lz4hc`, `zlib`) via the `codec` parameter in `NFCPrototype.__init__`, allowing users to choose different compression algorithms for finer control over compression ratio and speed.

## v0.4.0 (2025-12-17)
- **Feature:** Implemented LMCompress-style prediction using delta encoding for `numpy.ndarray` data.
  - Added `use_prediction` parameter to `compress` method to enable/disable prediction.
  - Automatically handles `dtype` promotion for residuals to preserve lossless integrity.
  - Updated `decompress` method to reconstruct original data from residuals.
- **Fix:** Resolved hash mismatch for floating-point data with delta encoding by promoting `residuals_dtype` to `float64` for higher precision during prediction calculations.
- **Test:** Added `v040_test.py` to verify lossless compression/decompression with prediction and evaluate compression ratio improvements.

## v0.2.2 (2025-12-17)
- **Fix:** Refactored `compress` and `decompress` methods to correctly handle different `numpy` `dtypes`. This resolves an issue where tests for various `dtypes` were failing. The fix involves removing duplicated code, ensuring metadata is correctly serialized, and properly using `zlib` as a fallback for `zipnn` unsupported `dtypes`.

## v0.2.1 (2025-12-15)
- **Fix:** Resolved `UnboundLocalError` for `dtype` in `compress` method by moving its assignment to an earlier point in the code.
- **Fix:** Implemented `zlib` wrapping workaround for `zipnn`'s aggressive `uint32` format enforcement. Non-`uint32` data is now `zlib` compressed before passing to `zipnn` (when `ArithmeticCoder` is not used) to ensure generic byte compression.
- **Fix:** Implemented workaround for `zipnn`'s `uint32` format limitation by converting `uint8` data to `uint32` before compression (when `ArithmeticCoder` is not used) and reverting during decompression, preserving lossless integrity.
- **Fix:** Addressed `ValueError: Support only uint32 with NumPy format` from `zipnn` by restricting `bytearray_dtype` usage to only `uint32` data, and otherwise compressing as generic byte streams.
- **Fix:** Moved `original_hash` calculation in `compress` method to immediately after `data_bytes` initialization to prevent accidental mutation of `data_bytes` by compression libraries before hashing.
- **Fix:** Corrected `AttributeError: 'bytes' object has no attribute 'hexdigest'` in debug print statements by using `.hex()` method for byte objects.
- **Debug:** Added extensive print statements to `compress` and `decompress` methods for detailed tracing and debugging of data integrity issues.
- **Fix:** Enhanced `decompress` method's `ArithmeticCoder` handling to strictly depend on the `ARITHMETIC_CODING_FLAG` from the header and validate `neuralcompression` installation for arithmetic-coded binaries.
- **Fix:** Corrected data corruption issue by conditionally initializing `zipnn.ZipNN` based on whether arithmetic coding was applied. A new flag in the header (`ARITHMETIC_CODING_FLAG`) was introduced to ensure symmetric compression/decompression.
- **Fix:** Addressed `UnboundLocalError` in `compress` method by refactoring the compression logic to ensure `compressed` variable is always assigned before use.
- **Fix:** Resolved data corruption during decompression by ensuring the `entropy_compressed` data is correctly passed to `zipnn_instance.compress` when `ArithmeticCoder` is enabled.
- **Fix:** Updated `zipnn` API calls from `encode`/`decode` to `compress`/`decompress` to match the latest version.

## v0.2.0 (2025-12-08)
- **Feature:** Integrated `neural-compression` library to add an entropy coding layer (`ArithmeticCoder`), improving compression ratios.
- **Fix:** Corrected package installation issues by fixing the package structure and updating `requirements.txt`.
- **Docs:** Updated `README.md` with improved installation instructions.
- Added `v020_test.py` for entropy-specific tests.

## v0.1.0 (2025-12-03)
- Initial prototype with ZipNN, robust .nfc, streaming, and tests.