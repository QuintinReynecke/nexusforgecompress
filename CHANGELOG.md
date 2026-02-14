# Changelog

## v2.0.0 (2026-02-14)
- **Major Milestone:** **Distributed Swarm Rehydration**. `NFCStore` can now rehydrate models by fetching missing blocks from multiple remote registries in parallel (BitTorrent-style).
- **Feature:** **Prefetching**. Added `prefetch_manifest` API and `prefetch` CLI command to explicitly materialize a model from the swarm before usage.
- **Refactor:** `registry.py` server now respects `PORT` environment variable, facilitating multi-registry testing and deployment.

## v1.8.0 (2026-02-14)
- **Unique Feature:** **Neural Registry (Remote Sync)**. Implemented bidirectional synchronization between local `NFCStore` and remote HTTP/Cloud registries. Only missing blocks are downloaded during rehydration.
- **Unique Feature:** **Neural Registry Server**. Added a lightweight Flask-based server (`registry.py`) to allow anyone to host their own private model block repository.
- **Feature:** **Enhanced Manifests**. Manifests now include full original metadata and hashes for perfect lossless rehydration across the network.

## v1.7.0 (2026-02-14)
- **Unique Feature:** **Neural Forge Manifests (.nfm)**. Generate lightweight model recipes that list tensor hashes and metadata without including the raw data.
- **Unique Feature:** **Model Rehydration**. Reconstruct full NFC files instantly from a manifest by pulling blocks from the local `NFCStore`.
- **Feature:** **Distributed Foundation**. Paving the way for decentralized model sharing (v2.0).

## v1.6.0 (2026-02-14)
- **Validation:** **Hard Proof Benchmarking Suite**. Added a comprehensive benchmark comparing NFC against industry standards (Zstd, LZ4).
- **Evidence:** Confirmed NFC achieves up to **97% savings** using `NFC-Diff` and out-performs Zstd on numerical gradient distributions by **~20%** in compression ratio.
- **Deduplication Proof:** Verified that identical weight blocks are reduced to **<1KB pointers**, regardless of original size.

## v1.5.0 (2026-02-14)
- **Unique Feature:** **Virtual Model Merging**. Create new NFC files that reference existing weights in the global store without copying data. Perfect for model ensemble and merging experiments.
- **Feature:** **NFCLoader (Lazy Loading)**. A dictionary-like interface for NFC files that only decompresses tensors when accessed. Reduces VRAM/RAM overhead for massive models.
- **Feature:** **Unified Header Checksums**. Improved self-healing consistency across all block types.

## v1.4.0 (2026-02-14)
- **Performance:** **Hardware Acceleration (JIT)**. Integrated `numba` for JIT-optimized numerical predictors (Delta encoding/decoding), providing significant throughput improvements on multi-core CPUs.

## v1.3.0 (2026-02-14)
- **Unique Feature:** **Global Deduplication (NFC-CAS)**. Implemented a Content-Addressable Storage system (`~/.nfc_store`). Files can now store 32-byte hash pointers instead of data, enabling massive disk savings across model versions.
- **Unique Feature:** **NFC-Explorer (TUI)**. Added an interactive Terminal UI (`nfc explore`) to browse model layers, view compression ratios, and inspect metadata/stats.
- **Feature:** **Stream Deduplication**. Added `--dedup` flag to `compress` CLI command for deduplicating streamed archives.

## v1.2.0 (2026-02-14)
- **Unique Feature:** **Differential Tensor Compression (NFC-Diff)**. Compress the difference between two model checkpoints (e.g., base model vs. fine-tuned).
- **Unique Feature:** **Numerical Peek (NFC-Peek)**. Queryable metadata (Min, Max, Mean) stored in the block header for instant analysis without decompression.
- **Feature:** **Self-Healing Header**. Added an Adler-32 checksum to the header to detect corruption/bit-rot instantly.
- **Refactor:** Improved `compress` API to support extensible metadata and custom format hints.

## v1.1.0 (2026-02-14)
- **Feature:** Integrated **Safetensors** support. Added `compress_safetensors` and `decompress_safetensors` to easily compress HuggingFace model files.
- **Feature:** Added a basic **CLI interface** to `core.py` for command-line compression/decompression.

## v1.0.0 (2026-02-14)
- **Major Feature:** "Full AI Engine" milestone reached.
- **Feature:** Added PyTorch integration with `compress_tensor` and `decompress_tensor` helper methods.
- **Feature:** Implemented **Auto-tuning** heuristic to automatically select the best prediction strategy (e.g., delta vs none) based on data characteristics.
- **Feature:** Implemented **Parallel Streaming Compression/Decompression** using multi-threading, significantly improving performance for large datasets.
- **Feature:** Added **Random Access** support for streamed files via indexed block framing and footer indexing.
- **Refactor:** Cleaned up `nfc_prototype/core.py` to remove legacy code and improve maintainability.
- **Docs:** Updated `README.md` and `structure.md` to reflect the new architecture and features.

## v0.5.0 (2025-12-17)
- **Cleanup:** Removed duplicated code blocks and redundant method definitions in `nfc_prototype/core.py`.
- **Verification:** Successfully verified all v0.5.0 features (dynamic timestamps, prediction strategies, Blosc filters) with comprehensive test suite.
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