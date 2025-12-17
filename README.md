# NexusForgeCompress-Prototype (NFC-Prototype)
Prototype scaffolding for a lossless compression framework targeting AI data. This is the architectural skeleton—this version integrates Blosc for basic tensor compression with codec selection, NeuralCompression for entropy coding, and LMCompress-style prediction for improved ratios. Full fusions (e.g., SpQR sparsity) are on the roadmap. Feedback welcome!


## Disclaimer
This is **prototype code** to validate architecture, APIs, and lossless guarantees. Do not expect production 5x+ ratios yet—the current version uses Blosc backend for credible gains. See the roadmap for planned innovations.

NFC does not claim a theoretical upper bound. The compression ratio depends entirely on redundancy in the specific data distribution. Future versions will aim for 3–8× with learned entropy and structured model compression.

Current .nfc format does NOT provide true random access yet; this ships in v0.3+.


## Features
- Strict lossless with raw SHA256 verification.
- 64-bit robust .nfc binary container (single-entry for now).
- Basic streaming for large files.
- CPU-only.


## Installation

It is recommended to use a virtual environment.

### Install `nexusforgecompress`
You can install the base package and its core dependencies (numpy, blosc) with:
```bash
pip install .
```
To include optional dependencies for advanced features (e.g., arithmetic coding via neuralcompression), install with the `full` extra:
```bash
pip install .[full]
```
Note: `neuralcompression` might have its own dependencies, such as PyTorch. Refer to `neuralcompression`'s documentation for details.




## Usage
See `examples/example.py`.

## Benchmarks
Run `bench/run_bench.py` for reproducible results (e.g., vs zstd/snappy on synthetic tensors).

## Testing
- Run all tests: `python -m unittest discover tests`
- Version-specific: `python tests/v010_test.py` (for v0.1), `python tests/v020_test.py` (for v0.2, includes prior).
- CI runs automatically on PRs via GitHub Actions.

## Roadmap (Track Progress)
Use GitHub Issues or Milestones to mark these off as completed.

- **v0.1 (Completed):** Blosc integration + robust container + streaming.
- **v0.2 (Completed):** NeuralCompression entropy coders.
- **v0.3 (Completed):** Indexed block framing for partial decode (enabling true random access).
- **v0.4 (Completed):** LMCompress-style prediction.
- **v0.5 (In Progress):** [Placeholder for next major feature]
- **v1.0:** Full AI engine.
- **v1.1:** Rust CLI.
- **v2.0:** Distributed graph-based chunk maps.

Licensed under MIT.
