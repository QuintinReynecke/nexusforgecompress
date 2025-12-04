# NexusForgeCompress-Prototype (NFC-Prototype)
Prototype scaffolding for a lossless compression framework targeting AI data. This is the architectural skeleton—this version integrates ZipNN for basic tensor compression, with hooks for NeuralCompression entropy. Full fusions (e.g., LMCompress semantics, SpQR sparsity) are on the roadmap. Feedback welcome!


## Disclaimer
This is **prototype code** to validate architecture, APIs, and lossless guarantees. Do not expect production 5x+ ratios yet—the current version uses the ZipNN backend for credible gains. See the roadmap for planned innovations.

NFC does not claim a theoretical upper bound. The compression ratio depends entirely on redundancy in the specific data distribution. Future versions will aim for 3–8× with learned entropy and structured model compression.

Current .nfc format does NOT provide true random access yet; this ships in v0.3+.


## Features
- Strict lossless with raw SHA256 verification.
- 64-bit robust .nfc binary container (single-entry for now).
- Basic streaming for large files.
- CPU-only.


## Installation

<br>

> **Important Prerequisite for Windows Users**
>
> The `zipnn` library requires a C++ compiler to be installed. If you are on Windows, you must install the **Microsoft C++ Build Tools**.
>
> 1.  Download the tools from: [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> 2.  During installation, select the **"Desktop development with C++"** workload.
>
> Failure to do this will result in a `Microsoft Visual C++ 14.0 or greater is required` error.

<br>

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install .
# pip install git+https://github.com/facebookresearch/NeuralCompression.git # Optional, for future backends
```

## Usage
See `examples/example.py`.

## Benchmarks
Run `bench/run_bench.py` for reproducible results (e.g., vs zstd/snappy on synthetic tensors).

## Roadmap (Track Progress)
Use GitHub Issues or Milestones to mark these off as completed.

- **v0.1 (Current):** ZipNN integration + robust container + streaming.
- **v0.2:** NeuralCompression entropy coders.
- **v0.3:** Indexed block framing for partial decode (enabling true random access).
- **v0.4:** LMCompress-style prediction.
- **v1.0:** Full AI engine.
- **v1.1:** Rust CLI.
- **v2.0:** Distributed graph-based chunk maps.

Licensed under MIT.
