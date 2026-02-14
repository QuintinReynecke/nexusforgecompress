# NexusForgeCompress (NFC) v2.0.0
**The Neural File System for AI Data.**

NexusForgeCompress (NFC) is a specialized, lossless compression, deduplication, and distributed sharing framework built for AI tensors and model weights. By combining numerical knowledge with a decentralized architecture, NFC reduces storage and bandwidth costs by up to 97% while maintaining strict bit-perfect integrity.

---

## 📊 Hard Proof: The Evidence
The following results are verified by the `bench/hard_proof_bench.py` suite against industry standards.

### 1. Compression Ratio (Numerical Advantage)
NFC utilizes numerical predictors (Delta Encoding) that understand the structure of AI weights.

| Dataset | Orig Size | NFC Ratio (Lossless) | Zstd Ratio | NFC Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **Gradients (Smooth)** | 4.0 MB | **1.311x** | 1.089x | **+20% Efficiency** |
| **AI Weights (Normal)** | 4.0 MB | **1.177x** | 1.078x | +9% Efficiency |
| **Sparse Tensors** | 4.0 MB | **2.149x** | 2.111x | +2% Efficiency |

### 2. Differential Savings (Fine-tuning Proof)
Storing fine-tuned models usually means copying the whole model. **NFC-Diff** stores only the numerical change.

| Strategy | Size on Disk | Ratio | Savings |
| :--- | :--- | :--- | :--- |
| **Normal (Zstd)** | 3,479.9 KB | 1.07x | Baseline |
| **NFC-Diff (Ours)** | **102.7 KB** | **34.00x** | **97.0% Savings** |

### 3. Deduplication (Infinite Scaling)
If model weights are identical across files, NFC stores them **once**.
- **Result:** Two identical 16MB models occupy **~460 bytes** of additional file space (pointers only).

---

## 🚀 Complete Feature List (v0.1 to v2.0)

### Foundation (v0.1 - v1.0)
- **Lossless Guarantee:** SHA256 payload verification for bit-perfect reconstruction.
- **Blosc Integration:** High-speed backend with selectable codecs (`zstd`, `lz4`, `lz4hc`, `zlib`).
- **Streaming Engine:** Process multi-gigabyte files without loading them into RAM.
- **True Random Access:** Indexed block framing allows decompressing specific tensors without reading the whole file.
- **Auto-tuning Heuristic:** Automatically detects data distribution and picks the best predictor (Delta vs None).
- **PyTorch Integration:** Direct `torch.Tensor` support with zero-copy potential.

### Intelligence & Integrity (v1.1 - v1.2)
- **NFC-Diff:** Numerical differential compression for checkpoints.
- **NFC-Peek:** Numerical metadata (Min, Max, Mean) stored in headers for instant analysis.
- **Safetensors Support:** Direct conversion from/to HuggingFace `.safetensors`.
- **Self-Healing:** Adler-32 header checksums to detect bit-rot/corruption instantly.

### Neural File System (v1.3 - v1.5)
- **Global Deduplication (CAS):** Content-Addressable Storage (`~/.nfc_store`) ensures unique weights are only stored once on your system.
- **Hardware Acceleration:** Numba JIT-optimized numerical predictors for near-native speed.
- **Lazy Loading (NFCLoader):** Dictionary-like interface to access model layers on-demand.
- **Virtual Merging:** Create new model configurations instantly using store pointers (Zero-copy ensemble/merging).
- **NFC-Explorer (TUI):** Interactive terminal dashboard to visualize model structure and stats.

### Distributed Swarm (v1.6 - v2.0)
- **Neural Forge Manifests (.nfm):** Tiny JSON "recipes" to share 100GB model structures as 10KB files.
- **Remote Sync (Registry):** Bidirectional sync with HTTP/Cloud registries.
- **Neural Registry Server:** Built-in lightweight server to host private block repositories.
- **Distributed Swarm Rehydration:** BitTorrent-style parallel fetching from multiple regional mirrors simultaneously.

---

## 🛠 Modular Usage Guide

NFC is designed to be modular. You can use only the features you need.

### Case A: Simple High-Speed Compression
If you just want faster/better compression for your tensors:
```python
from nfc_prototype.core import NFCPrototype
import torch

proto = NFCPrototype(codec='zstd')
tensor = torch.randn(1024, 1024)

# Compress
nfc_binary, _, _ = proto.compress_tensor(tensor, auto_tune=True)

# Decompress
original = proto.decompress_tensor(nfc_binary)
```

### Case B: Collaborative Model Sharing (Manifests)
If you want to share a model with a teammate without uploading 50GB:
```python
# 1. On your machine: Generate a recipe
manifest = proto.generate_manifest("my_model.nfc")
# Share manifest.json...

# 2. On teammate's machine: Rehydrate
# NFC will automatically check local store and configured remotes
proto = NFCPrototype(remotes=["http://your-registry:5000"])
proto.rehydrate_from_manifest(manifest, "reconstructed_model.nfc")
```

### Case C: Lazy-Loading for Inference
If you have a massive model and limited VRAM:
```python
from nfc_prototype.core import NFCLoader

loader = NFCLoader("huge_model.nfc")
# No data is decompressed yet.

# Fetch only the layer you need for execution:
layer_weights = loader["model.layers.31.weight"] 
```

### Case D: Differential Training Artifacts
If you are saving checkpoints during training:
```python
# Save only what changed since the last epoch
nfc_diff, _, _ = proto.compress_diff(epoch_10_weights, epoch_11_weights)
# Saves 97% disk space compared to a full save.
```

### Case E: Swarm Fetching (CLI)
Download a model from multiple mirrors at once:
```bash
python -m nfc_prototype.core prefetch model.nfm --remotes "http://us-mirror:5000,http://eu-mirror:5000"
```

---

## 🎯 Detailed Use Cases

### 1. Global Scale Model Distribution
A research lab releases a 100GB model. Instead of a single bottleneck download, they host it on multiple regional mirrors. Users using **NFC v2.0** can rehydrate the model by pulling chunks from the fastest/closest mirrors automatically in parallel.

### 2. Efficient Model Versioning & MLOps
Maintain dozens of fine-tuned versions of the same model using **NFC-CAS**. Since fine-tunes share 99% of base weights, every new version occupies only a few megabytes of unique space. 
*   **Impact:** Reduce model registry storage costs by 10x-20x.

### 3. Instant Model Merging & Ensemble Experiments
Using **Virtual Model Merging**, researchers can experiment with "weight averaging" or "layer swapping" without ever copying massive binary files. You create a "Virtual Model" (a tiny pointer file) that points to weight blocks already in your store.

### 4. Edge Device Deployment
For edge devices with limited storage, NFC's **Lazy Loading** allows weights to stay compressed on disk/RAM and only decompress into the GPU precisely when the execution reaches that specific layer.

### 5. Scientific Data Integrity (The Shield)
In mission-critical research (Genomics, Physics), NFC provides **Forensic Integrity**. With Adler-32 self-healing and SHA256 block hashes, you can prove that your multi-terabyte dataset has not suffered a single bit-flip over years of storage.

---

## 🗺 Roadmap Status

- [x] **v2.0:** Distributed Swarm Rehydration (Parallel Mirrors).
- [x] **v1.8:** Neural Registry & Built-in Registry Server.
- [x] **v1.7:** Neural Manifests (.nfm) & Rehydration.
- [x] **v1.6:** Hard Proof Benchmarking.
- [x] **v1.5:** NFCLoader (Lazy Loading) & Virtual Merging.
- [x] **v1.4:** Hardware acceleration (JIT/Numba).
- [x] **v1.3:** Global Deduplication (CAS) & TUI Explorer.
- [x] **v1.2:** NFC-Diff, NFC-Peek, and Self-Healing.
- [x] **v1.1:** Safetensors integration & CLI.
- [x] **v1.0:** Full AI engine & Auto-tune.

---
Licensed under MIT.
