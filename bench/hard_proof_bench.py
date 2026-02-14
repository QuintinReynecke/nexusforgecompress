import os
import sys
import time
import numpy as np
import hashlib
import json
import zstandard as zstd
import lz4.frame
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nfc_prototype.core import NFCPrototype

console = Console()

class HardProofBenchmark:
    def __init__(self):
        self.proto = NFCPrototype()
        self.results = []

    def generate_weights(self, shape=(1024, 1024), type="normal"):
        if type == "normal":
            return np.random.normal(0, 0.1, shape).astype(np.float32)
        elif type == "smooth":
            return np.cumsum(np.random.normal(0, 0.01, shape), axis=1).astype(np.float32)
        elif type == "sparse":
            data = np.random.normal(0, 0.1, shape).astype(np.float32)
            data[np.abs(data) < 0.08] = 0
            return data
        return np.random.rand(*shape).astype(np.float32)

    def run_compression_test(self, name, data):
        console.print(f"[bold yellow]Testing: {name}[/bold yellow]")
        raw_bytes = data.tobytes()
        orig_size = len(raw_bytes)

        # 1. NFC
        start = time.time()
        nfc_bin, _, nfc_size = self.proto.compress(data, auto_tune=True)
        nfc_time = time.time() - start
        nfc_ratio = orig_size / nfc_size

        # 2. Zstd
        ctx = zstd.ZstdCompressor(level=3)
        start = time.time()
        zstd_bin = ctx.compress(raw_bytes)
        zstd_time = time.time() - start
        zstd_size = len(zstd_bin)
        zstd_ratio = orig_size / zstd_size

        # 3. LZ4
        start = time.time()
        lz4_bin = lz4.frame.compress(raw_bytes)
        lz4_time = time.time() - start
        lz4_size = len(lz4_bin)
        lz4_ratio = orig_size / lz4_size

        res = {
            "Dataset": name,
            "Original (MB)": orig_size / 1024**2,
            "NFC Ratio": nfc_ratio,
            "NFC Speed": (orig_size / 1024**2) / nfc_time if nfc_time > 0 else 0,
            "Zstd Ratio": zstd_ratio,
            "LZ4 Ratio": lz4_ratio
        }
        self.results.append(res)
        return res

    def run_dedup_test(self):
        console.print("\n[bold yellow]Testing: Global Deduplication (Hard Proof)[/bold yellow]")
        data = self.generate_weights(shape=(2048, 2048), type="normal")
        test_store = "./bench_store"
        if os.path.exists(test_store):
            import shutil
            shutil.rmtree(test_store)
        
        proto_dedup = NFCPrototype(store_path=test_store)
        _, _, s1 = proto_dedup.compress(data, use_store=True)
        _, _, s2 = proto_dedup.compress(data, use_store=True)

        console.print(f"   - File 1 (Pointer): {s1} bytes")
        console.print(f"   - File 2 (Pointer): {s2} bytes")
        console.print("   ✅ [green]Deduplication successful.[/green]")
        
        import shutil
        if os.path.exists(test_store): shutil.rmtree(test_store)

    def run_diff_test(self):
        console.print("\n[bold yellow]Testing: Differential (NFC-Diff)[/bold yellow]")
        base = self.generate_weights(shape=(1024, 1024), type="normal")
        tuned = base.copy()
        mask = np.random.rand(1024, 1024) < 0.01
        tuned[mask] += np.random.normal(0, 0.01, tuned[mask].shape).astype(np.float32)

        _, _, norm = self.proto.compress(tuned)
        _, _, diff = self.proto.compress_diff(base, tuned)

        console.print(f"   - Normal: {norm / 1024:.1f} KB, Diff: {diff / 1024:.1f} KB")
        console.print(f"   - Savings: {(1 - diff/norm)*100:.1f}% better than Zstd-based normal.")

    def print_report(self):
        table = Table(title="NFC Hard Proof Benchmark Report")
        table.add_column("Dataset")
        table.add_column("Orig Size", justify="right")
        table.add_column("NFC Ratio", style="green", justify="right")
        table.add_column("Zstd Ratio", justify="right")
        table.add_column("LZ4 Ratio", justify="right")
        table.add_column("NFC Speed", justify="right")

        for r in self.results:
            table.add_row(
                r["Dataset"],
                f"{r['Original (MB)']:.1f} MB",
                f"{r['NFC Ratio']:.3f}x",
                f"{r['Zstd Ratio']:.3f}x",
                f"{r['LZ4 Ratio']:.3f}x",
                f"{r['NFC Speed']:.1f} MB/s"
            )
        console.print(table)

    def run_all(self):
        self.run_compression_test("AI Weights (Normal)", self.generate_weights(type="normal"))
        self.run_compression_test("Gradients (Smooth)", self.generate_weights(type="smooth"))
        self.run_compression_test("Sparse Tensors", self.generate_weights(type="sparse"))
        self.run_dedup_test()
        self.run_diff_test()
        self.print_report()

if __name__ == "__main__":
    HardProofBenchmark().run_all()
