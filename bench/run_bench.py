import numpy as np
from nfc_prototype.core import NFCPrototype
import snappy  # Baseline
import time

def bench_compress(proto, data, runs=3):
    times = []
    for _ in range(runs):
        start = time.time()
        nfc_binary, orig_size, comp_size = proto.compress(data)
        times.append(time.time() - start)
    ratio = orig_size / comp_size
    print(f"NFC Ratio: {ratio:.2f}x, Avg Time: {sum(times)/runs:.4f}s")

def bench_snappy(data):
    data_bytes = data.tobytes()
    compressed = snappy.compress(data_bytes)
    ratio = len(data_bytes) / len(compressed)
    print(f"Snappy Ratio: {ratio:.2f}x")

if __name__ == "__main__":
    data = np.random.rand(10000, 10000).astype(np.float32)  # ~400MB
    proto = NFCPrototype()
    bench_compress(proto, data)
    bench_snappy(data)
    # Add zstd if installed: import zstd; ...