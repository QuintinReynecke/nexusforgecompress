import sys
import os
import numpy as np
import hashlib
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nfc_prototype.core import NFCPrototype

def test_parallel_streaming():
    print("Starting v1.0.1 Parallel Streaming Tests")
    print("=========================================")
    
    proto = NFCPrototype()
    test_file = "test_parallel.bin"
    nfc_file = "test_parallel.nfc"
    decomp_file = "test_parallel_decomp.bin"
    
    # Create a 50MB file
    size_mb = 50
    print(f"   - Creating {size_mb}MB test file...")
    data = np.random.rand(size_mb * 1024 * 1024 // 8).astype(np.float64)
    data.tofile(test_file)
    original_hash = hashlib.sha256(open(test_file, 'rb').read()).hexdigest()
    
    # Parallel Compression
    print(f"   - Compressing with 4 workers...")
    start = time.time()
    proto.compress_stream_parallel(test_file, nfc_file, chunk_size=5*1024*1024, max_workers=4)
    print(f"     Compression took {time.time() - start:.2f}s")
    
    # Parallel Decompression
    print(f"   - Decompressing with 4 workers...")
    start = time.time()
    proto.decompress_stream_parallel(nfc_file, decomp_file, max_workers=4)
    print(f"     Decompression took {time.time() - start:.2f}s")
    
    # Verify
    decomp_hash = hashlib.sha256(open(decomp_file, 'rb').read()).hexdigest()
    assert original_hash == decomp_hash
    print("     ✅ Passed: Parallel streaming round-trip lossless.")
    
    # Cleanup
    for f in [test_file, nfc_file, decomp_file]:
        if os.path.exists(f):
            os.remove(f)
            
    print("All v1.0.1 Parallel Streaming Tests Passed!")

if __name__ == "__main__":
    test_parallel_streaming()
