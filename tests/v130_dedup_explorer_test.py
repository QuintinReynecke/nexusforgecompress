import sys
import os
import numpy as np
import shutil
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nfc_prototype.core import NFCPrototype
from nfc_prototype.explorer import NFCExplorer

def test_dedup_and_explorer():
    print("Starting v1.3.0 Deduplication & Explorer Tests")
    print("==============================================")
    
    # Setup custom store path for testing to avoid polluting user's home
    test_store_path = os.path.join(os.path.dirname(__file__), "test_store")
    if os.path.exists(test_store_path):
        shutil.rmtree(test_store_path)
    
    proto = NFCPrototype(store_path=test_store_path)
    
    # 1. Create specific test data
    data = np.random.rand(100, 100).astype(np.float32)
    original_size = data.nbytes
    
    # 2. Compress with embedding (Standard)
    print("   - Compressing Block A (Standard)...")
    nfc_standard, _, size_standard = proto.compress(data, use_store=False)
    
    # 3. Compress with deduplication (Store)
    print("   - Compressing Block A (Dedup)...")
    nfc_dedup, _, size_dedup = proto.compress(data, use_store=True)
    
    print(f"     Standard Size: {size_standard} bytes")
    print(f"     Dedup Size: {size_dedup} bytes")
    
    # Dedup size should be smaller because payload is just 32 bytes hash
    assert size_dedup < size_standard
    print("     ✅ Passed: Deduplication reduced file size significantly.")
    
    # 4. Decompress from store
    print("   - Decompressing deduped block...")
    decomp_data = proto.decompress(nfc_dedup)
    assert np.allclose(data, decomp_data)
    print("     ✅ Passed: Deduplicated block restored correctly.")

    # 5. Explorer Test (Smoke Test)
    print("   - Testing Explorer initialization...")
    # Write raw data to a file first
    raw_data_path = "test_raw.bin"
    with open(raw_data_path, "wb") as f:
        f.write(data.tobytes())
    
    # Compress stream with dedup
    proto.compress_stream(raw_data_path, "test_explore_stream.nfc", chunk_size=1024*1024, use_store=True)
    
    try:
        exp = NFCExplorer("test_explore_stream.nfc")
        # Run summary (non-interactive)
        exp.show_summary()
        print("     ✅ Passed: Explorer loaded and showed summary.")
    except Exception as e:
        print(f"     ❌ Failed: Explorer crashed: {e}")
        raise

    # Cleanup
    if os.path.exists(test_store_path):
        shutil.rmtree(test_store_path)
    for f in ["test_explore.nfc", "test_explore_stream.nfc", raw_data_path]:
        if os.path.exists(f):
            os.remove(f)

    print("All v1.3.0 Tests Passed!")

if __name__ == "__main__":
    test_dedup_and_explorer()
