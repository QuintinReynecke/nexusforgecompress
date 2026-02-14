import sys
import os
import numpy as np
import hashlib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nfc_prototype.core import NFCPrototype

def test_random_access():
    print("\nStarting v0.3.1 Random Access Tests")
    print("====================================")
    
    proto = NFCPrototype()
    test_file = "test_random.bin"
    nfc_file = "test_random.nfc"
    
    # 1. Create a file with 3 distinct blocks of equal size
    print("   - Creating test file with 3 blocks...")
    block1 = b"A" * 100
    block2 = b"B" * 100
    block3 = b"C" * 100
    
    with open(test_file, "wb") as f:
        f.write(block1)
        f.write(block2)
        f.write(block3)
    
    # 2. Compress with chunk size matching block size
    print("   - Compressing with chunk_size=100...")
    proto.compress_stream(test_file, nfc_file, chunk_size=100)
    
    # 3. Test decompress_stream (sequential) still works
    print("   - Testing sequential decompress_stream...")
    decomp_file = "test_random_decomp.bin"
    proto.decompress_stream(nfc_file, decomp_file)
    with open(decomp_file, "rb") as f:
        content = f.read()
    assert content == block1 + block2 + block3
    print("     ✅ Passed: Sequential decompression still works and skips index.")
    
    # 4. Test random access
    print("   - Testing random access to block 0...")
    res0 = proto.decompress_block(nfc_file, 0)
    assert res0 == block1
    print("     ✅ Passed: Random access to block 0 successful.")
    
    print("   - Testing random access to block 2...")
    res2 = proto.decompress_block(nfc_file, 2)
    assert res2 == block3
    print("     ✅ Passed: Random access to block 2 successful.")
    
    # 5. Test get_stream_index
    print("   - Testing get_stream_index...")
    index_info = proto.get_stream_index(nfc_file)
    if isinstance(index_info, dict):
        index = index_info["offsets"]
    else:
        index = index_info
    assert len(index) == 3
    print(f"     ✅ Passed: Index found with {len(index)} blocks.")
    
    # Cleanup
    for f in [test_file, nfc_file, decomp_file]:
        if os.path.exists(f):
            os.remove(f)
            
    print("\nAll v0.3.1 Random Access Tests Passed!")

if __name__ == "__main__":
    test_random_access()
