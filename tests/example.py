import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from nfc_prototype.core import NFCPrototype
import os

def main():
    """
    Runs a self-contained test of the NFCPrototype compression and decompression.
    """
    print("NFC-Prototype Verification Script")
    print("="*35)

    proto = NFCPrototype()
    
    # 1. Create a sample tensor
    print("1. Creating a sample 1024x1024 float32 tensor...")
    original_tensor = np.random.rand(1024, 1024).astype(np.float32)
    print(f"   Original size: {original_tensor.nbytes / 1024**2:.2f} MB")

    # 2. Compress the tensor in memory
    print("\n2. Compressing tensor...")
    nfc_binary, orig_size, comp_size = proto.compress(original_tensor)
    ratio = orig_size / comp_size
    print(f"   Compressed size: {comp_size / 1024**2:.2f} MB")
    print(f"   Compression ratio: {ratio:.2f}x")

    # 3. Decompress the tensor in memory
    print("\n3. Decompressing tensor...")
    decompressed_tensor = proto.decompress(nfc_binary)
    print("   Decompression complete.")

    # 4. Verify the result
    print("\n4. Verifying data integrity...")
    if np.allclose(original_tensor, decompressed_tensor):
        print("\n✅ SUCCESS: Decompressed tensor is identical to the original.")
    else:
        print("\n❌ FAILURE: Data corruption detected. Tensors do not match.")
        exit(1)

if __name__ == "__main__":
    main()
