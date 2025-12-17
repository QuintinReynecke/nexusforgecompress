import sys
import os
import numpy as np
import hashlib

# Add project root to path (prioritize local package)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nfc_prototype.core import NFCPrototype
import time

# Configuration
TEST_DIR = "test_data"
ORIGINAL_LARGE_FILE = os.path.join(TEST_DIR, "large_original.bin")
COMPRESSED_LARGE_FILE = os.path.join(TEST_DIR, "large_compressed.nfc")
DECOMPRESSED_LARGE_FILE = os.path.join(TEST_DIR, "large_decompressed.bin")
FILE_SIZE_MB = 200 # 200 MB for testing streaming, can be adjusted
FILE_SIZE_BYTES = FILE_SIZE_MB * 1024 * 1024

def generate_large_dummy_file(filepath, size_bytes):
    print(f"Generating large dummy file: {filepath} ({size_bytes / (1024*1024):.2f} MB)...")
    # Generate data that compresses reasonably well
    data = np.arange(size_bytes // np.dtype(np.float32).itemsize, dtype=np.float32)
    # If the size is not perfectly divisible, trim or pad
    if data.nbytes < size_bytes:
        # Pad with zeros if data is smaller
        data = np.pad(data, (0, (size_bytes - data.nbytes) // np.dtype(np.float32).itemsize), 'constant')
    elif data.nbytes > size_bytes:
        # Trim if data is larger
        data = data[:size_bytes // np.dtype(np.float32).itemsize]
    
    data.tofile(filepath)
    print("Generation complete.")

def calculate_file_hash(filepath, hash_algo='sha256'):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def test_streaming_large_file():
    print("\n--- Starting v0.3.0 Streaming Tests ---")
    
    # Setup
    os.makedirs(TEST_DIR, exist_ok=True)
    nfc_prototype = NFCPrototype()

    try:
        # 1. Generate a large dummy file
        generate_large_dummy_file(ORIGINAL_LARGE_FILE, FILE_SIZE_BYTES)
        original_hash = calculate_file_hash(ORIGINAL_LARGE_FILE)
        print(f"Original file hash: {original_hash}")

        # 2. Compress the large dummy file using compress_stream
        print(f"Compressing {ORIGINAL_LARGE_FILE} to {COMPRESSED_LARGE_FILE}...")
        start_time = time.time()
        nfc_prototype.compress_stream(ORIGINAL_LARGE_FILE, COMPRESSED_LARGE_FILE)
        compress_duration = time.time() - start_time
        print(f"Compression complete in {compress_duration:.2f} seconds.")

        # 3. Decompress the NFC compressed file using the refactored decompress_stream
        print(f"Decompressing {COMPRESSED_LARGE_FILE} to {DECOMPRESSED_LARGE_FILE}...")
        start_time = time.time()
        nfc_prototype.decompress_stream(COMPRESSED_LARGE_FILE, DECOMPRESSED_LARGE_FILE)
        decompress_duration = time.time() - start_time
        print(f"Decompression complete in {decompress_duration:.2f} seconds.")

        # 4. Verify integrity
        decompressed_hash = calculate_file_hash(DECOMPRESSED_LARGE_FILE)
        print(f"Decompressed file hash: {decompressed_hash}")

        assert original_hash == decompressed_hash, "File integrity check failed: Hashes do not match!"
        print("✅ Passed: Large file streaming compression/decompression is lossless.")

    finally:
        # 5. Clean up
        print("Cleaning up test files...")
        if os.path.exists(ORIGINAL_LARGE_FILE):
            os.remove(ORIGINAL_LARGE_FILE)
        if os.path.exists(COMPRESSED_LARGE_FILE):
            os.remove(COMPRESSED_LARGE_FILE)
        if os.path.exists(DECOMPRESSED_LARGE_FILE):
            os.remove(DECOMPRESSED_LARGE_FILE)
        if os.path.exists(TEST_DIR):
            try:
                os.rmdir(TEST_DIR) # Only removes if empty
            except OSError as e:
                print(f"Could not remove test directory {TEST_DIR}: {e}. It might not be empty.")
        print("Cleanup complete.")

def test_codec_selection():
    print("\n--- Starting v0.3.0 Codec Selection Tests ---")
    
    # Setup
    test_data = np.random.rand(100, 100).astype(np.float32) # Small data for quick codec tests
    codecs_to_test = ['zstd', 'lz4', 'lz4hc', 'zlib'] # Common Blosc codecs

    all_passed = True
    for codec in codecs_to_test:
        print(f"\nTesting with codec: {codec}")
        try:
            nfc_prototype = NFCPrototype(codec=codec)
            
            # Compress
            nfc_binary, original_size, compressed_size = nfc_prototype.compress(test_data)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
            print(f"  Compression Ratio ({codec}): {compression_ratio:.2f}x")

            # Decompress
            decompressed_data = nfc_prototype.decompress(nfc_binary)

            # Verify integrity
            assert np.allclose(test_data, decompressed_data), f"Data mismatch for codec {codec}!"
            print(f"  ✅ Passed: Codec {codec} compression/decompression is lossless.")
        except Exception as e:
            print(f"  ❌ Failed: Codec {codec} test failed with error: {e}")
            all_passed = False
    
    assert all_passed, "One or more codec selection tests failed!"
    print("\n--- All v0.3.0 Codec Selection Tests Passed! ---")

if __name__ == "__main__":
    test_streaming_large_file()
    test_codec_selection()
    print("\nAll v0.3.0 Tests Passed!")

