import sys
import os
import numpy as np
import hashlib
import blosc
import datetime
import json

# Add the parent directory to sys.path to allow importing nfc_prototype
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nfc_prototype.core import NFCPrototype

def calculate_hash(data):
    if isinstance(data, np.ndarray):
        return hashlib.sha256(data.tobytes()).hexdigest()
    return hashlib.sha256(data).hexdigest()

def run_v050_tests():
    print("\nStarting v0.5.0 Prediction & Blosc Filter Tests")
    print("=================================================")
    proto = NFCPrototype()

    # --- Test 1: Dynamic created_at timestamp ---
    print("   - Dynamic created_at timestamp")
    test_data_ts = np.array([1, 2, 3], dtype=np.int32)
    nfc_binary_ts, _, _ = proto.compress(test_data_ts)
    
    # Extract metadata using helper methods
    meta_len = proto._extract_meta_len(nfc_binary_ts)
    metadata_json = nfc_binary_ts[proto.calculated_header_len : proto.calculated_header_len + meta_len]
    metadata_ts = json.loads(metadata_json)

    try:
        created_at_str = metadata_ts.get("created_at")
        assert created_at_str is not None
        # Try to parse it to confirm it's a valid ISO format
        datetime.datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
        print("     ✅ Passed: created_at is a dynamic, valid ISO timestamp.")
    except Exception as e:
        print(f"     ❌ Failed: created_at is not dynamic or invalid. Error: {e}")

    # --- Test 2: prediction_strategy='none' ---
    print("\n   - Prediction Strategy: 'none'")
    test_data_none = np.random.rand(10, 100).astype(np.float32)
    original_hash_none = calculate_hash(test_data_none)
    nfc_binary_none, orig_size_none, comp_size_none = proto.compress(test_data_none, use_prediction=True, prediction_strategy='none')
    decompressed_data_none = proto.decompress(nfc_binary_none)
    decompressed_hash_none = calculate_hash(decompressed_data_none)

    if original_hash_none == decompressed_hash_none:
        print(f"     ✅ Passed: Data compressed with 'none' prediction is lossless. Compressed size: {comp_size_none} bytes.")
    else:
        print(f"     ❌ Failed: Data compressed with 'none' prediction is NOT lossless. Original hash: {original_hash_none}, Decompressed hash: {decompressed_hash_none}")

    # --- Test 3: prediction_strategy='delta' (re-test for regression) ---
    print("\n   - Prediction Strategy: 'delta' (Regression Test)")
    test_data_delta = np.array([i * 0.1 for i in range(100)]).reshape(10, 10).astype(np.float32)
    original_hash_delta = calculate_hash(test_data_delta)
    nfc_binary_delta, orig_size_delta, comp_size_delta = proto.compress(test_data_delta, use_prediction=True, prediction_strategy='delta')
    decompressed_data_delta = proto.decompress(nfc_binary_delta)
    decompressed_hash_delta = calculate_hash(decompressed_data_delta)

    if original_hash_delta == decompressed_hash_delta:
        print(f"     ✅ Passed: Data compressed with 'delta' prediction is lossless. Compressed size: {comp_size_delta} bytes.")
    else:
        print(f"     ❌ Failed: Data compressed with 'delta' prediction is NOT lossless. Original hash: {original_hash_delta}, Decompressed hash: {decompressed_hash_delta}")

    # --- Test 4: blosc_filters parameter ---
    print("\n   - Blosc Filters Parameter")
    test_data_filters = np.arange(1000).reshape(10, 100).astype(np.int32)
    original_hash_filters = calculate_hash(test_data_filters)
    
    # Test with a specific filter (e.g., BLOSCDELTA)
    try:
        nfc_binary_filters, _, comp_size_filters = proto.compress(test_data_filters, blosc_filters=[4]) # 4 is BLOSC_DELTA
        decompressed_data_filters = proto.decompress(nfc_binary_filters)
        decompressed_hash_filters = calculate_hash(decompressed_data_filters)

        if original_hash_filters == decompressed_hash_filters:
            print(f"     ✅ Passed: Data compressed with custom Blosc filter (BLOSCDELTA) is lossless. Compressed size: {comp_size_filters} bytes.")
            # Check metadata for filter
            meta_len_f = proto._extract_meta_len(nfc_binary_filters)
            metadata_json_f = nfc_binary_filters[proto.calculated_header_len : proto.calculated_header_len + meta_len_f]
            metadata_f = json.loads(metadata_json_f)
            assert metadata_f.get("blosc_filters") == [4]
            print("     ✅ Passed: Blosc filters stored in metadata.")
        else:
            print(f"     ❌ Failed: Data compressed with custom Blosc filter (BLOSCDELTA) is NOT lossless. Original hash: {original_hash_filters}, Decompressed hash: {decompressed_hash_filters}")
    except Exception as e:
        print(f"     ❌ Failed: Blosc filters test failed. Error: {e}")
    
    print("\nAll v0.5.0 Tests Finished.")

if __name__ == '__main__':
    run_v050_tests()
