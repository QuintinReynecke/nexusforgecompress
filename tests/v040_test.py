import sys
import os
import numpy as np
import hashlib
import time

# Add project root to path (prioritize local package)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nfc_prototype.core import NFCPrototype

def run_v040_tests():
    print("\nStarting v0.4.0 Prediction Tests")
    print("================================")
    
    proto = NFCPrototype()

    # --- Test Case 1: Basic Delta Encoding (smooth float data) ---
    print("\n   - Basic Delta Encoding (smooth float data)")
    original_data_float = np.linspace(0, 100, 1000).reshape(10, 100).astype(np.float32)
    
    # Test with prediction
    nfc_binary_pred_float, _, comp_size_pred_float = proto.compress(original_data_float, use_prediction=True)
    decompressed_data_pred_float = proto.decompress(nfc_binary_pred_float)
    
    assert np.allclose(original_data_float, decompressed_data_pred_float), "Float data with prediction not lossless"
    print(f"     ✅ Passed: Float data with prediction is lossless. Compressed size: {comp_size_pred_float} bytes.")
    
    # Compare with no prediction (optional, for ratio improvement check)
    nfc_binary_no_pred_float, _, comp_size_no_pred_float = proto.compress(original_data_float, use_prediction=False)
    if comp_size_pred_float < comp_size_no_pred_float:
        print(f"       Info: Prediction improved compression for float data ({comp_size_no_pred_float/comp_size_pred_float:.2f}x smaller)")
    else:
        print("       Info: Prediction did not improve compression for float data (or made it worse).")


    # --- Test Case 2: Delta Encoding with uint8 data ---
    print("\n   - Delta Encoding (uint8 data with wrap-around potential)")
    # Data that goes up and down to ensure signed residuals are handled
    original_data_uint8 = np.array([10, 12, 10, 5, 8, 20, 255, 250, 0, 5], dtype=np.uint8)

    nfc_binary_pred_uint8, _, comp_size_pred_uint8 = proto.compress(original_data_uint8, use_prediction=True)
    decompressed_data_pred_uint8 = proto.decompress(nfc_binary_pred_uint8)
    
    assert np.array_equal(original_data_uint8, decompressed_data_pred_uint8), "UInt8 data with prediction not lossless"
    print(f"     ✅ Passed: UInt8 data with prediction is lossless. Compressed size: {comp_size_pred_uint8} bytes.")

    # --- Test Case 3: Random data (prediction not expected to help) ---
    print("\n   - Random data (prediction not expected to help)")
    original_data_random = np.random.rand(100, 100).astype(np.float32)
    nfc_binary_pred_random, _, comp_size_pred_random = proto.compress(original_data_random, use_prediction=True)
    decompressed_data_pred_random = proto.decompress(nfc_binary_pred_random)
    
    assert np.allclose(original_data_random, decompressed_data_pred_random), "Random data with prediction not lossless"
    print(f"     ✅ Passed: Random data with prediction is lossless. Compressed size: {comp_size_pred_random} bytes.")
    
    nfc_binary_no_pred_random, _, comp_size_no_pred_random = proto.compress(original_data_random, use_prediction=False)
    if comp_size_pred_random < comp_size_no_pred_random:
        print(f"       Info: Prediction improved compression for random data ({comp_size_no_pred_random/comp_size_pred_random:.2f}x smaller)")
    else:
        print("       Info: Prediction did not improve compression for random data (as expected, or made it worse).")


    # --- Test Case 4: Edge case - empty array ---
    print("\n   - Edge case: Empty array")
    original_data_empty = np.array([], dtype=np.float32)
    nfc_binary_empty, _, comp_size_empty = proto.compress(original_data_empty, use_prediction=True)
    decompressed_data_empty = proto.decompress(nfc_binary_empty)
    assert np.array_equal(original_data_empty, decompressed_data_empty), "Empty array with prediction not lossless"
    print(f"     ✅ Passed: Empty array with prediction is lossless. Compressed size: {comp_size_empty} bytes.")

    print("\nAll v0.4.0 Prediction Tests Passed!")

if __name__ == "__main__":
    run_v040_tests()
