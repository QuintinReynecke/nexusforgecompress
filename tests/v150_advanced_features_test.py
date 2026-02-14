import sys
import os
import numpy as np
import shutil
import torch
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nfc_prototype.core import NFCPrototype, NFCLoader

def test_advanced_features():
    print("Starting v1.5.0 Advanced Features Tests")
    print("=========================================")
    
    test_store_path = os.path.join(os.path.dirname(__file__), "test_store_adv")
    if os.path.exists(test_store_path):
        shutil.rmtree(test_store_path)
    
    proto = NFCPrototype(store_path=test_store_path)
    
    # 1. Test JIT-Accelerated Delta
    print("   - Testing Accelerated Delta Encoding...")
    data = np.arange(1000).astype(np.int32)
    nfc_bin, _, _ = proto.compress(data, use_prediction=True, prediction_strategy='delta')
    decomp = proto.decompress(nfc_bin)
    assert np.array_equal(data, decomp)
    print("     ✅ Passed: Accelerated Delta round-trip lossless.")

    # 2. Test NFCLoader (Lazy Loading)
    print("   - Testing NFCLoader (Lazy Loading)...")
    st_file = "test_loader.safetensors"
    nfc_file = "test_loader.nfc"
    
    # Create indexed NFC via safetensors
    import safetensors.numpy as st_np
    tensors = {
        "layer1": np.random.rand(10, 10).astype(np.float32),
        "layer2": np.random.rand(5, 5).astype(np.float32)
    }
    st_np.save_file(tensors, st_file)
    proto.compress_safetensors(st_file, nfc_file)
    
    loader = NFCLoader(nfc_file, proto=proto)
    assert "layer1" in loader.keys()
    assert "layer2" in loader.keys()
    
    # Lazy fetch
    l1 = loader["layer1"]
    assert np.allclose(tensors["layer1"], l1)
    print("     ✅ Passed: NFCLoader correctly retrieved layers by name.")

    # 3. Test Virtual Merging
    print("   - Testing Virtual Merging...")
    # Compress with dedup to get it into the store and get its hash
    nfc_l1, _, _ = proto.compress(tensors["layer1"], use_store=True)
    
    # Extract metadata and hashes
    meta_len = proto._extract_meta_len(nfc_l1)
    meta = json.loads(nfc_l1[34:34+meta_len].decode('utf-8'))
    h_store = nfc_l1[34+meta_len:34+meta_len+32].hex()
    h_orig = nfc_l1[34+meta_len+32 : 34+meta_len+32+32].hex()
    
    recipe = {
        "merged_layer": {
            "hash": h_store,
            "metadata": meta,
            "original_hash": h_orig
        }
    }
    proto.create_virtual_model(recipe, "virtual.nfc")
    
    # Load virtual model
    v_loader = NFCLoader("virtual.nfc", proto=proto)
    assert np.allclose(tensors["layer1"], v_loader["merged_layer"])
    print("     ✅ Passed: Virtual Merging successfully assembled model from pointers.")

    # Cleanup
    if os.path.exists(test_store_path):
        shutil.rmtree(test_store_path)
    for f in [st_file, nfc_file, "virtual.nfc"]:
        if os.path.exists(f):
            os.remove(f)

    print("All v1.5.0 Tests Passed!")

if __name__ == "__main__":
    test_advanced_features()
