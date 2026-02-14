import sys
import os
import numpy as np
import safetensors.numpy as st_np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nfc_prototype.core import NFCPrototype

def test_safetensors():
    print("Starting v1.1.0 Safetensors Tests")
    print("====================================")
    
    proto = NFCPrototype()
    st_file = "test_model.safetensors"
    nfc_file = "test_model.nfc"
    restored_st_file = "restored_model.safetensors"
    
    # 1. Create a dummy safetensors file
    print("   - Creating dummy .safetensors file...")
    data = {
        "weight1": np.random.rand(10, 10).astype(np.float32),
        "bias1": np.random.rand(10).astype(np.float32),
        "weight2": np.random.rand(20, 20).astype(np.float64)
    }
    st_np.save_file(data, st_file)
    
    # 2. Compress
    print("   - Compressing .safetensors to .nfc...")
    proto.compress_safetensors(st_file, nfc_file, auto_tune=True)
    
    # 3. Decompress
    print("   - Restoring .safetensors from .nfc...")
    proto.decompress_safetensors(nfc_file, restored_st_file)
    
    # 4. Verify
    print("   - Verifying restored data...")
    restored_data = st_np.load_file(restored_st_file)
    for name, original_arr in data.items():
        assert np.allclose(original_arr, restored_data[name])
        print(f"     ✅ Passed: Tensor '{name}' restored correctly.")
    
    # Cleanup
    for f in [st_file, nfc_file, restored_st_file]:
        if os.path.exists(f):
            os.remove(f)
            
    print("All v1.1.0 Safetensors Tests Passed!")

if __name__ == "__main__":
    test_safetensors()
