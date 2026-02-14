import sys
import os
import numpy as np
import json
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nfc_prototype.core import NFCPrototype, NFCLoader

def test_manifests():
    print("Starting v1.7.0 Neural Manifest Tests")
    print("======================================")
    
    test_store = "./test_store_manifest"
    if os.path.exists(test_store): shutil.rmtree(test_store)
    proto = NFCPrototype(store_path=test_store)
    
    # 1. Create a model with multiple layers
    print("   - Creating source model...")
    data = {
        "weights": np.random.rand(100, 100).astype(np.float32),
        "bias": np.random.rand(100).astype(np.float32)
    }
    import safetensors.numpy as st_np
    st_np.save_file(data, "source.safetensors")
    proto.compress_safetensors("source.safetensors", "source.nfc", use_store=True)
    
    # 2. Generate Manifest
    print("   - Generating Manifest (.nfm)...")
    manifest = proto.generate_manifest("source.nfc")
    assert len(manifest["blocks"]) == 2
    names = [b["name"] for b in manifest["blocks"]]
    assert "weights" in names
    assert "bias" in names
    print("     ✅ Passed: Manifest generated with correct structure.")
    
    # 3. Rehydrate from Manifest
    print("   - Rehydrating from Manifest...")
    proto.rehydrate_from_manifest(manifest, "rehydrated.nfc")
    
    # 4. Verify rehydrated model
    loader = NFCLoader("rehydrated.nfc", proto=proto)
    assert np.allclose(data["weights"], loader["weights"])
    assert np.allclose(data["bias"], loader["bias"])
    print("     ✅ Passed: Rehydrated model is identical to source.")
    
    # Cleanup
    if os.path.exists(test_store): shutil.rmtree(test_store)
    for f in ["source.safetensors", "source.nfc", "rehydrated.nfc"]:
        if os.path.exists(f): os.remove(f)
        
    print("All v1.7.0 Tests Passed!")

if __name__ == "__main__":
    test_manifests()
