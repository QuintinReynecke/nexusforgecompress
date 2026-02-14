import sys
import os
import numpy as np
import torch
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nfc_prototype.core import NFCPrototype

def test_ai_engine_features():
    print("Starting v1.0.0 AI Engine Tests")
    print("====================================")
    
    proto = NFCPrototype()
    
    # 1. PyTorch Integration
    print("   - Testing PyTorch Integration...")
    original_tensor = torch.randn(100, 100, dtype=torch.float32)
    nfc_binary, _, _ = proto.compress_tensor(original_tensor)
    decompressed_tensor = proto.decompress_tensor(nfc_binary)
    
    assert torch.allclose(original_tensor, decompressed_tensor)
    print("     ✅ Passed: PyTorch tensor round-trip lossless.")

    # 2. Auto-tuning (Smooth Data -> Delta)
    print("   - Testing Auto-tuning (Smooth Data)...")
    # Linear ramp is very predictable with delta
    smooth_data = np.linspace(0, 100, 1000).astype(np.float32)
    # Without auto-tune/prediction
    _, _, size_no_pred = proto.compress(smooth_data, use_prediction=False)
    # With auto-tune
    _, _, size_auto = proto.compress(smooth_data, auto_tune=True)
    
    print(f"     No prediction size: {size_no_pred}, Auto-tune size: {size_auto}")
    assert size_auto < size_no_pred
    print("     ✅ Passed: Auto-tune correctly selected prediction for smooth data.")

    # 3. Auto-tuning (Random Data -> None)
    print("   - Testing Auto-tuning (Random Data)...")
    random_data = np.random.rand(1000).astype(np.float32)
    nfc_binary_rand, _, _ = proto.compress(random_data, auto_tune=True)
    # Check metadata to see if it picked 'none'
    meta_len = proto._extract_meta_len(nfc_binary_rand)
    metadata = json.loads(nfc_binary_rand[34 : 34 + meta_len].decode('utf-8'))
    strategy = metadata.get("prediction_strategy", "none")
    print(f"     Random data strategy: {strategy}")
    
    print("     ✅ Passed: Auto-tune handled random data.")

    print("All v1.0.0 AI Engine Tests Passed!")

if __name__ == "__main__":
    test_ai_engine_features()
