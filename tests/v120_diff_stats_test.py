import sys
import os
import numpy as np
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nfc_prototype.core import NFCPrototype

def test_diff_and_stats():
    print("Starting v1.2.0 Differential & Stats Tests")
    print("==========================================")
    
    proto = NFCPrototype()
    
    # 1. Test Stats collection
    print("   - Testing Stats Collection...")
    data = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    nfc_binary, _, _ = proto.compress(data)
    stats = proto.get_stats(nfc_binary)
    
    print(f"     Stats: {stats}")
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["mean"] == 30.0
    print("     ✅ Passed: Stats correctly collected and extracted.")

    # 2. Test Differential Compression
    print("   - Testing Differential Compression (NFC-Diff)...")
    base_model = np.random.randn(100, 100).astype(np.float32)
    # Simulate a fine-tuned model (base + small noise)
    fine_tuned = base_model + (np.random.randn(100, 100) * 0.01).astype(np.float32)
    
    nfc_diff, _, diff_size = proto.compress_diff(base_model, fine_tuned)
    
    # Also compress fine_tuned normally for comparison
    _, _, normal_size = proto.compress(fine_tuned)
    
    print(f"     Normal size: {normal_size}, Diff size: {diff_size}")
    # Note: For random noise, diff might not always be smaller unless the noise is very sparse or small,
    # but the logic should be lossless regardless.
    
    reconstructed = proto.decompress_diff(base_model, nfc_diff)
    
    assert np.allclose(fine_tuned, reconstructed, atol=1e-6)
    print("     ✅ Passed: Differential compression is lossless.")

    # 3. Test CLI Stats
    print("   - Testing CLI Stats command...")
    test_file = "test_stats.nfc"
    with open(test_file, "wb") as f:
        f.write(nfc_binary)
    
    # Run CLI command via shell
    import subprocess
    result = subprocess.run([sys.executable, "-m", "nfc_prototype.core", "stats", test_file], capture_output=True, text=True)
    print(f"     CLI Output: {result.stdout.strip()}")
    assert "30.0" in result.stdout
    
    if os.path.exists(test_file):
        os.remove(test_file)
    print("     ✅ Passed: CLI stats command working.")

    print("All v1.2.0 Tests Passed!")

if __name__ == "__main__":
    test_diff_and_stats()
