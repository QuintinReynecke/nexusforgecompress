import numpy as np
import hashlib
from nfc_prototype.core import NFCPrototype

def test_roundtrip():
    proto = NFCPrototype()
    original = np.random.rand(1000, 1000).astype(np.float32)
    nfc_binary, _, _ = proto.compress(original)
    decompressed = proto.decompress(nfc_binary)
    assert np.allclose(original, decompressed)

def test_corruption():
    proto = NFCPrototype()
    original = np.arange(100).astype(np.int16)
    nfc_binary, _, _ = proto.compress(original)
    # Flip bit
    corrupted = bytearray(nfc_binary)
    corrupted[-10] ^= 0xFF
    try:
        proto.decompress(bytes(corrupted))
        assert False, "Should detect corruption"
    except ValueError:
        pass

def test_dtypes():
    dtypes = [np.float16, np.bfloat16, np.float32, np.int8, np.uint8]
    proto = NFCPrototype()
    for dt in dtypes:
        original = np.arange(10).astype(dt)
        nfc_binary, _, _ = proto.compress(original)
        decompressed = proto.decompress(nfc_binary)
        assert np.allclose(original, decompressed, atol=1e-6 if 'float' in str(dt) else 0)

if __name__ == "__main__":
    test_roundtrip()
    test_corruption()
    test_dtypes()
    print("All tests passed!")