import numpy as np
import zipnn
from zipnn import ZipNN
import json
import hashlib
import struct
import io
import os

try:
    import neural_compression  # Hook for future entropy (v0.2)
except ImportError:
    neural_compression = None  # Fallback: Skip if not installed

class NFCPrototype:
    def __init__(self):
        self.magic = b'NFC1'  # 4 bytes
        self.hash_algo = 'sha256'  # Fixed
    
    def _get_metadata(self, data):
        if isinstance(data, np.ndarray):
            return {
                "schema_version": "nfc-0.1",
                "dtype": data.dtype.name,
                "endianness": data.dtype.byteorder,
                "shape": list(data.shape),
                "format_hint": "numpy_tensor",
                "orig_bytes": data.nbytes,
                "created_by": "nfc-prototype 0.1.0",
                "created_at": "2025-12-03T00:00:00Z"  # Placeholder
            }
        return {}
    
    def compress(self, data):
        data_bytes = data.tobytes() if isinstance(data, np.ndarray) else data
        metadata_json = json.dumps(self._get_metadata(data)).encode('utf-8')
        
        # Compress with ZipNN
        zipnn_instance = zipnn.ZipNN()
        compressed = zipnn_instance.compress(data_bytes)
        
        # Raw hash
        original_hash = hashlib.sha256(data_bytes).digest()  # 32 bytes
        
        # Pack binary: magic(4) + version(1) + flags(1) + reserved(2) + header_len(8) + meta_len(8) + payload_len(8) + hash_len(2) + meta + payload + hash
        header = (
            self.magic +
            b'\x01' +  # Version
            b'\x00' +  # Flags
            b'\x00\x00' +  # Reserved
            struct.pack('!Q', 24) +  # Header len (fixed for v0.1)
            struct.pack('!Q', len(metadata_json)) +
            struct.pack('!Q', len(compressed)) +
            struct.pack('!H', len(original_hash))
        )
        nfc_binary = header + metadata_json + compressed + original_hash
        
        return nfc_binary, len(data_bytes), len(nfc_binary)
    
    def decompress(self, nfc_binary):
        if nfc_binary[:4] != self.magic:
            raise ValueError("Invalid magic")
        
        version, flags = nfc_binary[4:5], nfc_binary[5:6]
        reserved = nfc_binary[6:8]
        header_len = struct.unpack('!Q', nfc_binary[8:16])[0]
        meta_len = struct.unpack('!Q', nfc_binary[16:24])[0]
        payload_len = struct.unpack('!Q', nfc_binary[24:32])[0]
        hash_len = struct.unpack('!H', nfc_binary[32:34])[0]
        
        metadata_json = nfc_binary[34:34+meta_len]
        compressed = nfc_binary[34+meta_len:34+meta_len+payload_len]
        original_hash = nfc_binary[34+meta_len+payload_len:34+meta_len+payload_len+hash_len]
        
        zipnn_instance = zipnn.ZipNN()
        decompressed_bytes = zipnn_instance.decompress(compressed)
        decompressed_hash = hashlib.sha256(decompressed_bytes).digest()
        if decompressed_hash != original_hash:
            raise ValueError("Corruption detected!")
        
        metadata = json.loads(metadata_json) if meta_len > 0 else {}
        if metadata:
            dtype = np.dtype(metadata["dtype"])
            if metadata["endianness"] != dtype.byteorder:
                dtype = dtype.newbyteorder(metadata["endianness"])
            return np.frombuffer(decompressed_bytes, dtype=dtype).reshape(metadata["shape"])
        return decompressed_bytes
    
    # Streaming APIs (file-based, chunked)
    def compress_stream(self, in_path, out_path, chunk_size=1024*1024*64):  # 64MB chunks
        with open(in_path, 'rb') as fin, open(out_path, 'wb') as fout:
            while chunk := fin.read(chunk_size):
                nfc_chunk, _, _ = self.compress(chunk)
                fout.write(nfc_chunk)  # For v2, add global index
    
    def decompress_stream(self, in_path, out_path, chunk_size=1024*1024*64):
        with open(in_path, 'rb') as fin, open(out_path, 'wb') as fout:
            while chunk := fin.read(chunk_size):
                decompressed = self.decompress(chunk)
                fout.write(decompressed)