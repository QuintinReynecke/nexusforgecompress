import numpy as np
import json
import hashlib
import struct
import blosc

try:
    from neuralcompression.coders import ArithmeticCoder
except ImportError:
    ArithmeticCoder = None

class NFCPrototype:
    def __init__(self, clevel=9, shuffle=blosc.SHUFFLE):
        self.magic = b'NFC2'
        self.version = 2
        self.hash_algo = 'sha256'
        self.clevel = clevel
        self.shuffle = shuffle
        self.ARITHMETIC_CODING_FLAG = 0x01

        self.calculated_header_len = (
            len(self.magic) +
            struct.calcsize('!B') + # version (1 byte)
            struct.calcsize('!B') + # flags (1 byte)
            2 +                     # Reserved (2 bytes)
            struct.calcsize('!Q') + # header_len (8 bytes)
            struct.calcsize('!Q') + # meta_len (8 bytes)
            struct.calcsize('!Q') + # payload_len (8 bytes)
            struct.calcsize('!H')   # hash_len (2 bytes)
        )


    def _get_metadata(self, data):
        if isinstance(data, np.ndarray):
            return {
                "schema_version": "nfc-0.2",
                "dtype": data.dtype.name,
                "endianness": data.dtype.byteorder,
                "shape": list(data.shape),
                "format_hint": "numpy_tensor",
                "orig_bytes": data.nbytes,
                "created_by": "nfc-prototype 0.2.0",
                "created_at": "2025-12-17T00:00:00Z"
            }
        return {"format_hint": "bytes"}

    def compress(self, data, force_arithmetic=False):
        is_numpy = isinstance(data, np.ndarray)
        data_bytes = data.tobytes() if is_numpy else data
        original_hash = hashlib.sha256(data_bytes).digest()

        metadata = self._get_metadata(data)
        
        flags = 0
        payload = data_bytes
        
        # Optional Step 1: Arithmetic Coding
        use_arithmetic = force_arithmetic and ArithmeticCoder is not None
        if use_arithmetic:
            coder = ArithmeticCoder()
            payload = coder.compress(payload)
            flags |= self.ARITHMETIC_CODING_FLAG

        # Step 2: Blosc Compression
        if is_numpy:
            itemsize = data.dtype.itemsize
            compressed = blosc.compress(payload, typesize=itemsize, clevel=self.clevel, shuffle=self.shuffle)
        else:
            compressed = blosc.compress(payload, clevel=self.clevel, shuffle=self.shuffle)

        metadata["compression_stack"] = ["arithmetic", "blosc"] if use_arithmetic else ["blosc"]
        metadata_json = json.dumps(metadata).encode('utf-8')
        header = (
            self.magic +
            self.version.to_bytes(1, 'big') +
            flags.to_bytes(1, 'big') +
            b'\x00' * 2 +  # Reserved
            struct.pack('!Q', self.calculated_header_len) +
            struct.pack('!Q', len(metadata_json)) +
            struct.pack('!Q', len(compressed)) +
            struct.pack('!H', len(original_hash))
        )
        
        nfc_binary = header + metadata_json + compressed + original_hash
        return nfc_binary, len(data_bytes), len(nfc_binary)

    def decompress(self, nfc_binary):
        if nfc_binary[:4] != self.magic:
            raise ValueError(f"Invalid magic. Expected {self.magic}, got {nfc_binary[:4]}")
        
        version = nfc_binary[4]
        if version != self.version:
            raise ValueError(f"Version mismatch. Expected {self.version}, got {version}")
        
        flags = nfc_binary[5]
        was_arithmetic_coded = (flags & self.ARITHMETIC_CODING_FLAG) != 0

        # Read header fields
        header_len = struct.unpack('!Q', nfc_binary[8:16])[0]
        meta_len = struct.unpack('!Q', nfc_binary[16:24])[0]
        payload_len = struct.unpack('!Q', nfc_binary[24:32])[0]
        hash_len = struct.unpack('!H', nfc_binary[32:34])[0]


        
        current_offset = int(header_len) # Should be 34        
        
        
        # Add explicit boundary checks for each slice
        if (current_offset + meta_len) > len(nfc_binary):
            raise ValueError(f"Metadata slice exceeds binary length. Expected to read {meta_len} bytes from offset {current_offset}, but total binary length is {len(nfc_binary)}.")
        metadata_json = nfc_binary[current_offset : current_offset + meta_len]
        current_offset += meta_len
        
        if (current_offset + payload_len) > len(nfc_binary):
            raise ValueError(f"Payload slice exceeds binary length. Expected to read {payload_len} bytes from offset {current_offset}, but total binary length is {len(nfc_binary)}.")
        compressed_payload = nfc_binary[current_offset : current_offset + payload_len]
        current_offset += payload_len
        
        if (current_offset + hash_len) > len(nfc_binary):
            raise ValueError(f"Hash slice exceeds binary length. Expected to read {hash_len} bytes from offset {current_offset}, but total binary length is {len(nfc_binary)}.")
        original_hash = nfc_binary[current_offset : current_offset + hash_len]

        


        metadata = json.loads(metadata_json) if meta_len > 0 else {}        
        # Step 1: Blosc Decompression
        decompressed_payload = blosc.decompress(compressed_payload)
        
        # Optional Step 2: Arithmetic De-coding
        if was_arithmetic_coded:
            if ArithmeticCoder is None:
                raise RuntimeError("File was compressed with arithmetic coding, but 'neuralcompression' is not installed.")
            coder = ArithmeticCoder()
            final_bytes = coder.decompress(decompressed_payload)
        else:
            final_bytes = decompressed_payload

        decompressed_hash = hashlib.sha256(final_bytes).digest()
        if decompressed_hash != original_hash:
            raise ValueError("Corruption detected! Hash mismatch.")
            
        if metadata.get("format_hint") == "numpy_tensor":
            np_dtype = np.dtype(metadata["dtype"])
            if metadata.get("endianness") and metadata["endianness"] != np_dtype.byteorder:
                np_dtype = np_dtype.newbyteorder(metadata["endianness"])
            return np.frombuffer(final_bytes, dtype=np_dtype).reshape(metadata["shape"])
            
        return final_bytes

    def compress_stream(self, in_path, out_path, chunk_size=1024 * 1024 * 64):
        with open(in_path, 'rb') as fin, open(out_path, 'wb') as fout:
            while chunk := fin.read(chunk_size):
                # For streaming, we don't use arithmetic coding as it's stateful across chunks
                nfc_chunk, _, _ = self.compress(chunk, force_arithmetic=False)
                fout.write(nfc_chunk)

    def decompress_stream(self, in_path, out_path, chunk_size=None):
        with open(in_path, 'rb') as fin, open(out_path, 'wb') as fout:
            buffer = fin.read() 
            offset = 0
            while offset < len(buffer):
                if buffer[offset:offset+4] != self.magic:
                    raise ValueError("Invalid magic number in stream")
                
                header_len = struct.unpack('!Q', buffer[offset+8:offset+16])[0]
                meta_len = struct.unpack('!Q', buffer[offset+16:offset+24])[0]
                payload_len = struct.unpack('!Q', buffer[offset+24:offset+32])[0]
                hash_len = struct.unpack('!H', buffer[offset+32:offset+34])[0]
                
                total_len = int(header_len + meta_len + payload_len + hash_len)
                
                if offset + total_len > len(buffer):
                    raise ValueError("Incomplete NFC chunk in stream")

                nfc_chunk = buffer[offset:offset + total_len]
                decompressed = self.decompress(nfc_chunk)
                fout.write(decompressed)
                offset += total_len