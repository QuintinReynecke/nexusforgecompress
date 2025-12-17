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
    def __init__(self, clevel=9, shuffle=blosc.SHUFFLE, codec='zstd'):
        self.magic = b'NFC2'
        self.version = 2
        self.hash_algo = 'sha256'
        self.clevel = clevel
        self.shuffle = shuffle
        self.codec = codec
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

    def compress(self, data, force_arithmetic=False, use_prediction=False):
        is_numpy = isinstance(data, np.ndarray)
        original_data_bytes = data.tobytes() if is_numpy else data # Renamed for clarity
        original_hash = hashlib.sha256(original_data_bytes).digest() # Use original_data_bytes for hash
        # print(f"DEBUG: compress - Calculated original_hash: {original_hash.hex()}")

        metadata = self._get_metadata(data)
        
        flags = 0
        payload = original_data_bytes # Default payload
        
        # --- NEW: Prediction Step ---
        if is_numpy and use_prediction:
            metadata["prediction_model"] = "delta_encoding"
            metadata["original_dtype"] = data.dtype.name
            metadata["original_shape"] = list(data.shape)

            # Determine residuals_dtype to handle potential negative differences and store it
            if np.issubdtype(data.dtype, np.unsignedinteger):
                info = np.iinfo(data.dtype)
                if info.bits <= 8:
                    residuals_dtype = np.int16
                elif info.bits <= 16:
                    residuals_dtype = np.int32
                else: # info.bits <= 32 or 64
                    residuals_dtype = np.int64
            elif np.issubdtype(data.dtype, np.floating):
                residuals_dtype = np.float64 # Use float64 for residuals to maintain precision
            else:
                residuals_dtype = data.dtype

            metadata["residuals_dtype"] = np.dtype(residuals_dtype).name # Store the name of the residuals dtype

            residuals = np.empty(data.shape, dtype=residuals_dtype)
            # Handle first element
            if data.size > 0:
                # Cast to residuals_dtype before operations to prevent overflow/underflow
                data_as_residuals_type = data.astype(residuals_dtype)
                residuals.ravel()[0] = data_as_residuals_type.ravel()[0]
                # Calculate differences for the rest
                if data.size > 1:
                    residuals.ravel()[1:] = np.diff(data_as_residuals_type.ravel())
            
            payload = residuals.tobytes() # residuals are now the payload
        # --- END NEW: Prediction Step ---

        # Optional Step 1: Arithmetic Coding
        use_arithmetic = force_arithmetic and ArithmeticCoder is not None
        if use_arithmetic:
            coder = ArithmeticCoder()
            payload = coder.compress(payload)
            flags |= self.ARITHMETIC_CODING_FLAG

        # Step 2: Blosc Compression
        if is_numpy:
            itemsize = data.dtype.itemsize # Default itemsize
            if use_prediction:
                # If prediction is used, the payload is the residuals array, so use its itemsize.
                itemsize = np.dtype(metadata["residuals_dtype"]).itemsize
            compressed = blosc.compress(payload, cname=self.codec, typesize=itemsize, clevel=self.clevel, shuffle=self.shuffle)
        else:
            compressed = blosc.compress(payload, cname=self.codec, clevel=self.clevel, shuffle=self.shuffle)

        metadata["compression_stack"] = ["arithmetic", f"blosc_{self.codec}"] if use_arithmetic else [f"blosc_{self.codec}"]
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
        return nfc_binary, len(original_data_bytes), len(nfc_binary)

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


        
        current_offset = 34
        
        
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

        # --- NEW: Reconstruction Step ---
        if metadata.get("prediction_model") == "delta_encoding":
            original_dtype_name = metadata["original_dtype"]
            original_shape = tuple(metadata["original_shape"])
            residuals_dtype = np.dtype(metadata["residuals_dtype"]) # Use the stored residuals_dtype
            
            # print(f"DEBUG: decompress - original_dtype_name: {original_dtype_name}, original_shape: {original_shape}, residuals_dtype: {residuals_dtype}")

            # Convert final_bytes (decompressed residuals) back to numpy array with its stored dtype
            residuals_array = np.frombuffer(final_bytes, dtype=residuals_dtype)
            # print(f"DEBUG: decompress - residuals_array (first 5): {residuals_array.flatten()[:5]}")
            
            # Perform cumulative sum to reconstruct the original data
            # Use original floating point precision for float types to prevent precision issues from float64 intermediates,
            # int64 for integers to prevent overflow.
            if np.issubdtype(residuals_dtype, np.floating):
                cumsum_dtype = residuals_dtype # Use the residuals_dtype directly
            else:
                cumsum_dtype = np.int64 # For integers, still promote to int64 to prevent overflow
            reconstructed_data = np.cumsum(residuals_array.astype(cumsum_dtype), dtype=cumsum_dtype).reshape(original_shape)
            # print(f"DEBUG: decompress - reconstructed_data (first 5): {reconstructed_data.flatten()[:5]}")
            
            # Ensure the reconstructed data has the original dtype
            # This handles both dtype promotions during cumsum and restores the original type
            final_bytes = reconstructed_data.astype(original_dtype_name, copy=True).tobytes()
        # --- END NEW: Reconstruction Step ---

        decompressed_hash = hashlib.sha256(final_bytes).digest()
        # print(f"DEBUG: decompress - Extracted original_hash: {original_hash.hex()}")
        # print(f"DEBUG: decompress - Calculated decompressed_hash: {decompressed_hash.hex()}")
        # print(f"DEBUG: decompress - original_hash length: {len(original_hash)}, decompressed_hash length: {len(decompressed_hash)}")
        if decompressed_hash != original_hash:
            raise ValueError("Corruption detected! Hash mismatch.")
            
        if metadata.get("format_hint") == "numpy_tensor":
            np_dtype = np.dtype(metadata["dtype"])
            if metadata.get("endianness") and np_dtype.byteorder != metadata["endianness"]:
                np_dtype = np_dtype.newbyteorder(metadata["endianness"])
            return np.frombuffer(final_bytes, dtype=np_dtype).reshape(metadata["shape"])
            
        return final_bytes

    def compress_stream(self, in_path, out_path, chunk_size=1024 * 1024 * 64):
        with open(in_path, 'rb') as fin, open(out_path, 'wb') as fout:
            while chunk := fin.read(chunk_size):
                # For streaming, we don't use arithmetic coding as it's stateful across chunks
                nfc_chunk, _, _ = self.compress(chunk, force_arithmetic=False)
                fout.write(nfc_chunk)

    def decompress_stream(self, in_path, out_path):
        with open(in_path, 'rb') as fin, open(out_path, 'wb') as fout:
            while True:
                # Read enough to determine the size of the next NFC block
                # We need magic (4) + version (1) + flags (1) + reserved (2) + header_len (8) + meta_len (8) + payload_len (8) + hash_len (2) = 34 bytes
                initial_bytes = fin.read(34)
                if not initial_bytes:
                    break # End of file

                if len(initial_bytes) < 34:
                    raise ValueError("Incomplete NFC block header in stream")

                # Parse header fields from initial_bytes
                if initial_bytes[:4] != self.magic:
                    raise ValueError(f"Invalid magic. Expected {self.magic}, got {initial_bytes[:4]} in stream")

                version = initial_bytes[4]
                if version != self.version:
                    raise ValueError(f"Version mismatch. Expected {self.version}, got {version} in stream")
                
                # Extract header lengths
                # header_len is always 34 for v2, but we read it to be consistent with future versions
                header_len_parsed = struct.unpack('!Q', initial_bytes[8:16])[0]
                meta_len = struct.unpack('!Q', initial_bytes[16:24])[0]
                payload_len = struct.unpack('!Q', initial_bytes[24:32])[0]
                hash_len = struct.unpack('!H', initial_bytes[32:34])[0]

                # Calculate total size of the current NFC block
                total_block_size = int(header_len_parsed + meta_len + payload_len + hash_len)

                # Read the rest of the current NFC block
                # This accounts for the 34 bytes already read in initial_bytes
                remaining_to_read = total_block_size - len(initial_bytes)
                if remaining_to_read < 0:
                    raise ValueError("Calculated total block size is smaller than initial header read. This indicates an invalid header.")

                remaining_block_bytes = fin.read(remaining_to_read)
                if len(remaining_block_bytes) != remaining_to_read:
                    raise ValueError("Incomplete NFC block in stream")

                nfc_block = initial_bytes + remaining_block_bytes
                
                decompressed_data = self.decompress(nfc_block)
                fout.write(decompressed_data)