import numpy as np
import json
import hashlib
import struct
import blosc
import datetime
import logging
import os

# Initialize logger
logger = logging.getLogger(__name__)

try:
    from neuralcompression.coders import ArithmeticCoder
except ImportError:
    ArithmeticCoder = None

try:
    import torch
except ImportError:
    torch = None

try:
    import safetensors.numpy as st_np
except ImportError:
    st_np = None

from nfc_prototype.accelerators import fast_delta_encode, fast_delta_decode

class NFCStore:
    """
    The Vault: A Content-Addressable Storage system for NFC.
    Stores raw compressed blocks by their SHA256 hash.
    Allows massive deduplication across different models.
    """
    def __init__(self, store_path=None, remotes=None):
        if store_path is None:
            self.store_path = os.path.join(os.path.expanduser("~"), ".nfc_store")
        else:
            self.store_path = store_path
        
        self.blobs_path = os.path.join(self.store_path, "blobs")
        os.makedirs(self.blobs_path, exist_ok=True)
        self.remotes = remotes if remotes else [] # List of base URLs

    def put(self, compressed_data, data_hash):
        """Saves compressed data if it doesn't exist. Returns True if new write."""
        hash_hex = data_hash.hex()
        shard = hash_hex[:2]
        blob_dir = os.path.join(self.blobs_path, shard)
        os.makedirs(blob_dir, exist_ok=True)
        blob_path = os.path.join(blob_dir, hash_hex)
        if os.path.exists(blob_path): return False
        with open(blob_path, "wb") as f: f.write(compressed_data)
        return True

    def exists(self, hash_hex):
        """Checks if a block exists locally."""
        shard = hash_hex[:2]
        blob_path = os.path.join(self.blobs_path, shard, hash_hex)
        return os.path.exists(blob_path)

    def fetch(self, hash_hex):
        """Try to download a missing block from remote registries."""
        import requests
        shard = hash_hex[:2]
        for remote in self.remotes:
            # Expected remote structure: http://registry.com/blobs/ab/abcdef123...
            url = f"{remote.rstrip('/')}/{shard}/{hash_hex}"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    self.put(response.content, bytes.fromhex(hash_hex))
                    return response.content
            except Exception as e:
                logger.warning(f"Failed to fetch from {url}: {e}")
        return None

    def get(self, hash_hex):
        """Retrieves compressed data by hash. Fetches from remotes if missing."""
        shard = hash_hex[:2]
        blob_path = os.path.join(self.blobs_path, shard, hash_hex)
        if os.path.exists(blob_path):
            with open(blob_path, "rb") as f: return f.read()
        
        # If missing, try to fetch from remotes
        return self.fetch(hash_hex)

class NFCLoader:
    """
    Lazy-loading dictionary-like interface for NFC files.
    Allows accessing blocks by name (if available) or index.
    Only decompresses requested blocks.
    """
    def __init__(self, nfc_path, proto=None):
        self.nfc_path = nfc_path
        self.proto = proto if proto else NFCPrototype()
        self.index = self.proto.get_stream_index(nfc_path)
        if self.index is None:
            raise ValueError("File is not indexed. Lazy loading impossible.")
        
        self.names = []
        self.offsets = []
        
        if isinstance(self.index, dict):
            self.offsets = self.index.get("offsets", [])
            self.names = self.index.get("names", [])
        else:
            self.offsets = self.index
            self.names = [str(i) for i in range(len(self.offsets))]
            
        self._name_to_idx = {name: i for i, name in enumerate(self.names)}

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self._name_to_idx:
                raise KeyError(key)
            idx = self._name_to_idx[key]
        else:
            idx = key
            
        return self.proto.decompress_block_by_offset(self.nfc_path, self.offsets[idx])

    def keys(self):
        return self.names

    def __len__(self):
        return len(self.offsets)

class NFCPrototype:
    def __init__(self, clevel=9, shuffle=blosc.SHUFFLE, codec='zstd', store_path=None, remotes=None):
        self.magic = b'NFC2'
        self.version = 2
        self.hash_algo = 'sha256'
        self.clevel = clevel
        self.shuffle = shuffle
        self.codec = codec
        self.ARITHMETIC_CODING_FLAG = 0x01
        self.INDEX_BLOCK_FLAG = 0x02
        self.DEDUP_POINTER_FLAG = 0x04 # New flag for deduplication pointers

        # Initialize the global store with remotes
        self.store = NFCStore(store_path, remotes=remotes)

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

    def _get_metadata(self, data, extra_metadata=None, format_hint=None):
        if isinstance(data, np.ndarray):
            meta = {
                "schema_version": "nfc-0.2",
                "dtype": data.dtype.name,
                "endianness": data.dtype.byteorder,
                "shape": list(data.shape),
                "format_hint": format_hint if format_hint else "numpy_tensor",
                "orig_bytes": data.nbytes,
                "created_by": "nfc-prototype 1.2.0",
                "created_at": datetime.datetime.utcnow().isoformat() + 'Z'
            }
            if np.issubdtype(data.dtype, np.number) and data.size > 0:
                meta["stats"] = {
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "mean": float(np.mean(data))
                }
            if extra_metadata:
                meta.update(extra_metadata)
            return meta
        return {"format_hint": "bytes"}

    def compress(self, data, force_arithmetic=False, use_prediction=False, prediction_strategy='delta', blosc_filters=None, extra_flags=0, auto_tune=False, extra_metadata=None, format_hint=None, use_store=False):
        if torch is not None and isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        is_numpy = isinstance(data, np.ndarray)
        
        if auto_tune and is_numpy:
            sample_size = min(data.size, 10000)
            sample = data.ravel()[:sample_size]
            if np.issubdtype(data.dtype, np.number):
                orig_var = np.var(sample)
                delta_var = np.var(np.diff(sample.astype(np.float64))) if sample.size > 1 else orig_var
                if delta_var < orig_var * 0.8:
                    use_prediction = True
                    prediction_strategy = 'delta'
                else:
                    use_prediction = False

        original_data_bytes = data.tobytes() if is_numpy else data
        original_hash = hashlib.sha256(original_data_bytes).digest()

        metadata = self._get_metadata(data, extra_metadata=extra_metadata, format_hint=format_hint)
        flags = extra_flags
        payload = original_data_bytes
        
        if is_numpy and use_prediction:
            metadata["prediction_strategy"] = prediction_strategy
            if prediction_strategy == 'delta':
                metadata["prediction_model"] = "delta_encoding"
                metadata["original_dtype"] = data.dtype.name
                metadata["original_shape"] = list(data.shape)
                if np.issubdtype(data.dtype, np.unsignedinteger):
                    info = np.iinfo(data.dtype)
                    if info.bits <= 8: residuals_dtype = np.int16
                    elif info.bits <= 16: residuals_dtype = np.int32
                    else: residuals_dtype = np.int64
                elif np.issubdtype(data.dtype, np.floating):
                    residuals_dtype = np.float64
                else:
                    residuals_dtype = data.dtype
                metadata["residuals_dtype"] = np.dtype(residuals_dtype).name
                
                # Use Accelerated Encoder
                residuals = fast_delta_encode(data, residuals_dtype)
                payload = residuals.tobytes()
            elif prediction_strategy == 'none':
                pass
            else:
                raise ValueError(f"Unknown prediction strategy: {prediction_strategy}")

        if force_arithmetic and ArithmeticCoder is not None:
            coder = ArithmeticCoder()
            payload = coder.compress(payload)
            flags |= self.ARITHMETIC_CODING_FLAG

        if blosc_filters is not None:
            metadata["blosc_filters"] = blosc_filters

        blosc_kwargs = {'cname': self.codec, 'clevel': self.clevel, 'shuffle': self.shuffle}
        if is_numpy:
            itemsize = data.dtype.itemsize
            if use_prediction and prediction_strategy == 'delta':
                itemsize = np.dtype(metadata["residuals_dtype"]).itemsize
            blosc_kwargs['typesize'] = itemsize

        filters_arg = blosc_filters if blosc_filters is not None else []
        try:
            compressed = blosc.compress(payload, filters=filters_arg, **blosc_kwargs)
        except TypeError:
            compressed = blosc.compress(payload, **blosc_kwargs)
        
        # --- NEW: Deduplication Logic ---
        if use_store:
            # Hash the COMPRESSED content to identify this unique block
            compressed_hash = hashlib.sha256(compressed).digest()
            # Store it in the vault
            is_new = self.store.put(compressed, compressed_hash)
            
            # Replace payload with the hash pointer
            compressed = compressed_hash # Now just 32 bytes
            flags |= self.DEDUP_POINTER_FLAG
            
            if is_new:
                # Optional: Log or track new blobs
                pass
        # --- END NEW: Deduplication Logic ---

        metadata["compression_stack"] = ["arithmetic", f"blosc_{self.codec}"] if (flags & self.ARITHMETIC_CODING_FLAG) else [f"blosc_{self.codec}"]
        metadata_json = json.dumps(metadata).encode('utf-8')
        
        # Build preliminary header to calculate checksum
        header_pre = (
            self.magic +
            self.version.to_bytes(1, 'big') +
            flags.to_bytes(1, 'big') +
            b'\x00' * 2 + # Placeholder for checksum
            struct.pack('!Q', self.calculated_header_len) +
            struct.pack('!Q', len(metadata_json)) +
            struct.pack('!Q', len(compressed)) +
            struct.pack('!H', len(original_hash))
        )
        
        # Calculate checksum over the header (excluding the checksum field itself)
        import zlib
        header_checksum = zlib.adler32(header_pre[:6] + header_pre[8:]) & 0xFFFF
        
        header = header_pre[:6] + struct.pack('!H', header_checksum) + header_pre[8:]
        
        return header + metadata_json + compressed + original_hash, len(original_data_bytes), len(header + metadata_json + compressed + original_hash)

    def _extract_meta_len(self, nfc_binary):
        return struct.unpack('!Q', nfc_binary[16:24])[0]

    def _extract_hash_len(self, nfc_binary):
        return struct.unpack('!H', nfc_binary[32:34])[0]

    def get_stats(self, nfc_binary):
        meta_len = self._extract_meta_len(nfc_binary)
        metadata = json.loads(nfc_binary[34 : 34 + meta_len].decode('utf-8'))
        return metadata.get("stats", "No stats available")

    def decompress(self, nfc_binary):
        if nfc_binary[:4] != self.magic: raise ValueError("Invalid magic")
        
        # Verify header checksum
        import zlib
        stored_checksum = struct.unpack('!H', nfc_binary[6:8])[0]
        calculated_checksum = zlib.adler32(nfc_binary[:6] + nfc_binary[8:34]) & 0xFFFF
        if stored_checksum != calculated_checksum:
            raise ValueError("Header corruption detected (Checksum mismatch)!")

        version = nfc_binary[4]
        flags = nfc_binary[5]
        was_arithmetic_coded = (flags & self.ARITHMETIC_CODING_FLAG) != 0
        is_deduplicated = (flags & self.DEDUP_POINTER_FLAG) != 0

        meta_len = struct.unpack('!Q', nfc_binary[16:24])[0]
        payload_len = struct.unpack('!Q', nfc_binary[24:32])[0]
        hash_len = struct.unpack('!H', nfc_binary[32:34])[0]

        current_offset = 34
        metadata_json = nfc_binary[current_offset : current_offset + meta_len]
        current_offset += meta_len
        compressed_payload = nfc_binary[current_offset : current_offset + payload_len]
        current_offset += payload_len
        original_hash = nfc_binary[current_offset : current_offset + hash_len]

        metadata = json.loads(metadata_json) if meta_len > 0 else {}
        
        # --- NEW: Deduplication Fetch ---
        if is_deduplicated:
            # compressed_payload is the hash
            block_hash_hex = compressed_payload.hex()
            actual_compressed_data = self.store.get(block_hash_hex)
            if actual_compressed_data is None:
                raise ValueError(f"Missing deduplicated block in store: {block_hash_hex}")
            compressed_payload = actual_compressed_data
        # --- END NEW: Deduplication Fetch ---

        prediction_strategy = metadata.get("prediction_strategy", "none")        
        decompressed_payload = blosc.decompress(compressed_payload)
        
        if was_arithmetic_coded:
            if ArithmeticCoder is None: raise RuntimeError("Arithmetic coding requires neuralcompression")
            coder = ArithmeticCoder()
            final_bytes = coder.decompress(decompressed_payload)
        else:
            final_bytes = decompressed_payload

        if prediction_strategy == "delta":
            original_dtype_name = metadata["original_dtype"]
            original_shape = tuple(metadata["original_shape"])
            residuals_dtype = np.dtype(metadata["residuals_dtype"])
            residuals_array = np.frombuffer(final_bytes, dtype=residuals_dtype)
            
            # Use Accelerated Decoder
            final_bytes = fast_delta_decode(residuals_array, original_shape, original_dtype_name).tobytes()

        # Skip hash check for virtual pointers (since original_hash might be dummy)
        if metadata.get("format_hint") != "pointer":
            if hashlib.sha256(final_bytes).digest() != original_hash: raise ValueError("Corruption detected!")
            
        if metadata.get("format_hint") in ["numpy_tensor", "numpy_diff", "pointer"] and "dtype" in metadata:
            np_dtype = np.dtype(metadata["dtype"])
            if metadata.get("endianness") and np_dtype.byteorder != metadata["endianness"]:
                np_dtype = np_dtype.newbyteorder(metadata["endianness"])
            return np.frombuffer(final_bytes, dtype=np_dtype).reshape(metadata["shape"])
        return final_bytes

    def compress_diff(self, base_tensor, new_tensor, **kwargs):
        if torch is not None:
            if isinstance(base_tensor, torch.Tensor): base_tensor = base_tensor.detach().cpu().numpy()
            if isinstance(new_tensor, torch.Tensor): new_tensor = new_tensor.detach().cpu().numpy()
        if base_tensor.shape != new_tensor.shape: raise ValueError("Shape mismatch")
        residuals = new_tensor.astype(np.float64) - base_tensor.astype(np.float64)
        return self.compress(residuals, format_hint="numpy_diff", extra_metadata={"original_dtype": new_tensor.dtype.name}, **kwargs)

    def decompress_diff(self, base_tensor, nfc_diff):
        if torch is not None and isinstance(base_tensor, torch.Tensor): base_tensor = base_tensor.detach().cpu().numpy()
        residuals = self.decompress(nfc_diff)
        meta_len = self._extract_meta_len(nfc_diff)
        meta = json.loads(nfc_diff[34:34+meta_len].decode('utf-8'))
        target_dtype = meta.get("original_dtype", base_tensor.dtype.name)
        reconstructed = base_tensor.astype(np.float64) + residuals.astype(np.float64)
        return reconstructed.astype(target_dtype)

    def compress_stream(self, in_path, out_path, chunk_size=1024 * 1024 * 64, use_store=False):
        self.compress_stream_parallel(in_path, out_path, chunk_size=chunk_size, max_workers=1, use_store=use_store)

    def compress_stream_parallel(self, in_path, out_path, chunk_size=1024 * 1024 * 64, max_workers=None, use_store=False):
        from concurrent.futures import ThreadPoolExecutor
        block_offsets = []
        def compress_chunk(chunk): return self.compress(chunk, force_arithmetic=False, use_store=use_store)
        with open(in_path, 'rb') as fin, open(out_path, 'wb') as fout:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                while True:
                    chunk = fin.read(chunk_size)
                    if not chunk: break
                    futures.append(executor.submit(compress_chunk, chunk))
                for future in futures:
                    nfc_chunk, _, _ = future.result()
                    block_offsets.append(fout.tell())
                    fout.write(nfc_chunk)
            index_data = json.dumps({"offsets": block_offsets}).encode('utf-8')
            idx_off = fout.tell()
            nfc_idx, _, _ = self.compress(index_data, extra_flags=self.INDEX_BLOCK_FLAG)
            fout.write(nfc_idx)
            fout.write(struct.pack('!Q', idx_off) + b'INDX')

    def get_stream_index(self, nfc_path):
        with open(nfc_path, 'rb') as f:
            f.seek(-12, os.SEEK_END)
            footer = f.read(12)
            if len(footer) < 12 or footer[8:] != b'INDX': return None
            idx_off = struct.unpack('!Q', footer[:8])[0]
            f.seek(idx_off)
            hdr = f.read(34)
            m_len = struct.unpack('!Q', hdr[16:24])[0]
            p_len = struct.unpack('!Q', hdr[24:32])[0]
            h_len = struct.unpack('!H', hdr[32:34])[0]
            f.seek(idx_off)
            idx_blk = f.read(34 + m_len + p_len + h_len)
            return json.loads(self.decompress(idx_blk))

    def decompress_block(self, nfc_path, block_index):
        index = self.get_stream_index(nfc_path)
        if index is None: raise ValueError("File not indexed")
        if isinstance(index, list):
            if block_index >= len(index): raise ValueError("Invalid block index")
            offset = index[block_index]
        elif isinstance(index, dict):
            offsets = index.get("offsets", [])
            if block_index >= len(offsets): raise ValueError("Invalid block index")
            offset = offsets[block_index]
        else: raise ValueError("Unknown index format")
        return self.decompress_block_by_offset(nfc_path, offset)

    def decompress_block_by_offset(self, nfc_path, offset):
        with open(nfc_path, 'rb') as f:
            f.seek(offset)
            hdr = f.read(34)
            m_len = struct.unpack('!Q', hdr[16:24])[0]
            p_len = struct.unpack('!Q', hdr[24:32])[0]
            h_len = struct.unpack('!H', hdr[32:34])[0]
            f.seek(offset)
            return self.decompress(f.read(34 + m_len + p_len + h_len))

    def compress_tensor(self, tensor, **kwargs):
        if torch is None: raise ImportError("torch missing")
        return self.compress(tensor, **kwargs)

    def decompress_tensor(self, nfc_binary, device='cpu'):
        if torch is None: raise ImportError("torch missing")
        return torch.from_numpy(self.decompress(nfc_binary)).to(device)

    def compress_safetensors(self, in_path, out_path, **kwargs):
        if st_np is None: raise ImportError("safetensors missing")
        tensors = st_np.load_file(in_path)
        offs, names = [], []
        with open(out_path, 'wb') as fout:
            for name, arr in tensors.items():
                offs.append(fout.tell())
                nfc, _, _ = self.compress(arr, **kwargs)
                fout.write(nfc)
                names.append(name)
            idx_data = json.dumps({"offsets": offs, "names": names}).encode('utf-8')
            idx_off = fout.tell()
            nfc_idx, _, _ = self.compress(idx_data, extra_flags=self.INDEX_BLOCK_FLAG)
            fout.write(nfc_idx)
            fout.write(struct.pack('!Q', idx_off) + b'INDX')

    def decompress_safetensors(self, in_path, out_path):
        if st_np is None: raise ImportError("safetensors missing")
        idx = self.get_stream_index(in_path)
        t_out = {name: self.decompress_block_by_offset(in_path, off) for off, name in zip(idx["offsets"], idx["names"])}
        st_np.save_file(t_out, out_path)

    def generate_manifest(self, nfc_path):
        """
        Unique: Generates a .nfm (Neural Forge Manifest) for a model.
        A manifest allows sharing model structure without data.
        """
        idx = self.get_stream_index(nfc_path)
        if not idx: raise ValueError("File must be indexed to generate manifest")
        
        manifest = {
            "version": "1.0",
            "source_file": os.path.basename(nfc_path),
            "created_at": datetime.datetime.utcnow().isoformat(),
            "blocks": []
        }
        
        with open(nfc_path, "rb") as f:
            offsets = idx.get("offsets", [])
            names = idx.get("names", [f"block_{i}" for i in range(len(offsets))])
            
            for i, off in enumerate(offsets):
                f.seek(off)
                hdr = f.read(34)
                m_len = struct.unpack('!Q', hdr[16:24])[0]
                p_len = struct.unpack('!Q', hdr[24:32])[0]
                h_len = struct.unpack('!H', hdr[32:34])[0]
                meta = json.loads(f.read(m_len).decode('utf-8'))
                
                # Identify hash
                is_dedup = (hdr[5] & self.DEDUP_POINTER_FLAG) != 0
                if is_dedup:
                    # Pointer block: hash is the payload
                    block_hash = f.read(32).hex()
                else:
                    # Data block: we need the hash of the compressed data for store lookups
                    f.seek(off + 34 + m_len)
                    comp_data = f.read(p_len)
                    block_hash = hashlib.sha256(comp_data).digest().hex()
                
                # Also get the original hash from the footer of the block
                f.seek(off + 34 + m_len + p_len)
                orig_hash_hex = f.read(h_len).hex()

                manifest["blocks"].append({
                    "name": names[i],
                    "hash": block_hash,
                    "metadata": meta,
                    "original_hash": orig_hash_hex
                })
        return manifest

    def rehydrate_from_manifest(self, manifest, out_nfc_path):
        """Reconstructs a full NFC file from a manifest using the local store."""
        recipe = {b["name"]: {"hash": b["hash"], "metadata": b["metadata"], "original_hash": b.get("original_hash")} for b in manifest["blocks"]}
        return self.create_virtual_model(recipe, out_nfc_path)

    def push_model_to_remote(self, nfc_path, remote_url):
        """Uploads all unique blocks of a model to a remote registry."""
        import requests
        manifest = self.generate_manifest(nfc_path)
        for block in manifest["blocks"]:
            h_hex = block["hash"]
            data = self.store.get(h_hex)
            if data:
                url = f"{remote_url.rstrip('/')}/upload/{h_hex}"
                try:
                    r = requests.post(url, data=data, timeout=30)
                    if r.status_code == 200:
                        logger.info(f"Uploaded {h_hex}")
                except Exception as e:
                    logger.error(f"Failed to push {h_hex}: {e}")

    def prefetch_manifest(self, manifest, max_workers=8):
        """
        Unique: 'Swarm' Fetch.
        Ensures all blocks in the manifest exist locally.
        Downloads missing blocks from registered remotes in parallel.
        """
        from concurrent.futures import ThreadPoolExecutor
        
        missing_hashes = []
        for block in manifest["blocks"]:
            h_hex = block["hash"]
            # Check if exists locally without reading data
            if not self.store.exists(h_hex):
                missing_hashes.append(h_hex)
        
        # Deduplicate hashes (multiple layers might point to same weights)
        missing_hashes = list(set(missing_hashes))
        
        if not missing_hashes:
            return # Nothing to do
            
        print(f"Prefetching {len(missing_hashes)} missing blocks from {len(self.store.remotes)} remotes...")
        
        def fetch_task(h_hex):
            return self.store.fetch(h_hex) is not None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(fetch_task, missing_hashes))
            
        success_count = sum(results)
        print(f"Successfully fetched {success_count}/{len(missing_hashes)} blocks.")

    def create_virtual_model(self, recipe, out_path):
        """
        Unique: Creates a 'Virtual Model' (pointer-only NFC file).
        recipe: dict of {tensor_name: {"hash": hash_hex, "metadata": {...}, "original_hash": hash_hex}}
        """
        offs, names = [], []
        with open(out_path, 'wb') as fout:
            for name, info in recipe.items():
                offs.append(fout.tell())
                h_bytes = bytes.fromhex(info["hash"])
                orig_h_bytes = bytes.fromhex(info.get("original_hash", "")) if info.get("original_hash") else b""
                
                nfc_ptr = self._create_pointer_block(h_bytes, name, info["metadata"], orig_h_bytes)
                fout.write(nfc_ptr)
                names.append(name)
            
            idx_data = json.dumps({"offsets": offs, "names": names}).encode('utf-8')
            idx_off = fout.tell()
            nfc_idx, _, _ = self.compress(idx_data, extra_flags=self.INDEX_BLOCK_FLAG)
            fout.write(nfc_idx)
            fout.write(struct.pack('!Q', idx_off) + b'INDX')

    def _create_pointer_block(self, block_hash, name, metadata, original_hash):
        """Internal helper to create a block that only contains a hash pointer."""
        meta = metadata.copy()
        meta.update({"name": name, "format_hint": "pointer"})
        meta_json = json.dumps(meta).encode('utf-8')
        flags = self.INDEX_BLOCK_FLAG | self.DEDUP_POINTER_FLAG
        
        payload = block_hash # 32 bytes
        
        header_pre = (
            self.magic +
            self.version.to_bytes(1, 'big') +
            flags.to_bytes(1, 'big') +
            b'\x00' * 2 +
            struct.pack('!Q', self.calculated_header_len) +
            struct.pack('!Q', len(meta_json)) +
            struct.pack('!Q', len(payload)) +
            struct.pack('!H', len(original_hash))
        )
        import zlib
        chk = zlib.adler32(header_pre[:6] + header_pre[8:]) & 0xFFFF
        header = header_pre[:6] + struct.pack('!H', chk) + header_pre[8:]
        return header + meta_json + payload + original_hash

    def decompress_stream(self, in_path, out_path):
        self.decompress_stream_parallel(in_path, out_path, max_workers=1)

    def decompress_stream_parallel(self, in_path, out_path, max_workers=None):
        from concurrent.futures import ThreadPoolExecutor
        idx = self.get_stream_index(in_path)
        if idx and (max_workers is None or max_workers > 1):
            if isinstance(idx, dict): offsets = idx.get("offsets", [])
            else: offsets = idx
            with open(in_path, 'rb') as fin, open(out_path, 'wb') as fout:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(self.decompress_block_by_offset, in_path, off) for off in offsets]
                    for fut in futures: fout.write(fut.result())
            return
        with open(in_path, 'rb') as fin, open(out_path, 'wb') as fout:
            while True:
                hdr = fin.read(34)
                if not hdr: break
                if len(hdr) < 34:
                    if len(hdr) == 12 and hdr[8:] == b'INDX': break
                    raise ValueError("Truncated")
                m_len = struct.unpack('!Q', hdr[16:24])[0]
                p_len = struct.unpack('!Q', hdr[24:32])[0]
                h_len = struct.unpack('!H', hdr[32:34])[0]
                blk = hdr + fin.read(m_len + p_len + h_len)
                if not (hdr[5] & self.INDEX_BLOCK_FLAG): fout.write(self.decompress(blk))

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="NFC CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    c_p = subparsers.add_parser("compress")
    c_p.add_argument("input"); c_p.add_argument("output"); c_p.add_argument("--workers", type=int, default=4)
    c_p.add_argument("--dedup", action="store_true", help="Use global deduplication store")
    
    d_p = subparsers.add_parser("decompress")
    d_p.add_argument("input"); d_p.add_argument("output"); d_p.add_argument("--workers", type=int, default=4)
    
    s_p = subparsers.add_parser("stats")
    s_p.add_argument("input")
    
    e_p = subparsers.add_parser("explore", help="Launch interactive explorer (TUI)")
    e_p.add_argument("input")

    m_p = subparsers.add_parser("manifest", help="Generate a .nfm manifest")
    m_p.add_argument("input")

    pf_p = subparsers.add_parser("prefetch", help="Download missing blocks from remotes")
    pf_p.add_argument("manifest", help="Input .nfm manifest file")
    pf_p.add_argument("--remotes", help="Comma-separated list of remote URLs")

    args = parser.parse_args()
    
    # Handle remotes argument
    remotes = args.remotes.split(",") if hasattr(args, "remotes") and args.remotes else None
    proto = NFCPrototype(remotes=remotes)
    
    if args.command == "compress": 
        proto.compress_stream_parallel(args.input, args.output, max_workers=args.workers, use_store=args.dedup)
    elif args.command == "decompress": 
        proto.decompress_stream_parallel(args.input, args.output, max_workers=args.workers)
    elif args.command == "stats":
        with open(args.input, "rb") as f:
            hdr = f.read(34)
            m_len = struct.unpack('!Q', hdr[16:24])[0]
            print(json.dumps(json.loads(f.read(m_len)).get("stats", "No stats"), indent=4))
    elif args.command == "explore":
        try:
            from nfc_prototype.explorer import NFCExplorer
            exp = NFCExplorer(args.input)
            exp.run()
        except ImportError:
            print("To use 'explore', install 'rich': pip install rich")
    elif args.command == "manifest":
        manifest = proto.generate_manifest(args.input)
        print(json.dumps(manifest, indent=4))
    elif args.command == "prefetch":
        with open(args.manifest, "r") as f:
            manifest = json.load(f)
        proto.prefetch_manifest(manifest)
    else: parser.print_help()
