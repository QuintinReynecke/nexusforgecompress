import sys
import os
import numpy as np
import time
import shutil
import subprocess
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nfc_prototype.core import NFCPrototype, NFCLoader

def test_distributed_swarm():
    print("Starting v2.0.0 Distributed Swarm Tests")
    print("=======================================")
    
    # 1. Start TWO local Registry Servers
    p1 = subprocess.Popen([sys.executable, "-m", "nfc_prototype.registry"], env={**os.environ, "FLASK_RUN_PORT": "5001"})
    p2 = subprocess.Popen([sys.executable, "-m", "nfc_prototype.registry"], env={**os.environ, "FLASK_RUN_PORT": "5002"})
    
    url1 = "http://127.0.0.1:5001"
    url2 = "http://127.0.0.1:5002"
    
    # Give them time to start
    time.sleep(3)
    
    try:
        # 2. Create synthetic model with multiple blocks
        print("   - Creating distributed source model...")
        data1 = np.random.rand(100, 100).astype(np.float32) # Block A
        data2 = np.random.rand(100, 100).astype(np.float32) # Block B
        
        # Temp stores for seeders
        store1 = "./seed_store_1"
        store2 = "./seed_store_2"
        if os.path.exists(store1): shutil.rmtree(store1)
        if os.path.exists(store2): shutil.rmtree(store2)
        
        proto1 = NFCPrototype(store_path=store1)
        proto2 = NFCPrototype(store_path=store2)
        
        # Compress parts into separate stores (simulating distributed availability)
        # We cheat: we compress individually to get hashes, then push to different servers
        nfc1, _, _ = proto1.compress(data1, use_store=True)
        nfc2, _, _ = proto2.compress(data2, use_store=True)
        
        # Push Block A to Server 1
        # Extract hash
        h1 = nfc1[34 + proto1._extract_meta_len(nfc1) : 34 + proto1._extract_meta_len(nfc1) + 32].hex()
        import requests
        requests.post(f"{url1}/upload/{h1}", data=proto1.store.get(h1))
        
        # Push Block B to Server 2
        h2 = nfc2[34 + proto2._extract_meta_len(nfc2) : 34 + proto2._extract_meta_len(nfc2) + 32].hex()
        requests.post(f"{url2}/upload/{h2}", data=proto2.store.get(h2))
        
        # 3. Create a Manifest that needs BOTH
        print("   - Creating manifest requiring both servers...")
        # Create a virtual model that points to both hashes
        recipe = {
            "part_on_server_1": {"hash": h1, "metadata": {"shape": [100,100], "dtype": "float32"}},
            "part_on_server_2": {"hash": h2, "metadata": {"shape": [100,100], "dtype": "float32"}}
        }
        # We need a client to create this manifest file. 
        # Ideally we use create_virtual_model, but generate_manifest works on FILE.
        # So let's create the virtual file first.
        client_store = "./client_store_dist"
        if os.path.exists(client_store): shutil.rmtree(client_store)
        client_proto = NFCPrototype(store_path=client_store, remotes=[url1, url2])
        
        client_proto.create_virtual_model(recipe, "distributed.nfc")
        manifest = client_proto.generate_manifest("distributed.nfc")
        
        # 4. Client Prefetch (Swarm)
        print("   - Client prefetching from swarm (Server 1 & 2)...")
        # Ensure client store is empty of these blocks (it is, we just made it)
        assert not client_proto.store.exists(h1)
        assert not client_proto.store.exists(h2)
        
        client_proto.prefetch_manifest(manifest)
        
        # 5. Verify
        print("   - Verifying availability...")
        assert client_proto.store.exists(h1)
        assert client_proto.store.exists(h2)
        print("     ✅ Passed: Blocks fetched from different sources.")
        
        # 6. Verify Data Integrity
        loader = NFCLoader("distributed.nfc", proto=client_proto)
        assert np.allclose(data1, loader["part_on_server_1"])
        assert np.allclose(data2, loader["part_on_server_2"])
        print("     ✅ Passed: Data reconstructed correctly.")

    finally:
        p1.terminate()
        p2.terminate()
        # Cleanup
        for d in ["./seed_store_1", "./seed_store_2", "./client_store_dist"]:
            if os.path.exists(d): shutil.rmtree(d)
        if os.path.exists("distributed.nfc"): os.remove("distributed.nfc")

    print("All v2.0.0 Distributed Tests Passed!")

if __name__ == "__main__":
    test_distributed_swarm()
