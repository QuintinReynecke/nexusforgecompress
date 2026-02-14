import sys
import os
import numpy as np
import time
import shutil
import subprocess
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nfc_prototype.core import NFCPrototype, NFCLoader

def test_registry_sync():
    print("Starting v1.8.0 Neural Registry Tests")
    print("=======================================")
    
    # 1. Start local Registry Server in background
    registry_port = 5001
    registry_url = f"http://127.0.0.1:{registry_port}"
    
    # Clean registry store
    if os.path.exists("./registry_store"): shutil.rmtree("./registry_store")
    
    # Use sys.executable to run the module
    server_process = subprocess.Popen([sys.executable, "-m", "nfc_prototype.registry"], env=os.environ)
    # Give it a moment to start (we'll assume default 5000 for now, oh wait I set 5001 in my mind but script uses 5000)
    # Let's fix the server URL to 5000
    registry_url = "http://127.0.0.1:5000"
    time.sleep(3) # Wait for Flask
    
    try:
        # 2. Setup Client Store
        test_store = "./client_store"
        if os.path.exists(test_store): shutil.rmtree(test_store)
        client_proto = NFCPrototype(store_path=test_store, remotes=[registry_url])
        
        # 3. Create a model and PUSH to registry
        print("   - Creating model and pushing to registry...")
        data_dict = {"layer0": np.random.rand(100, 100).astype(np.float32)}
        import safetensors.numpy as st_np
        st_np.save_file(data_dict, "test_weights.safetensors")
        
        nfc_path = "model_to_push.nfc"
        client_proto.compress_safetensors("test_weights.safetensors", nfc_path, use_store=True)
        
        client_proto.push_model_to_remote(nfc_path, registry_url)
        print("     ✅ Passed: Model blocks pushed to remote.")
        
        # 4. Simulate a NEW client with an EMPTY store but the MANIFEST
        print("   - Simulating new client fetching from registry...")
        manifest = client_proto.generate_manifest(nfc_path)
        
        # New store
        if os.path.exists("./client_store_2"): shutil.rmtree("./client_store_2")
        client_2 = NFCPrototype(store_path="./client_store_2", remotes=[registry_url])
        
        # Rehydrate - this should trigger FETCH from registry!
        client_2.rehydrate_from_manifest(manifest, "rehydrated_from_remote.nfc")
        
        # Verify
        loader = NFCLoader("rehydrated_from_remote.nfc", proto=client_2)
        # Accessing block by name
        fetched_data = loader["layer0"]
        assert np.allclose(data_dict["layer0"], fetched_data)
        print("     ✅ Passed: Model rehydrated by fetching missing blocks from registry.")

    finally:
        server_process.terminate()
        # Cleanup
        for d in ["./registry_store", "./client_store", "./client_store_2"]:
            if os.path.exists(d): shutil.rmtree(d)
        for f in ["model_to_push.nfc", "rehydrated_from_remote.nfc"]:
            if os.path.exists(f): os.remove(f)
            
    print("All v1.8.0 Tests Passed!")

if __name__ == "__main__":
    test_registry_sync()
